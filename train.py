#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurgAtt-Tracker (ICML 2026) — Reference-guided Proposal Reranking + Motion-Aware Refinement
=========================================================================================
Implementation follows the paper’s method:
- Frozen detector D(.) produces Top-K proposals + multi-scale neck features (P3,P4,P5).
- Multi-Scale ROI Decoder (MSR): ROIAlign on P3/P4/P5 + per-level proj + 2D pos enc + sum fuse.
- Attention Score Rerank (AS-Rerank): multi-head cross-attention style dot-product matching
  between reference ROI token and target ROI tokens → rerank logits over K proposals.
- Motion-Aware Adaptive Refine (MAA-Refine): polar-based refinement (theta, d, s_w, s_h)
  conditioned on top-1 proposal token + geometry encoding Geo(B_t, B_r^s).
- Training: temporal gap sampling r=t-n with N={1,2,4,8,16,32}, P={0.4,0.2,0.1,0.1,0.1,0.1}.
- Reference box selection during training: oracle selection on reference frame proposals
  (min center error to GT, restricted to non-zero IoU proposals if available).
- Losses on target frame only:
  L_total = L_rerank + L_refine
  L_rerank = L_ce + 0.1 L_geo + 0.5 L_rank
  L_refine = 0.1 L_dist + L_scale

Notes:
- Label format: YOLO normalized "cls cx cy w h". This implementation uses the first line if multiple.
- This script trains reranker+refiner only. Detector is frozen.
"""

import os
import re
import time
import math
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
from torchvision.ops import roi_align

from ultralytics import YOLO

# ---- ultralytics NMS (fix LOGGER NameError edge case) ----
import ultralytics.utils.ops as uops
from ultralytics.utils import LOGGER as _ULTRA_LOGGER
if not hasattr(uops, "LOGGER"):
    uops.LOGGER = _ULTRA_LOGGER
from ultralytics.utils.ops import non_max_suppression


# =========================================================
# Consts / regex
# =========================================================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_FRAME_RE = re.compile(r"_frame_(\d+)\.(jpg|jpeg|png|bmp|tif|tiff|webp)$", re.IGNORECASE)


# =========================================================
# Utils
# =========================================================
def seed_all(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def frame_id_from_name(name: str) -> int:
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else -1

def yolo_norm_to_xyxy(cx, cy, w, h, img_w, img_h):
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return x1, y1, x2, y2

def diag_norm(input_w: int, input_h: int) -> float:
    return math.sqrt(float(input_w * input_w + input_h * input_h)) + 1e-6

def xyxy_center(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return torch.stack([cx, cy], dim=-1)

def xyxy_wh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    w = (x2 - x1).clamp(min=1e-3)
    h = (y2 - y1).clamp(min=1e-3)
    return torch.stack([w, h], dim=-1)

def xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    c = xyxy_center(xyxy)
    wh = xyxy_wh(xyxy)
    return torch.cat([c, wh], dim=-1)

def clamp_boxes_xyxy(boxes: torch.Tensor, W: int, H: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1 = x1.clamp(0, W - 1)
    y1 = y1.clamp(0, H - 1)
    x2 = x2.clamp(0, W - 1)
    y2 = y2.clamp(0, H - 1)
    x1_, x2_ = torch.minimum(x1, x2), torch.maximum(x1, x2)
    y1_, y2_ = torch.minimum(y1, y2), torch.maximum(y1, y2)
    x2_ = (x1_ + (x2_ - x1_).clamp(min=1.0)).clamp(0, W - 1)
    y2_ = (y1_ + (y2_ - y1_).clamp(min=1.0)).clamp(0, H - 1)
    return torch.stack([x1_, y1_, x2_, y2_], dim=-1)

def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (B,K,4) or (K,4)
    boxes2: (B,4) or (4,)
    returns IoU: (B,K) or (K,)
    """
    if boxes1.dim() == 2:
        b1 = boxes1.unsqueeze(0)   # (1,K,4)
        b2 = boxes2.unsqueeze(0) if boxes2.dim() == 1 else boxes2.unsqueeze(1)  # (1,1,4) or (1,1,4)
        squeeze = True
    else:
        b1 = boxes1
        b2 = boxes2.unsqueeze(1)   # (B,1,4)
        squeeze = False

    x11, y11, x12, y12 = b1.unbind(-1)
    x21, y21, x22, y22 = b2.unbind(-1)

    ix1 = torch.maximum(x11, x21)
    iy1 = torch.maximum(y11, y21)
    ix2 = torch.minimum(x12, x22)
    iy2 = torch.minimum(y12, y22)

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    a1 = ((x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0))
    a2 = ((x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0))

    union = (a1 + a2 - inter).clamp(min=1e-6)
    iou = inter / union
    return iou.squeeze(0) if squeeze else iou


# =========================================================
# Dataset (reference-target pairs with gap sampling)
# =========================================================
def _read_resize_to_tensor(path: str, out_hw: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, out_hw, interpolation=TF.InterpolationMode.BILINEAR)
    return TF.to_tensor(img)  # float32 CHW [0,1]

@dataclass
class RTTSample:
    xr: torch.Tensor
    gtr_xyxy: torch.Tensor
    xt: torch.Tensor
    gtt_xyxy: torch.Tensor

def collate_rtt(batch: List[RTTSample]):
    xr = torch.stack([b.xr for b in batch], dim=0)
    xt = torch.stack([b.xt for b in batch], dim=0)
    gtr = torch.stack([b.gtr_xyxy for b in batch], dim=0)
    gtt = torch.stack([b.gtt_xyxy for b in batch], dim=0)
    return xr, gtr, xt, gtt

class GapRTTDataset(Dataset):
    """
    Reference-target dataset with categorical temporal gap sampling:
      r = t - n,  n ~ Cat(N, P)
    """
    def __init__(
        self,
        img_root: str,
        lbl_root: str,
        input_w: int,
        input_h: int,
        gap_N: List[int] = [1, 2, 4, 8, 16, 32],
        gap_P: List[float] = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1],
        max_tries: int = 8,
    ):
        self.img_root = Path(img_root).resolve()
        self.lbl_root = Path(lbl_root).resolve()
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.max_tries = int(max_tries)

        assert len(gap_N) == len(gap_P) and len(gap_N) > 0
        self.gap_N = [int(x) for x in gap_N]
        p = torch.tensor([float(x) for x in gap_P], dtype=torch.float32)
        p = p / (p.sum() + 1e-9)
        self.gap_P = p.tolist()
        self.max_gap = max(self.gap_N)

        self.videos: List[Tuple[Path, Path, List[Path]]] = []
        self.base_index: List[Tuple[int, int]] = []
        self._build_index()

    def _build_index(self):
        video_dirs = []
        for d in self.img_root.rglob("*"):
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in IMG_EXTS:
                        video_dirs.append(d)
                        break
        video_dirs = sorted(set(video_dirs))
        if not video_dirs:
            raise RuntimeError(f"No video folders under: {self.img_root}")

        for vd in video_dirs:
            rel = vd.relative_to(self.img_root)
            ld = self.lbl_root / rel
            if not ld.exists():
                continue
            imgs = [p for p in vd.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            imgs.sort(key=lambda p: (frame_id_from_name(p.name), p.name))

            usable = []
            for p in imgs:
                lab = (ld / p.name).with_suffix(".txt")
                if lab.exists():
                    usable.append(p)
            if len(usable) <= self.max_gap:
                continue
            self.videos.append((vd, ld, usable))

        if not self.videos:
            raise RuntimeError(f"No usable videos under {self.img_root} + {self.lbl_root}")

        self.base_index = []
        for vi, (_, _, frames) in enumerate(self.videos):
            for t in range(self.max_gap, len(frames)):
                self.base_index.append((vi, t))

        if not self.base_index:
            raise RuntimeError("No base indices; check data or reduce max_gap.")

        print(
            f"[GapRTTDataset] videos={len(self.videos)} samples={len(self.base_index)} "
            f"gap_N={self.gap_N} gap_P={self.gap_P}"
        )

    def __len__(self):
        return len(self.base_index)

    def _read_gt_xyxy(self, img_path: Path, lbl_dir: Path) -> torch.Tensor:
        lab = (lbl_dir / img_path.name).with_suffix(".txt")
        lines = [ln.strip() for ln in lab.read_text().splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(f"Empty label: {lab}")
        s = lines[0].split()
        if len(s) < 5:
            raise RuntimeError(f"Bad label: {lab}")
        cx, cy, w, h = map(float, s[1:5])
        x1, y1, x2, y2 = yolo_norm_to_xyxy(cx, cy, w, h, self.input_w, self.input_h)
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    def _sample_gap(self) -> int:
        r = random.random()
        acc = 0.0
        for n, p in zip(self.gap_N, self.gap_P):
            acc += p
            if r <= acc:
                return n
        return self.gap_N[-1]

    def __getitem__(self, idx: int) -> RTTSample:
        vi, t = self.base_index[idx]
        _, ld, frames = self.videos[vi]

        for _ in range(self.max_tries):
            n = self._sample_gap()
            r = t - n
            if r >= 0:
                pr = frames[r]
                pt = frames[t]
                if (ld / pr.name).with_suffix(".txt").exists() and (ld / pt.name).with_suffix(".txt").exists():
                    gtr = self._read_gt_xyxy(pr, ld)
                    gtt = self._read_gt_xyxy(pt, ld)
                    xr = _read_resize_to_tensor(str(pr), (self.input_h, self.input_w))
                    xt = _read_resize_to_tensor(str(pt), (self.input_h, self.input_w))
                    return RTTSample(xr=xr, gtr_xyxy=gtr, xt=xt, gtt_xyxy=gtt)

        # fallback to smallest gap
        n = min(self.gap_N)
        r = t - n
        pr = frames[r]
        pt = frames[t]
        gtr = self._read_gt_xyxy(pr, ld)
        gtt = self._read_gt_xyxy(pt, ld)
        xr = _read_resize_to_tensor(str(pr), (self.input_h, self.input_w))
        xt = _read_resize_to_tensor(str(pt), (self.input_h, self.input_w))
        return RTTSample(xr=xr, gtr_xyxy=gtr, xt=xt, gtt_xyxy=gtt)


# =========================================================
# YOLO feature hooks (Frozen detector)
# =========================================================
class FeatureHook:
    def __init__(self):
        self.feats = {}
        self.handles = []

    def register(self, det_model, p3_idx: int, p4_idx: int, p5_idx: int):
        def _mk(name):
            def _fn(m, inp, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                self.feats[name] = out
            return _fn
        self.handles.append(det_model.model[p3_idx].register_forward_hook(_mk("p3")))
        self.handles.append(det_model.model[p4_idx].register_forward_hook(_mk("p4")))
        self.handles.append(det_model.model[p5_idx].register_forward_hook(_mk("p5")))

    def clear(self):
        self.feats = {}

    def pop(self):
        out = self.feats
        self.feats = {}
        return out

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

@torch.no_grad()
def yolo_forward_tensor(
    *,
    det_model,
    hook,
    x: torch.Tensor,          # (B,3,H,W) on device
    conf_thres: float,
    iou_thres: float,
    k: int,
    feats_force_fp32: bool = True,
):
    """
    Returns:
      feats: dict {p3,p4,p5} each (B,C,Hf,Wf) float32 on device (detached)
      boxes: (B,K,4) float32 xyxy
      confs: (B,K) float32
      valid: (B,K) bool
    """
    model_dtype = next(det_model.parameters()).dtype
    x = x.to(dtype=model_dtype)

    hook.clear()
    preds = det_model(x)  # triggers hooks
    feats = hook.pop()

    for kk in feats:
        ft = feats[kk].detach()
        if feats_force_fp32:
            ft = ft.float()
        feats[kk] = ft

    dets = non_max_suppression(
        preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=None,
        agnostic=True,
        max_det=k,
    )

    B = x.shape[0]
    boxes = torch.zeros((B, k, 4), device=x.device, dtype=torch.float32)
    confs = torch.zeros((B, k), device=x.device, dtype=torch.float32)
    valid = torch.zeros((B, k), device=x.device, dtype=torch.bool)

    for i, di in enumerate(dets):
        if di is None or len(di) == 0:
            continue
        n = min(k, di.shape[0])
        boxes[i, :n] = di[:n, :4].float()
        confs[i, :n] = di[:n, 4].float()
        valid[i, :n] = True

    return feats, boxes, confs, valid


# =========================================================
# Model: MSR + AS-Rerank + MAA-Refine
# =========================================================
def _boxes_to_roi_format(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """
    boxes_xyxy: (B,K,4) in input pixel coords
    returns rois: (B*K, 5) with (batch_idx, x1,y1,x2,y2)
    """
    B, K, _ = boxes_xyxy.shape
    device = boxes_xyxy.device
    idx = torch.arange(B, device=device).view(B, 1).expand(B, K).reshape(-1).float()
    rois = torch.cat([idx.unsqueeze(1), boxes_xyxy.reshape(-1, 4)], dim=1)
    return rois

class Sine2DPosEnc(nn.Module):
    """
    Lightweight 2D sine-cos positional encoding for feature maps.
    Adds position to projected feature maps (B,D,H,W).
    """
    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sine-cos"
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,D,H,W)
        B, D, H, W = x.shape
        device = x.device
        d_half = D // 2
        d_quarter = D // 4

        y_embed = torch.linspace(0, 1, steps=H, device=device).view(H, 1).expand(H, W)
        x_embed = torch.linspace(0, 1, steps=W, device=device).view(1, W).expand(H, W)

        dim_t = torch.arange(d_quarter, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / d_quarter)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)  # (H,W,d_quarter)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)  # (H,W,d_quarter)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H,W,2*d_quarter)= (H,W,D/2)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, d_half, H, W)  # (B,D/2,H,W)

        # If D/2 < D, pad zeros to match D
        if d_half < D:
            pad = torch.zeros((B, D - d_half, H, W), device=device, dtype=pos.dtype)
            pos = torch.cat([pos, pad], dim=1)

        return x + pos

class MultiScaleROIEmbedder(nn.Module):
    """
    MSR: ROIAlign on P3/P4/P5, per-level 1x1 proj -> D, add 2D pos enc, ROIAlign,
         GAP -> D, fuse by sum, then token MLP.
    """
    def __init__(self, p3_in: int, p4_in: int, p5_in: int, d_model: int = 256, pool: int = 3):
        super().__init__()
        self.d_model = int(d_model)
        self.pool = int(pool)

        self.p3_proj = nn.Conv2d(p3_in, d_model, 1)
        self.p4_proj = nn.Conv2d(p4_in, d_model, 1)
        self.p5_proj = nn.Conv2d(p5_in, d_model, 1)

        self.pos = Sine2DPosEnc(d_model)

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _spatial_scale(feat: torch.Tensor, input_w: int, input_h: int) -> float:
        Hf, Wf = feat.shape[-2:]
        sx = float(Wf) / float(input_w + 1e-6)
        sy = float(Hf) / float(input_h + 1e-6)
        return float(min(sx, sy))

    def forward(self, feats: Dict[str, torch.Tensor], boxes_xyxy: torch.Tensor, input_w: int, input_h: int) -> torch.Tensor:
        """
        feats: dict p3/p4/p5 each (B,C,Hf,Wf) float32
        boxes_xyxy: (B,K,4) float32 in input pixels
        returns: tokens (B,K,D)
        """
        B, K, _ = boxes_xyxy.shape
        rois = _boxes_to_roi_format(boxes_xyxy)  # (B*K,5)

        p3 = self.pos(self.p3_proj(feats["p3"]))
        p4 = self.pos(self.p4_proj(feats["p4"]))
        p5 = self.pos(self.p5_proj(feats["p5"]))

        s3 = self._spatial_scale(p3, input_w, input_h)
        s4 = self._spatial_scale(p4, input_w, input_h)
        s5 = self._spatial_scale(p5, input_w, input_h)

        r3 = roi_align(p3, rois, output_size=self.pool, spatial_scale=s3, sampling_ratio=2, aligned=True)
        r4 = roi_align(p4, rois, output_size=self.pool, spatial_scale=s4, sampling_ratio=2, aligned=True)
        r5 = roi_align(p5, rois, output_size=self.pool, spatial_scale=s5, sampling_ratio=2, aligned=True)

        r3 = r3.mean(dim=(-1, -2))
        r4 = r4.mean(dim=(-1, -2))
        r5 = r5.mean(dim=(-1, -2))

        tok = r3 + r4 + r5
        tok = self.token_mlp(tok)
        return tok.view(B, K, self.d_model).contiguous()

class ASRerank(nn.Module):
    """
    AS-Rerank: multi-head dot-product (cross-attention style) between reference token and K target tokens.
    Outputs rerank logits (B,K).
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ref_tok: torch.Tensor, tgt_toks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        ref_tok: (B,D)
        tgt_toks: (B,K,D)
        valid: (B,K)
        returns logits: (B,K) with invalid masked to -inf
        """
        B, K, D = tgt_toks.shape
        q = self.q_proj(ref_tok).view(B, self.n_heads, self.d_head)                  # (B,H,d)
        k = self.k_proj(tgt_toks).view(B, K, self.n_heads, self.d_head)              # (B,K,H,d)

        logits = torch.einsum("bhd,bkhd->bkh", q, k) / math.sqrt(float(self.d_head)) # (B,K,H)
        logits = self.dropout(logits)
        scores = logits.mean(dim=2)                                                 # (B,K)

        minv = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid, minv)
        return scores

class MAARefineHead(nn.Module):
    """
    MAA-Refine: polar-based refinement conditioned on top-1 token and explicit geometry Geo(B_t, B_r^s).

    Predict:
      theta (rad), d (pixels), log_sw, log_sh
    Update:
      (cx',cy') = (cx,cy) + d*(cos(theta), sin(theta))
      (w',h')   = (w,h) * (exp(log_sw), exp(log_sh))
    """
    def __init__(self, d_model: int = 256, geo_dim: int = 10, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.in_dim = d_model + geo_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, 4)  # theta, d_raw, log_sw, log_sh

    def forward(self, tok: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        x = torch.cat([tok, geo], dim=-1)
        h = self.mlp(x)
        return self.out(h)

class SurgAttTracker(nn.Module):
    def __init__(
        self,
        input_w: int,
        input_h: int,
        k: int,
        p3_in: int,
        p4_in: int,
        p5_in: int,
        d_model: int = 256,
        n_heads: int = 8,
        pool: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.k = int(k)

        self.msr = MultiScaleROIEmbedder(p3_in=p3_in, p4_in=p4_in, p5_in=p5_in, d_model=d_model, pool=pool)
        self.rerank = ASRerank(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.refine = MAARefineHead(d_model=d_model, geo_dim=10, hidden=d_model, dropout=dropout)

    @staticmethod
    def make_geo(
        boxes_xyxy: torch.Tensor,      # (B,K,4) or (B,4)
        ref_xyxy: torch.Tensor,        # (B,4)
        input_w: int,
        input_h: int,
    ) -> torch.Tensor:
        """
        Geo encoding:
        [cx/W, cy/H, w/W, h/H,
         d_cx/W, d_cy/H, log(w/wr), log(h/hr), log(ar/ar_r),
         area_norm]
        """
        if boxes_xyxy.dim() == 2:
            boxes_xyxy = boxes_xyxy.unsqueeze(1)  # (B,1,4)

        B, K, _ = boxes_xyxy.shape
        cxcywh = xyxy_to_cxcywh(boxes_xyxy)
        c = cxcywh[..., 0:2]
        wh = cxcywh[..., 2:4].clamp(min=1e-3)

        ref_cxcywh = xyxy_to_cxcywh(ref_xyxy)     # (B,4)
        rc = ref_cxcywh[:, 0:2].unsqueeze(1)
        rwh = ref_cxcywh[:, 2:4].unsqueeze(1).clamp(min=1e-3)

        d = (c - rc)
        dcx = d[..., 0] / (input_w + 1e-6)
        dcy = d[..., 1] / (input_h + 1e-6)

        logw = torch.log(wh[..., 0] / rwh[..., 0]).clamp(-4, 4)
        logh = torch.log(wh[..., 1] / rwh[..., 1]).clamp(-4, 4)

        ar = (wh[..., 0] / wh[..., 1].clamp(min=1e-3)).clamp(1e-3, 1e3)
        ar_r = (rwh[..., 0] / rwh[..., 1].clamp(min=1e-3)).clamp(1e-3, 1e3)
        logar = torch.log(ar / ar_r).clamp(-4, 4)

        area = (wh[..., 0] * wh[..., 1]).clamp(min=1.0)
        area_norm = torch.log(area) / math.log(float(input_w * input_h) + 1e-6)

        geo = torch.stack([
            c[..., 0] / (input_w + 1e-6),
            c[..., 1] / (input_h + 1e-6),
            wh[..., 0] / (input_w + 1e-6),
            wh[..., 1] / (input_h + 1e-6),
            dcx,
            dcy,
            logw,
            logh,
            logar,
            area_norm,
        ], dim=-1)
        return geo  # (B,K,10)

    def extract_tokens(self, feats: Dict[str, torch.Tensor], boxes_xyxy: torch.Tensor) -> torch.Tensor:
        return self.msr(feats, boxes_xyxy, self.input_w, self.input_h)

    def forward_train(
        self,
        feats_r: Dict[str, torch.Tensor],
        boxes_r: torch.Tensor,  # (B,K,4)
        valid_r: torch.Tensor,  # (B,K)
        gtr: torch.Tensor,      # (B,4)
        feats_t: Dict[str, torch.Tensor],
        boxes_t: torch.Tensor,  # (B,K,4)
        valid_t: torch.Tensor,  # (B,K)
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward for one (reference, target) pair.
        Returns dict:
          scores: (B,K) rerank logits (masked)
          ref_xyxy: (B,4) selected reference box
          top1_xyxy: (B,4) selected target proposal box (before refine)
          refined_xyxy: (B,4) refined target box
          idx_star: (B,) oracle best-1 index in target proposals (for losses)
          topM_idx: (B,M) oracle closest M indices (for listwise)
          valid_mask_t: (B,K) valid proposals used in losses (non-zero IoU if available)
        """
        B, K, _ = boxes_t.shape
        device = boxes_t.device

        # -------- reference box oracle selection (training) --------
        iou_r = box_iou_xyxy(boxes_r, gtr)  # (B,K)
        valid_iou_r = valid_r & (iou_r > 0)
        use_iou_r = valid_iou_r.any(dim=1, keepdim=True)
        cand_r = torch.where(use_iou_r, valid_iou_r, valid_r)  # (B,K)

        cr = xyxy_center(boxes_r)              # (B,K,2)
        gcr = xyxy_center(gtr).unsqueeze(1)    # (B,1,2)
        dr = torch.norm(cr - gcr, dim=-1)      # (B,K)
        dr = dr.masked_fill(~cand_r, 1e9)
        idx_ref = dr.argmin(dim=1).clamp(0, K - 1)
        ref_xyxy = boxes_r[torch.arange(B, device=device), idx_ref]  # (B,4)

        # -------- tokens --------
        ref_tok = self.extract_tokens(feats_r, ref_xyxy.unsqueeze(1)).squeeze(1)  # (B,D)
        tgt_toks = self.extract_tokens(feats_t, boxes_t)                          # (B,K,D)

        # -------- rerank --------
        scores = self.rerank(ref_tok, tgt_toks, valid_t)                          # (B,K)

        # -------- select top-1 proposal in target --------
        idx_top1 = scores.masked_fill(~valid_t, -1e30).argmax(dim=1)
        top1_xyxy = boxes_t[torch.arange(B, device=device), idx_top1]             # (B,4)
        top1_tok = tgt_toks[torch.arange(B, device=device), idx_top1]             # (B,D)

        # -------- refine (MAA-Refine) --------
        geo_top1 = self.make_geo(top1_xyxy, ref_xyxy, self.input_w, self.input_h).squeeze(1)  # (B,10)
        pred = self.refine(top1_tok, geo_top1)  # (B,4)

        theta = pred[:, 0]
        d = F.softplus(pred[:, 1])  # positive magnitude in pixels
        log_sw = pred[:, 2].clamp(-2.0, 2.0)
        log_sh = pred[:, 3].clamp(-2.0, 2.0)

        top1_cxcywh = xyxy_to_cxcywh(top1_xyxy)    # (B,4)
        cx, cy = top1_cxcywh[:, 0], top1_cxcywh[:, 1]
        w, h = top1_cxcywh[:, 2].clamp(min=1e-3), top1_cxcywh[:, 3].clamp(min=1e-3)

        cx_r = cx + d * torch.cos(theta)
        cy_r = cy + d * torch.sin(theta)
        w_r = w * torch.exp(log_sw)
        h_r = h * torch.exp(log_sh)

        refined_cxcywh = torch.stack([cx_r, cy_r, w_r, h_r], dim=-1)
        # cxcywh -> xyxy
        rx1 = refined_cxcywh[:, 0] - 0.5 * refined_cxcywh[:, 2]
        ry1 = refined_cxcywh[:, 1] - 0.5 * refined_cxcywh[:, 3]
        rx2 = refined_cxcywh[:, 0] + 0.5 * refined_cxcywh[:, 2]
        ry2 = refined_cxcywh[:, 1] + 0.5 * refined_cxcywh[:, 3]
        refined_xyxy = clamp_boxes_xyxy(torch.stack([rx1, ry1, rx2, ry2], dim=-1), self.input_w, self.input_h)

        # -------- oracle target indices for losses --------
        # valid proposals for loss: non-zero IoU with GT if available, else all valid
        iou_t = box_iou_xyxy(boxes_t, None if False else torch.zeros_like(gtr))  # placeholder

        iou_t = box_iou_xyxy(boxes_t, torch.zeros_like(gtr))  # overwritten below

        # compute IoU properly (B,K) with GT target
        # (avoid re-alloc by direct call)
        # NOTE: we don't have gtt here in forward_train signature; pass in via losses.
        # We'll compute valid_mask_t externally in loss for correctness.

        out = {
            "scores": scores,
            "ref_xyxy": ref_xyxy,
            "top1_xyxy": top1_xyxy,
            "refined_xyxy": refined_xyxy,
            "idx_ref": idx_ref,
            "idx_top1": idx_top1,
        }
        return out


# =========================================================
# Losses (paper-consistent)
# =========================================================
def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, temp: float) -> torch.Tensor:
    minv = torch.finfo(scores.dtype).min
    s = scores.masked_fill(~mask, minv)
    w = F.softmax(s / max(temp, 1e-6), dim=1)
    w = w * mask.float()
    w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
    return w

def smooth_l1(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(x, torch.zeros_like(x), beta=beta, reduction="none")

@torch.no_grad()
def oracle_best_and_topM_by_center(
    boxes_xyxy: torch.Tensor, valid_mask: torch.Tensor, gt_xyxy: torch.Tensor, M: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      best1_idx: (B,)
      topM_idx: (B,M) padded with -1 if not enough
      eps_topM: (B,M) center errors (pixels) for teacher distribution (inf where invalid)
    """
    B, K, _ = boxes_xyxy.shape
    c = xyxy_center(boxes_xyxy)                  # (B,K,2)
    gt_c = xyxy_center(gt_xyxy).unsqueeze(1)     # (B,1,2)
    d = torch.norm(c - gt_c, dim=-1)             # (B,K)
    d = d.masked_fill(~valid_mask, 1e9)

    best1 = d.argmin(dim=1).clamp(0, K - 1)      # (B,)

    m = min(M, K)
    vals, idx = torch.topk(d, k=m, dim=1, largest=False, sorted=True)  # (B,m)

    topM = torch.full((B, M), -1, device=boxes_xyxy.device, dtype=torch.long)
    epsM = torch.full((B, M), float("inf"), device=boxes_xyxy.device, dtype=torch.float32)
    topM[:, :m] = idx
    epsM[:, :m] = vals.float()

    # invalidate huge distances
    bad = epsM >= 1e8
    topM[bad] = -1
    epsM[bad] = float("inf")

    return best1, topM, epsM

def loss_rerank_ce(scores: torch.Tensor, best1_idx: torch.Tensor, valid_any: torch.Tensor) -> torch.Tensor:
    if not valid_any.any():
        return scores.new_tensor(0.0)
    s = scores[valid_any]
    t = best1_idx[valid_any]
    return F.cross_entropy(s, t, reduction="mean")

def loss_rerank_geo(scores: torch.Tensor, boxes_xyxy: torch.Tensor, valid_mask: torch.Tensor, gt_xyxy: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Soft geometric regularization:
      pi = softmax(scores/tau) over valid_mask
      soft_box = sum pi * cxcywh
      loss = Huber( center(soft_box) - center(GT) )
    """
    w = masked_softmax(scores, valid_mask, temp=tau)   # (B,K)
    cxcywh = xyxy_to_cxcywh(boxes_xyxy)               # (B,K,4)
    soft = (w.unsqueeze(-1) * cxcywh).sum(dim=1)      # (B,4)

    c_pred = soft[:, 0:2]
    c_gt = xyxy_center(gt_xyxy)
    delta = (c_pred - c_gt)
    hub = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=1.0, reduction="mean")
    return hub

def loss_rerank_listwise_topM(scores: torch.Tensor, topM_idx: torch.Tensor, eps_topM: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Listwise top-M ranking loss:
      q_i ∝ exp(-eps_i / sigma) over i in topM
      p_i = softmax(scores_i) over i in topM
      L = - sum q_i log p_i
    """
    B, K = scores.shape
    M = topM_idx.shape[1]
    device = scores.device

    # gather logits for topM, mask invalid (-1)
    idx = topM_idx.clamp(min=0)
    sM = scores.gather(1, idx)  # (B,M)
    validM = (topM_idx >= 0) & torch.isfinite(eps_topM)

    # if a row has no valid topM entries -> ignore
    keep = validM.any(dim=1)
    if not keep.any():
        return scores.new_tensor(0.0)

    sM = sM[keep]
    eM = eps_topM[keep]
    vM = validM[keep]

    # teacher q
    q = torch.exp(-eM / max(sigma, 1e-6))
    q = q * vM.float()
    q = q / (q.sum(dim=1, keepdim=True) + 1e-9)

    # student p
    minv = torch.finfo(sM.dtype).min
    sM_masked = sM.masked_fill(~vM, minv)
    p = F.softmax(sM_masked, dim=1)

    # cross entropy
    loss = -(q * torch.log(p.clamp(min=1e-12))).sum(dim=1)
    return loss.mean()

def loss_refine_dist(refined_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, input_w: int, input_h: int, beta: float = 0.02) -> torch.Tensor:
    diag = diag_norm(input_w, input_h)
    c_pred = xyxy_center(refined_xyxy)
    c_gt = xyxy_center(gt_xyxy)
    delta = (c_pred - c_gt) / diag
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=beta, reduction="mean")

def loss_refine_scale(refined_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, beta: float = 0.02) -> torch.Tensor:
    wh_pred = xyxy_wh(refined_xyxy).clamp(min=1e-3)
    wh_gt = xyxy_wh(gt_xyxy).clamp(min=1e-3)
    d = torch.log(wh_pred) - torch.log(wh_gt)
    return F.smooth_l1_loss(d, torch.zeros_like(d), beta=beta, reduction="mean")


# =========================================================
# Train
# =========================================================
def main():
    ap = argparse.ArgumentParser("SurgAtt-Tracker (ICML 2026) Training")

    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--lbl_root", type=str, required=True)

    ap.add_argument("--input_h", type=int, default=384)
    ap.add_argument("--input_w", type=int, default=640)

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)

    ap.add_argument("--half", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_dir", type=str, default="./ckpt_surgatt_tracker")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=50)

    # hook indices (YOLO)
    ap.add_argument("--p3_idx", type=int, default=14)
    ap.add_argument("--p4_idx", type=int, default=17)
    ap.add_argument("--p5_idx", type=int, default=20)

    # feature channels
    ap.add_argument("--p3_in", type=int, default=256)
    ap.add_argument("--p4_in", type=int, default=512)
    ap.add_argument("--p5_in", type=int, default=512)

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--pool", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)

    # gap sampling (fixed to paper defaults unless overridden)
    ap.add_argument("--gap_N", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--gap_P", type=str, default="0.4,0.2,0.1,0.1,0.1,0.1")

    # rerank loss hyperparams
    ap.add_argument("--tau", type=float, default=0.15)      # soft geometric temperature
    ap.add_argument("--sigma", type=float, default=15.0)    # teacher distribution scale (pixels)
    ap.add_argument("--M", type=int, default=5)             # top-M listwise
    ap.add_argument("--lambda_geo", type=float, default=0.1)
    ap.add_argument("--lambda_rank", type=float, default=0.5)

    # refine loss hyperparams
    ap.add_argument("--lambda_dist", type=float, default=0.1)
    ap.add_argument("--huber_beta", type=float, default=0.02)

    args = ap.parse_args()
    seed_all(args.seed)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gap_N = [int(x) for x in args.gap_N.split(",") if x.strip()]
    gap_P = [float(x) for x in args.gap_P.split(",") if x.strip()]
    assert len(gap_N) == len(gap_P)

    # ---- YOLO frozen ----
    yolo = YOLO(args.weights)
    try:
        yolo.fuse()
    except Exception:
        pass

    det_model = yolo.model
    det_model.eval()
    for p in det_model.parameters():
        p.requires_grad_(False)
    det_model.to(device)

    use_half = bool(args.half) and (device.type == "cuda")
    det_model.half() if use_half else det_model.float()

    hook = FeatureHook()
    hook.register(det_model, p3_idx=args.p3_idx, p4_idx=args.p4_idx, p5_idx=args.p5_idx)

    # ---- dataset/loader ----
    ds = GapRTTDataset(
        img_root=args.img_root,
        lbl_root=args.lbl_root,
        input_w=args.input_w,
        input_h=args.input_h,
        gap_N=gap_N,
        gap_P=gap_P,
        max_tries=8,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_rtt,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )

    # ---- model ----
    model = SurgAttTracker(
        input_w=args.input_w,
        input_h=args.input_h,
        k=args.k,
        p3_in=args.p3_in, p4_in=args.p4_in, p5_in=args.p5_in,
        d_model=args.d_model,
        n_heads=args.n_heads,
        pool=args.pool,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    from tqdm import tqdm
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_t0 = time.time()

        pbar = tqdm(dl, total=len(dl), dynamic_ncols=True, desc=f"Epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        running_n = 0

        for (xr, gtr, xt, gtt) in pbar:
            global_step += 1

            xr = xr.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)
            gtr = gtr.to(device, non_blocking=True)
            gtt = gtt.to(device, non_blocking=True)

            B = xr.shape[0]
            x = torch.cat([xr, xt], dim=0)  # (2B,3,H,W)

            # ---- detector forward (frozen) ----
            with torch.no_grad():
                feats, boxes, confs, valid = yolo_forward_tensor(
                    det_model=det_model,
                    hook=hook,
                    x=x,
                    conf_thres=args.conf,
                    iou_thres=args.iou,
                    k=args.k,
                    feats_force_fp32=True,
                )

            feats_r = {k: v[:B] for k, v in feats.items()}
            feats_t = {k: v[B:] for k, v in feats.items()}
            boxes_r, boxes_t = boxes[:B], boxes[B:]
            valid_r, valid_t = valid[:B], valid[B:]

            any_r = valid_r.any(dim=1)
            any_t = valid_t.any(dim=1)
            keep = any_r & any_t
            if not keep.any():
                continue

            feats_r = {k: v[keep] for k, v in feats_r.items()}
            feats_t = {k: v[keep] for k, v in feats_t.items()}
            boxes_r = boxes_r[keep]
            boxes_t = boxes_t[keep]
            valid_r = valid_r[keep]
            valid_t = valid_t[keep]
            gtr_k = gtr[keep]
            gtt_k = gtt[keep]
            Bk = boxes_r.shape[0]

            # ---- forward ----
            out = model.forward_train(
                feats_r=feats_r,
                boxes_r=boxes_r,
                valid_r=valid_r,
                gtr=gtr_k,
                feats_t=feats_t,
                boxes_t=boxes_t,
                valid_t=valid_t,
            )
            scores = out["scores"]             # (Bk,K)
            refined_xyxy = out["refined_xyxy"] # (Bk,4)

            # ---- build valid_mask_t for losses (non-zero IoU if available else valid) ----
            iou_t = box_iou_xyxy(boxes_t, gtt_k)          # (Bk,K)
            valid_iou_t = valid_t & (iou_t > 0)
            use_iou_t = valid_iou_t.any(dim=1, keepdim=True)
            valid_mask_t = torch.where(use_iou_t, valid_iou_t, valid_t)

            valid_any = valid_mask_t.any(dim=1)

            # ---- oracle indices for rerank losses ----
            best1_idx, topM_idx, eps_topM = oracle_best_and_topM_by_center(
                boxes_xyxy=boxes_t,
                valid_mask=valid_mask_t,
                gt_xyxy=gtt_k,
                M=int(args.M),
            )

            # ---- rerank losses ----
            L_ce = loss_rerank_ce(scores, best1_idx, valid_any)
            L_geo = loss_rerank_geo(scores, boxes_t, valid_mask_t, gtt_k, tau=float(args.tau))
            L_rank = loss_rerank_listwise_topM(scores, topM_idx, eps_topM, sigma=float(args.sigma))
            L_rerank = L_ce + float(args.lambda_geo) * L_geo + float(args.lambda_rank) * L_rank

            # ---- refine losses ----
            L_dist = loss_refine_dist(refined_xyxy, gtt_k, args.input_w, args.input_h, beta=float(args.huber_beta))
            L_scale = loss_refine_scale(refined_xyxy, gtt_k, beta=float(args.huber_beta))
            L_refine = float(args.lambda_dist) * L_dist + L_scale

            loss = L_rerank + L_refine

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            running_loss += float(loss.item())
            running_n += 1

            pbar.set_postfix(
                loss=f"{running_loss/max(running_n,1):.3f}",
                Lr=f"{float(L_rerank.item()):.3f}",
                Lf=f"{float(L_refine.item()):.3f}",
                ce=f"{float(L_ce.item()):.3f}",
                geo=f"{float(L_geo.item()):.3f}",
                rank=f"{float(L_rank.item()):.3f}",
                dist=f"{float(L_dist.item()):.3f}",
                scale=f"{float(L_scale.item()):.3f}",
            )

            if (global_step % args.log_every) == 0:
                with torch.no_grad():
                    idx_sel = scores.masked_fill(~valid_t, -1e30).argmax(dim=1)
                    sel_xyxy = boxes_t[torch.arange(Bk, device=device), idx_sel]
                    ce_px = torch.norm(xyxy_center(sel_xyxy) - xyxy_center(gtt_k), dim=1).mean().item()
                    ce_refined_px = torch.norm(xyxy_center(refined_xyxy) - xyxy_center(gtt_k), dim=1).mean().item()

                print(
                    f"[E{epoch:02d} step={global_step}] "
                    f"loss={running_loss/max(running_n,1):.4f} "
                    f"L_rerank={float(L_rerank.item()):.4f} L_refine={float(L_refine.item()):.4f} "
                    f"CEpx(sel)={ce_px:.1f} CEpx(refined)={ce_refined_px:.1f}"
                )

            if (global_step % 500) == 0:
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "args": vars(args),
                }
                out_path = save_dir / f"surgatt_tracker_step_{global_step}.pt"
                torch.save(ckpt, out_path)
                print(f"[Saved] {out_path}")

        epoch_dt = time.time() - epoch_t0
        print(f"[Epoch {epoch}] finished. epoch_time={epoch_dt/60:.2f} min ({epoch_dt:.1f}s)")

        if (epoch % args.save_every) == 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "args": vars(args),
            }
            out_path = save_dir / f"surgatt_tracker_epoch_{epoch:03d}.pt"
            torch.save(ckpt, out_path)
            print(f"[Saved] {out_path}")

    hook.remove()
    print("Training finished.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    main()
