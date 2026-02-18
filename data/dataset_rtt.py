# surgatt_tracker/data/dataset_rtt.py
# -*- coding: utf-8 -*-
"""
GapRTTDataset: Reference-Target dataset with categorical temporal gap sampling
  r = t - n,  n ~ Categorical(gap_N, gap_P)

Expected layout:
  img_root/<...>/<video>/*.jpg|png|...
  lbl_root/<...>/<video>/*.txt  (same filename, YOLO normalized: cls cx cy w h)

Each sample returns:
  xr: (3,H,W) float32 in [0,1]  (reference frame)
  xt: (3,H,W) float32 in [0,1]  (target frame)
  gtr_xyxy: (4,) float32 in input pixel coords (x1,y1,x2,y2)
  gtt_xyxy: (4,) float32 in input pixel coords (x1,y1,x2,y2)
"""

from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------
# Consts / regex
# -------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_FRAME_RE = re.compile(r"_frame_(\d+)\.(jpg|jpeg|png|bmp|tif|tiff|webp)$", re.IGNORECASE)


# -------------------------
# Utils
# -------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def frame_id_from_name(name: str) -> int:
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else -1

def yolo_norm_to_xyxy(cx, cy, w, h, img_w, img_h):
    """
    YOLO normalized (cx,cy,w,h) -> pixel xyxy, in the resized input coordinate system.
    """
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return x1, y1, x2, y2

def _read_resize_to_tensor(path: str, out_hw: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, out_hw, interpolation=TF.InterpolationMode.BILINEAR)
    return TF.to_tensor(img)  # float32 CHW [0,1]


# -------------------------
# Collate types
# -------------------------
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


# -------------------------
# Dataset
# -------------------------
class GapRTTDataset(Dataset):
    """
    Reference-target dataset with categorical temporal gap sampling:
      r = t - n,  n ~ Cat(gap_N, gap_P)

    - Enumerate each valid target frame index t in each video: t in [max_gap, len(frames)-1]
    - Sample gap n -> pick reference index r=t-n
    - Both frames must have label files.
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

        if len(gap_N) != len(gap_P) or len(gap_N) == 0:
            raise ValueError("gap_N and gap_P must have same non-zero length.")
        self.gap_N = [int(x) for x in gap_N]
        p = torch.tensor([float(x) for x in gap_P], dtype=torch.float32)
        p = p / (p.sum() + 1e-9)
        self.gap_P = p.tolist()
        self.max_gap = max(self.gap_N)

        # videos: list of (video_dir, label_dir, frames_sorted)
        self.videos: List[Tuple[Path, Path, List[Path]]] = []
        # base_index: list of (video_idx, target_frame_idx_t)
        self.base_index: List[Tuple[int, int]] = []

        self._build_index()

    def _build_index(self):
        # collect all leaf-ish dirs that contain at least one image
        video_dirs: List[Path] = []
        for d in self.img_root.rglob("*"):
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in IMG_EXTS:
                        video_dirs.append(d)
                        break
        video_dirs = sorted(set(video_dirs))
        if not video_dirs:
            raise RuntimeError(f"No video folders under: {self.img_root}")

        # build per-video frame list with paired labels
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

            # need enough length to allow max_gap backward reference
            if len(usable) <= self.max_gap:
                continue

            self.videos.append((vd, ld, usable))

        if not self.videos:
            raise RuntimeError(f"No usable videos under {self.img_root} + {self.lbl_root}")

        # enumerate all target indices t per video
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

        # try categorical sampling
        for _ in range(self.max_tries):
            n = self._sample_gap()
            r = t - n
            if r < 0:
                continue

            pr = frames[r]
            pt = frames[t]

            if (ld / pr.name).with_suffix(".txt").exists() and (ld / pt.name).with_suffix(".txt").exists():
                gtr = self._read_gt_xyxy(pr, ld)
                gtt = self._read_gt_xyxy(pt, ld)
                xr = _read_resize_to_tensor(str(pr), (self.input_h, self.input_w))
                xt = _read_resize_to_tensor(str(pt), (self.input_h, self.input_w))
                return RTTSample(xr=xr, gtr_xyxy=gtr, xt=xt, gtt_xyxy=gtt)

        # fallback: smallest gap
        n = min(self.gap_N)
        r = t - n
        r = max(r, 0)

        pr = frames[r]
        pt = frames[t]
        gtr = self._read_gt_xyxy(pr, ld)
        gtt = self._read_gt_xyxy(pt, ld)
        xr = _read_resize_to_tensor(str(pr), (self.input_h, self.input_w))
        xt = _read_resize_to_tensor(str(pt), (self.input_h, self.input_w))
        return RTTSample(xr=xr, gtr_xyxy=gtr, xt=xt, gtt_xyxy=gtt)
