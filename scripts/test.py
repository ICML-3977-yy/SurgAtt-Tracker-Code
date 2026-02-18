# surgatt_tracker/scripts/infer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path
from typing import List

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile

from surgatt_tracker.detector.yolo_runner import DetectorRunner
from surgatt_tracker.model.tracker import SurgAttTrackerCore

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_FRAME_RE = re.compile(r"_frame_(\d+)\.(jpg|jpeg|png|bmp|tif|tiff|webp)$", re.IGNORECASE)


def frame_id_from_name(name: str) -> int:
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else -1

def _read_resize_to_tensor(path: str, out_hw: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, out_hw, interpolation=TF.InterpolationMode.BILINEAR)
    return TF.to_tensor(img)  # float32 CHW [0,1]

def xyxy_to_yolo_norm(xyxy: torch.Tensor, W: int, H: int) -> List[float]:
    """
    xyxy: (4,) tensor in pixels -> (cx,cy,w,h) normalized
    """
    x1, y1, x2, y2 = xyxy.tolist()
    cx = (x1 + x2) * 0.5 / float(W)
    cy = (y1 + y2) * 0.5 / float(H)
    w = (x2 - x1) / float(W)
    h = (y2 - y1) / float(H)
    # clamp for safety
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return [cx, cy, w, h]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("SurgAtt-Tracker Inference (Online, output YOLO txt)")

    ap.add_argument("--weights", type=str, required=True, help="YOLO weights")
    ap.add_argument("--ckpt", type=str, required=True, help="tracker checkpoint (.pt)")

    ap.add_argument("--img_dir", type=str, required=True, help="folder containing frames")
    ap.add_argument("--out_dir", type=str, required=True, help="output folder for YOLO txt labels")

    ap.add_argument("--input_h", type=int, default=384)
    ap.add_argument("--input_w", type=int, default=640)

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--half", action="store_true")

    # yolo hook
    ap.add_argument("--p3_idx", type=int, default=14)
    ap.add_argument("--p4_idx", type=int, default=17)
    ap.add_argument("--p5_idx", type=int, default=20)
    ap.add_argument("--p3_in", type=int, default=256)
    ap.add_argument("--p4_in", type=int, default=512)
    ap.add_argument("--p5_in", type=int, default=512)

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--pool", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)

    # init policy
    ap.add_argument("--init_ref", type=str, default="yolo_top1", choices=["yolo_top1"],
                    help="frame0 ref init (currently: yolo_top1)")

    # output class id
    ap.add_argument("--cls", type=int, default=0)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = Path(args.img_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # list frames
    frames = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    frames.sort(key=lambda p: (frame_id_from_name(p.name), p.name))
    if not frames:
        raise RuntimeError(f"No images found in {img_dir}")

    detector = DetectorRunner(
        weights=args.weights,
        device=device,
        half=bool(args.half),
        p3_idx=args.p3_idx, p4_idx=args.p4_idx, p5_idx=args.p5_idx,
    )

    model = SurgAttTrackerCore(
        input_w=args.input_w,
        input_h=args.input_h,
        k=args.k,
        p3_in=args.p3_in, p4_in=args.p4_in, p5_in=args.p5_in,
        d_model=args.d_model,
        n_heads=args.n_heads,
        pool=args.pool,
        dropout=args.dropout,
    ).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)

    ref_xyxy = None
    feats_prev = None

    for t, fp in enumerate(frames):
        x = _read_resize_to_tensor(str(fp), (args.input_h, args.input_w)).unsqueeze(0).to(device)  # (1,3,H,W)

        feats, boxes, confs, valid = detector(
            x, conf_thres=args.conf, iou_thres=args.iou, k=args.k, feats_force_fp32=True
        )
        feats_cur = feats
        boxes_cur = boxes          # (1,K,4)
        confs_cur = confs          # (1,K)
        valid_cur = valid          # (1,K)

        has_any = bool(valid_cur.any().item())

        # ---- init reference on frame0 ----
        if t == 0:
            if has_any:
                conf_masked = confs_cur.masked_fill(~valid_cur, -1e30)
                idx0 = conf_masked.argmax(dim=1)  # (1,)
                ref_xyxy = boxes_cur[0, idx0.item()].detach()  # (4,)
            else:
                ref_xyxy = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=torch.float32)  # dummy
            feats_prev = feats_cur

            # output yolo txt for frame0
            out_txt = out_dir / fp.with_suffix(".txt").name
            if has_any:
                cx, cy, w, h = xyxy_to_yolo_norm(ref_xyxy, args.input_w, args.input_h)
                out_txt.write_text(f"{args.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            else:
                out_txt.write_text("")  # empty
            continue

        # ---- online step: rerank + refine using previous frame as reference ----
        if (ref_xyxy is not None) and (feats_prev is not None) and has_any:
            out = model.rerank_and_refine(
                feats_ref=feats_prev,
                ref_xyxy=ref_xyxy.view(1, 4),
                feats_tgt=feats_cur,
                boxes_tgt=boxes_cur,
                valid_tgt=valid_cur,
            )
            ref_xyxy = out["refined_xyxy"][0].detach()
        # if no valid proposals, keep ref_xyxy as is (carry)

        feats_prev = feats_cur

        # ---- write YOLO-format output ----
        out_txt = out_dir / fp.with_suffix(".txt").name
        if ref_xyxy is None:
            out_txt.write_text("")
        else:
            cx, cy, w, h = xyxy_to_yolo_norm(ref_xyxy, args.input_w, args.input_h)
            out_txt.write_text(f"{args.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    detector.close()
    print(f"Done. YOLO txt written to: {out_dir}")


if __name__ == "__main__":
    main()
