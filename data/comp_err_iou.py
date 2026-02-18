#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import math


def list_txts(root: Path):
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def read_first_yolo_line(txt_path: Path):
    """
    Expect: cls cx cy w h (optional conf)
    """
    if not txt_path.exists():
        return None
    s = txt_path.read_text().strip()
    if not s or "NO_DET" in s:
        return None
    parts = s.splitlines()[0].strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        w  = float(parts[3])
        h  = float(parts[4])
        return cls, cx, cy, w, h
    except Exception:
        return None


def box_iou_norm(a, b):
    """
    a,b: (cx,cy,w,h) in [0,1]
    IoU in normalized space (equivalent to px IoU)
    """
    acx, acy, aw, ah = a
    bcx, bcy, bw, bh = b

    ax1 = acx - aw / 2
    ay1 = acy - ah / 2
    ax2 = acx + aw / 2
    ay2 = acy + ah / 2

    bx1 = bcx - bw / 2
    by1 = bcy - bh / 2
    bx2 = bcx + bw / 2
    by2 = bcy + bh / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, aw * ah)
    area_b = max(0.0, bw * bh)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def center_err_px(gt, pr, W, H):
    _, gcx, gcy, _, _ = gt
    _, pcx, pcy, _, _ = pr
    dx = (pcx - gcx) * W
    dy = (pcy - gcy) * H
    return math.sqrt(dx * dx + dy * dy)


def main():
    ap = argparse.ArgumentParser("Eval avg IoU & center error (fixed image size)")
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--out_json", type=str, default="metrics_fixed_size.json")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    gt_txts = list_txts(gt_dir)
    if not gt_txts:
        raise RuntimeError(f"No GT txt under {gt_dir}")

    n_total = 0
    n_used = 0
    n_missing_pred = 0
    n_invalid = 0

    sum_iou = 0.0
    sum_err = 0.0

    for gt_path in gt_txts:
        rel = gt_path.relative_to(gt_dir)
        pr_path = pred_dir / rel

        n_total += 1
        if not pr_path.exists():
            n_missing_pred += 1
            continue

        gt = read_first_yolo_line(gt_path)
        pr = read_first_yolo_line(pr_path)
        if gt is None or pr is None:
            n_invalid += 1
            continue

        iou = box_iou_norm(gt[1:], pr[1:])
        err = center_err_px(gt, pr, args.width, args.height)

        sum_iou += iou
        sum_err += err
        n_used += 1

    payload = {
        "meta": {
            "gt_dir": str(gt_dir),
            "pred_dir": str(pred_dir),
            "image_size": [args.width, args.height],
        },
        "counts": {
            "n_total_gt": n_total,
            "n_used": n_used,
            "n_missing_pred": n_missing_pred,
            "n_invalid": n_invalid,
            "det_rate": n_used / n_total if n_total > 0 else 0.0,
        },
        "metrics": {
            "avg_iou": sum_iou / n_used if n_used > 0 else 0.0,
            "avg_center_err_px": sum_err / n_used if n_used > 0 else 0.0,
        }
    }

    out = Path(args.out_json).resolve()
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


"""
python /anwang/zrl/SurgFov/baseline_model/yolo_code/comp_err_iou.py\
  --gt_dir /anwang/zrl/SurgFov/SurgAtt_100k_yolo_all_fps/labels/val \
  --pred_dir /anwang/zrl/SurgFov/baseline_result/Surgatt_tracker/box/out_txt \
  --out_json /anwang/zrl/SurgFov/baseline_result/Surgatt_tracker/metrics_txt_no_refine.json

"""