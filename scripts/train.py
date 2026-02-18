# surgatt_tracker/scripts/train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from surgatt_tracker.data.dataset_rtt import GapRTTDataset, collate_rtt, seed_all
from surgatt_tracker.detector.yolo_runner import DetectorRunner
from surgatt_tracker.model.tracker import SurgAttTrackerCore
from surgatt_tracker.loss.oracle import oracle_select_ref_box, oracle_best1_and_topM


# -------------------------
# box utils (training-only helpers)
# -------------------------
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

def diag_norm(W: int, H: int) -> float:
    return math.sqrt(float(W * W + H * H)) + 1e-6

def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (B,K,4), boxes2: (B,4) -> (B,K)
    """
    b1 = boxes1
    b2 = boxes2.unsqueeze(1)

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
    return inter / union


# -------------------------
# losses (paper-style building blocks)
# -------------------------
def _masked_log_softmax(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    s = scores.clone()
    s = s.masked_fill(~valid, -1e30)
    return F.log_softmax(s, dim=1)

def loss_rerank_ce(scores: torch.Tensor, best1_idx: torch.Tensor, valid_any: torch.Tensor) -> torch.Tensor:
    """
    CE on oracle best-1 (only on rows that have any valid candidate)
    """
    if not valid_any.any():
        return scores.new_tensor(0.0)
    s = scores[valid_any]
    t = best1_idx[valid_any]
    return F.cross_entropy(s, t, reduction="mean")

def loss_rerank_geo(scores: torch.Tensor, boxes_xyxy: torch.Tensor, valid: torch.Tensor, gt_xyxy: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
    """
    Geometry-aware soft supervision:
      q_k ∝ exp(-eps_k / tau) over valid
      L = - sum_k q_k log p_k
    eps_k = center distance (pixels)
    """
    valid_any = valid.any(dim=1)
    if not valid_any.any():
        return scores.new_tensor(0.0)

    c = xyxy_center(boxes_xyxy)                 # (B,K,2)
    gt_c = xyxy_center(gt_xyxy).unsqueeze(1)    # (B,1,2)
    eps = torch.norm(c - gt_c, dim=-1)          # (B,K)

    eps = eps.masked_fill(~valid, 1e9)
    q_logits = -eps / max(tau, 1e-6)
    q_logits = q_logits.masked_fill(~valid, -1e30)
    q = F.softmax(q_logits, dim=1).detach()

    logp = _masked_log_softmax(scores, valid)
    L = -(q * logp).sum(dim=1)
    return L[valid_any].mean()

def loss_rerank_listwise_topM(scores: torch.Tensor, topM_idx: torch.Tensor, eps_topM: torch.Tensor, valid: torch.Tensor, sigma: float = 15.0) -> torch.Tensor:
    """
    Listwise Top-M (weighted set likelihood):
      numerator = sum_{i in topM} exp(s_i) * w_i,  w_i = exp(-sigma * eps_i)
      denom     = sum_{j in valid} exp(s_j)
      L = -log(numerator / denom) = log(denom) - log(numerator)

    topM_idx: (B,M) with -1 padding
    eps_topM: (B,M) with +inf padding
    """
    B, K = scores.shape
    device = scores.device

    valid_any = valid.any(dim=1)
    if not valid_any.any():
        return scores.new_tensor(0.0)

    # denom: logsumexp over valid
    s_valid = scores.masked_fill(~valid, -1e30)
    log_denom = torch.logsumexp(s_valid, dim=1)  # (B,)

    # build numerator in log-space
    # collect s_i for i in topM; invalid indices contribute -inf
    M = topM_idx.shape[1]
    gather_idx = topM_idx.clamp(min=0)  # safe for gather
    s_top = torch.gather(scores, dim=1, index=gather_idx)  # (B,M)

    mask_top = (topM_idx >= 0) & torch.isfinite(eps_topM)  # (B,M)
    w = torch.exp(-float(sigma) * eps_topM.clamp(min=0.0))  # (B,M)
    w = w.masked_fill(~mask_top, 0.0)

    # numerator = sum exp(s_top) * w
    num = (torch.exp(s_top) * w).sum(dim=1)  # (B,)
    # avoid log(0)
    num = torch.clamp(num, min=1e-12)
    log_num = torch.log(num)

    L = (log_denom - log_num)
    return L[valid_any].mean()

def loss_refine_dist(refined_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, input_w: int, input_h: int, beta: float = 0.02) -> torch.Tensor:
    """
    SmoothL1 on normalized center displacement
    """
    diag = diag_norm(input_w, input_h)
    c_pred = xyxy_center(refined_xyxy)
    c_gt = xyxy_center(gt_xyxy)
    delta = (c_pred - c_gt) / diag
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=beta, reduction="mean")

def loss_refine_scale(refined_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, beta: float = 0.02) -> torch.Tensor:
    """
    SmoothL1 on log-scale (w,h)
    """
    wh_p = xyxy_wh(refined_xyxy).clamp(min=1e-3)
    wh_g = xyxy_wh(gt_xyxy).clamp(min=1e-3)
    d = torch.log(wh_p) - torch.log(wh_g)
    return F.smooth_l1_loss(d, torch.zeros_like(d), beta=beta, reduction="mean")


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser("SurgAtt-Tracker Training (Rerank + Polar Refine)")

    # data
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--lbl_root", type=str, required=True)
    ap.add_argument("--input_h", type=int, default=384)
    ap.add_argument("--input_w", type=int, default=640)

    # detector
    ap.add_argument("--weights", type=str, required=True)
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

    # gap sampling
    ap.add_argument("--gap_N", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--gap_P", type=str, default="0.4,0.2,0.1,0.1,0.1,0.1")
    ap.add_argument("--max_tries", type=int, default=8)

    # loader
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)

    # optim
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)

    # loss weights
    ap.add_argument("--M", type=int, default=3, help="Top-M for listwise")
    ap.add_argument("--tau_geo", type=float, default=0.15)
    ap.add_argument("--sigma_list", type=float, default=15.0)

    ap.add_argument("--lambda_ce", type=float, default=0.2)
    ap.add_argument("--lambda_geo", type=float, default=1.0)
    ap.add_argument("--lambda_list", type=float, default=1.0)
    ap.add_argument("--lambda_refine_dist", type=float, default=1.0)
    ap.add_argument("--lambda_refine_scale", type=float, default=0.3)

    # io
    ap.add_argument("--save_dir", type=str, default="./ckpt_surgatt_tracker")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    gap_N = [int(x) for x in args.gap_N.split(",") if x.strip()]
    gap_P = [float(x) for x in args.gap_P.split(",") if x.strip()]
    assert len(gap_N) == len(gap_P) and len(gap_N) > 0

    ds = GapRTTDataset(
        img_root=args.img_root,
        lbl_root=args.lbl_root,
        input_w=args.input_w,
        input_h=args.input_h,
        gap_N=gap_N,
        gap_P=gap_P,
        max_tries=args.max_tries,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_rtt,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )

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
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    from tqdm import tqdm
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        pbar = tqdm(dl, total=len(dl), dynamic_ncols=True, desc=f"Epoch {epoch}/{args.epochs}")
        run_loss, run_n = 0.0, 0

        for (xr, gtr, xt, gtt) in pbar:
            global_step += 1

            xr = xr.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)
            gtr = gtr.to(device, non_blocking=True)
            gtt = gtt.to(device, non_blocking=True)

            B = xr.shape[0]
            x = torch.cat([xr, xt], dim=0)  # (2B,3,H,W)

            with torch.no_grad():
                feats, boxes, confs, valid = detector(
                    x, conf_thres=args.conf, iou_thres=args.iou, k=args.k, feats_force_fp32=True
                )

            feats_r = {k: v[:B] for k, v in feats.items()}
            feats_t = {k: v[B:] for k, v in feats.items()}
            boxes_r, boxes_t = boxes[:B], boxes[B:]
            valid_r, valid_t = valid[:B], valid[B:]

            # --------- oracle reference selection mask (IoU>0 if exists else valid) ----------
            iou_r = box_iou_xyxy(boxes_r, gtr)
            mask_r_iou = valid_r & (iou_r > 0)
            use_iou_r = mask_r_iou.any(dim=1, keepdim=True)
            mask_r = torch.where(use_iou_r, mask_r_iou, valid_r)

            # ref selection (oracle): prefer IoU>0 proposals if any; else all valid
            # (function itself re-checks iou>0 internally; passing full valid_r is fine)
            ref_xyxy, _ = oracle_select_ref_box(boxes_r, valid_r, gtr)  # (B,4)

            # --------- oracle target mask (IoU>0 if exists else valid) ----------
            iou_t = box_iou_xyxy(boxes_t, gtt)
            mask_t_iou = valid_t & (iou_t > 0)
            use_iou_t = mask_t_iou.any(dim=1, keepdim=True)
            valid_mask_t = torch.where(use_iou_t, mask_t_iou, valid_t)

            best1_idx, topM_idx, eps_topM = oracle_best1_and_topM(
                boxes_t=boxes_t, valid_mask_t=valid_mask_t, gtt=gtt, M=int(args.M)
            )

            # --------- forward: rerank + refine ----------
            out = model.rerank_and_refine(
                feats_ref=feats_r,
                ref_xyxy=ref_xyxy,
                feats_tgt=feats_t,
                boxes_tgt=boxes_t,
                valid_tgt=valid_t,
            )
            scores = out["scores"]                # (B,K)
            refined_xyxy = out["refined_xyxy"]    # (B,4)

            valid_any_t = valid_mask_t.any(dim=1)

            # --------- losses ----------
            L_ce = loss_rerank_ce(scores, best1_idx, valid_any_t) * float(args.lambda_ce)
            L_geo = loss_rerank_geo(scores, boxes_t, valid_mask_t, gtt, tau=float(args.tau_geo)) * float(args.lambda_geo)
            L_list = loss_rerank_listwise_topM(scores, topM_idx, eps_topM, valid_mask_t, sigma=float(args.sigma_list)) * float(args.lambda_list)

            L_rd = loss_refine_dist(refined_xyxy, gtt, args.input_w, args.input_h) * float(args.lambda_refine_dist)
            L_rs = loss_refine_scale(refined_xyxy, gtt) * float(args.lambda_refine_scale)

            loss = L_ce + L_geo + L_list + L_rd + L_rs

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            run_loss += float(loss.item())
            run_n += 1
            pbar.set_postfix(
                loss=f"{run_loss/max(run_n,1):.3f}",
                Lce=f"{float(L_ce.item()):.3f}",
                Lgeo=f"{float(L_geo.item()):.3f}",
                Llist=f"{float(L_list.item()):.3f}",
                Lrd=f"{float(L_rd.item()):.3f}",
                Lrs=f"{float(L_rs.item()):.3f}",
            )

            if (global_step % args.log_every) == 0:
                with torch.no_grad():
                    ce_px = torch.norm(xyxy_center(refined_xyxy) - xyxy_center(gtt), dim=1)
                    ce_px = ce_px[valid_any_t].mean().item() if valid_any_t.any() else 0.0
                print(f"[E{epoch:02d} step={global_step}] loss={run_loss/max(run_n,1):.4f} ce_refined={ce_px:.2f}px")

        dt = time.time() - t0
        print(f"[Epoch {epoch}] done. time={dt/60:.2f} min ({dt:.1f}s)")

        if (epoch % args.save_every) == 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "args": vars(args),
            }
            outp = save_dir / f"surgatt_tracker_epoch_{epoch:03d}.pt"
            torch.save(ckpt, outp)
            print(f"[Saved] {outp}")

    detector.close()
    print("Training finished.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    main()
