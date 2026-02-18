# surgatt_tracker/loss/refine_losses.py
import torch
import torch.nn.functional as F
from ..utils.boxes import xyxy_center, xyxy_wh, diag_norm

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