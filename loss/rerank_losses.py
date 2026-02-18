# surgatt_tracker/loss/rerank_losses.py
import torch
import torch.nn.functional as F
from ..model.refine import xyxy_to_cxcywh, xyxy_center

def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, temp: float) -> torch.Tensor:
    minv = torch.finfo(scores.dtype).min
    s = scores.masked_fill(~mask, minv)
    w = F.softmax(s / max(temp, 1e-6), dim=1)
    w = w * mask.float()
    w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
    return w


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
