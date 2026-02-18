# surgatt_tracker/model/refine.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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


import torch
import torch.nn.functional as F

def xyxy_center(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return torch.stack([cx, cy], dim=-1)

def cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = cxcywh.unbind(dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = xyxy.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=1e-3)
    h = (y2 - y1).clamp(min=1e-3)
    return torch.stack([cx, cy, w, h], dim=-1)

def clamp_boxes_xyxy(boxes: torch.Tensor, W: int, H: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x1 = x1.clamp(0, W - 1)
    y1 = y1.clamp(0, H - 1)
    x2 = x2.clamp(0, W - 1)
    y2 = y2.clamp(0, H - 1)
    # ensure x2>x1, y2>y1
    x1_, x2_ = torch.minimum(x1, x2), torch.maximum(x1, x2)
    y1_, y2_ = torch.minimum(y1, y2), torch.maximum(y1, y2)
    x2_ = (x1_ + (x2_ - x1_).clamp(min=1.0)).clamp(0, W - 1)
    y2_ = (y1_ + (y2_ - y1_).clamp(min=1.0)).clamp(0, H - 1)
    return torch.stack([x1_, y1_, x2_, y2_], dim=-1)

def polar_refine_update(
    top1_xyxy: torch.Tensor,   # (B,4)
    pred: torch.Tensor,        # (B,4): theta, d_raw, log_sw, log_sh
    input_w: int,
    input_h: int,
    max_log_scale: float = 2.0
) -> torch.Tensor:
    """
    Apply polar refinement to top1 box:
      theta = pred[:,0]
      d     = softplus(pred[:,1])          (>=0)
      sw,sh = exp(clamp(log_sw/log_sh))    (scale)
    """
    top1_cxcywh = xyxy_to_cxcywh(top1_xyxy)     # (B,4)
    cx, cy, w, h = top1_cxcywh.unbind(dim=-1)

    theta = pred[:, 0]
    d = F.softplus(pred[:, 1])                 # positive distance
    log_sw = pred[:, 2].clamp(-max_log_scale, max_log_scale)
    log_sh = pred[:, 3].clamp(-max_log_scale, max_log_scale)

    # center update (polar)
    cx_new = cx + d * torch.cos(theta)
    cy_new = cy + d * torch.sin(theta)

    # size update (scale)
    w_new = w * torch.exp(log_sw)
    h_new = h * torch.exp(log_sh)

    refined = torch.stack([cx_new, cy_new, w_new, h_new], dim=-1)
    refined_xyxy = cxcywh_to_xyxy(refined)
    refined_xyxy = clamp_boxes_xyxy(refined_xyxy, input_w, input_h)
    return refined_xyxy
