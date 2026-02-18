# surgatt_tracker/model/msr.py
import torch
import torch.nn as nn
from torchvision.ops import roi_align
from typing import List, Tuple, Optional, Dict

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

