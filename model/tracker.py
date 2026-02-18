# surgatt_tracker/model/tracker.py
import torch
import torch.nn as nn
from .msr import MultiScaleROIEmbedder
from .rerank import ASRerank
from .refine import MAARefineHead, polar_refine_update
from ..model.refine import xyxy_to_cxcywh
import math

class SurgAttTrackerCore(nn.Module):
    def __init__(self, input_w, input_h, k, p3_in, p4_in, p5_in, d_model=256, n_heads=8, pool=3, dropout=0.0):
        super().__init__()
        self.input_w = input_w
        self.input_h = input_h
        self.k = k
        self.msr = MultiScaleROIEmbedder(p3_in, p4_in, p5_in, d_model=d_model, pool=pool)
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

    def extract_tokens(self, feats, boxes_xyxy):
        return self.msr(feats, boxes_xyxy, self.input_w, self.input_h)

    def rerank_and_refine(self, feats_ref, ref_xyxy, feats_tgt, boxes_tgt, valid_tgt):
        """
        ref_xyxy: (B,4)
        boxes_tgt: (B,K,4)
        valid_tgt: (B,K)
        returns dict: scores(B,K), top1_xyxy(B,4), refined_xyxy(B,4), idx_top1(B,)
        """
        B, K, _ = boxes_tgt.shape
        ref_tok = self.extract_tokens(feats_ref, ref_xyxy.unsqueeze(1)).squeeze(1)  # (B,D)
        tgt_toks = self.extract_tokens(feats_tgt, boxes_tgt)                        # (B,K,D)
        scores = self.rerank(ref_tok, tgt_toks, valid_tgt)                          # (B,K)

        idx_top1 = scores.masked_fill(~valid_tgt, -1e30).argmax(dim=1)
        top1_xyxy = boxes_tgt[torch.arange(B, device=boxes_tgt.device), idx_top1]
        top1_tok = tgt_toks[torch.arange(B, device=boxes_tgt.device), idx_top1]

        geo_top1 = self.make_geo(top1_xyxy, ref_xyxy, self.input_w, self.input_h)   # (B,10)
        pred = self.refine(top1_tok, geo_top1)
        refined_xyxy = polar_refine_update(top1_xyxy, pred, self.input_w, self.input_h)

        return {
            "scores": scores,
            "idx_top1": idx_top1,
            "top1_xyxy": top1_xyxy,
            "refined_xyxy": refined_xyxy,
        }
