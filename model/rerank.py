# surgatt_tracker/model/rerank.py
import math
import torch
import torch.nn as nn

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
