import torch

def xyxy_center(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return torch.stack([cx, cy], dim=-1)

def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU for XYXY boxes.

    boxes1: (B,K,4) or (K,4)
    boxes2: (B,4) or (4,)
    returns: (B,K) or (K,)
    """
    if boxes1.dim() == 2:
        b1 = boxes1.unsqueeze(0)  # (1,K,4)
        b2 = boxes2.unsqueeze(0) if boxes2.dim() == 1 else boxes2  # (1,4) or (1,4)
        b2 = b2.unsqueeze(1)  # (1,1,4)
        squeeze = True
    else:
        b1 = boxes1  # (B,K,4)
        b2 = boxes2.unsqueeze(1)  # (B,1,4)
        squeeze = False

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
    iou = inter / union
    return iou.squeeze(0) if squeeze else iou


@torch.no_grad()
def oracle_select_ref_box(
    boxes_r: torch.Tensor,   # (B,K,4)
    valid_r: torch.Tensor,   # (B,K) bool
    gtr: torch.Tensor,       # (B,4)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Training reference selection (oracle):
    - If any valid proposal has non-zero IoU with GT, restrict candidates to those IoU>0 proposals.
    - Otherwise, use all valid proposals.
    - Select the proposal with minimum center distance to GT center.

    Returns:
      ref_xyxy: (B,4)
      idx_ref:  (B,) long
    """
    assert boxes_r.dim() == 3 and boxes_r.size(-1) == 4
    assert valid_r.shape[:2] == boxes_r.shape[:2]
    assert gtr.dim() == 2 and gtr.size(-1) == 4

    B, K, _ = boxes_r.shape
    device = boxes_r.device

    iou = box_iou_xyxy(boxes_r, gtr)             # (B,K)
    valid_iou = valid_r & (iou > 0)              # (B,K)
    use_iou = valid_iou.any(dim=1, keepdim=True) # (B,1)

    # cand = valid_iou if any IoU>0 exists else valid_r
    cand = torch.where(use_iou, valid_iou, valid_r)  # (B,K)

    # center distance
    c = xyxy_center(boxes_r)                       # (B,K,2)
    gt_c = xyxy_center(gtr).unsqueeze(1)           # (B,1,2)
    d = torch.norm(c - gt_c, dim=-1)               # (B,K)
    d = d.masked_fill(~cand, 1e9)

    idx_ref = d.argmin(dim=1)                      # (B,)
    # safety clamp (should already be valid)
    idx_ref = idx_ref.clamp(min=0, max=K - 1)

    ref_xyxy = boxes_r[torch.arange(B, device=device), idx_ref]  # (B,4)
    return ref_xyxy, idx_ref


@torch.no_grad()
def oracle_best1_and_topM(
    boxes_t: torch.Tensor,        # (B,K,4)
    valid_mask_t: torch.Tensor,   # (B,K) bool (e.g., IoU>0 if exists else valid)
    gtt: torch.Tensor,            # (B,4)
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Oracle target indices for reranking losses:
    - best1_idx: argmin center error among valid_mask_t
    - topM_idx: indices of smallest center errors (size M), padded with -1 when insufficient
    - eps_topM: corresponding center errors (pixels), padded with +inf where invalid

    Returns:
      best1_idx: (B,) long
      topM_idx:  (B,M) long (padded with -1)
      eps_topM:  (B,M) float (padded with +inf)
    """
    assert boxes_t.dim() == 3 and boxes_t.size(-1) == 4
    assert valid_mask_t.shape[:2] == boxes_t.shape[:2]
    assert gtt.dim() == 2 and gtt.size(-1) == 4
    assert M >= 1

    B, K, _ = boxes_t.shape
    device = boxes_t.device

    c = xyxy_center(boxes_t)                    # (B,K,2)
    gt_c = xyxy_center(gtt).unsqueeze(1)        # (B,1,2)
    d = torch.norm(c - gt_c, dim=-1)            # (B,K)
    d = d.masked_fill(~valid_mask_t, 1e9)

    # best-1
    best1_idx = d.argmin(dim=1).clamp(0, K - 1) # (B,)

    # top-M
    m = min(int(M), K)
    vals, idx = torch.topk(d, k=m, dim=1, largest=False, sorted=True)  # (B,m)

    topM_idx = torch.full((B, int(M)), -1, device=device, dtype=torch.long)
    eps_topM = torch.full((B, int(M)), float("inf"), device=device, dtype=torch.float32)

    topM_idx[:, :m] = idx
    eps_topM[:, :m] = vals.float()

    # invalidate huge distances (means no valid candidates)
    bad = eps_topM >= 1e8
    topM_idx[bad] = -1
    eps_topM[bad] = float("inf")

    # if a row has no valid candidate at all, keep best1_idx as 0 but caller should mask by valid_any
    return best1_idx, topM_idx, eps_topM
