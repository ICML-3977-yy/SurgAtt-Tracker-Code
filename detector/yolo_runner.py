# surgatt_tracker/detector/yolo_runner.py
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression

class FeatureHook:
    ...

@torch.no_grad()
def yolo_forward_tensor(
    *,
    det_model,
    hook,
    x: torch.Tensor,          # (B,3,H,W) on device
    conf_thres: float,
    iou_thres: float,
    k: int,
    feats_force_fp32: bool = True,
):
    """
    Returns:
      feats: dict {p3,p4,p5} each (B,C,Hf,Wf) float32 on device (detached)
      boxes: (B,K,4) float32 xyxy
      confs: (B,K) float32
      valid: (B,K) bool
    """
    model_dtype = next(det_model.parameters()).dtype
    x = x.to(dtype=model_dtype)

    hook.clear()
    preds = det_model(x)  # triggers hooks
    feats = hook.pop()

    for kk in feats:
        ft = feats[kk].detach()
        if feats_force_fp32:
            ft = ft.float()
        feats[kk] = ft

    dets = non_max_suppression(
        preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=None,
        agnostic=True,
        max_det=k,
    )

    B = x.shape[0]
    boxes = torch.zeros((B, k, 4), device=x.device, dtype=torch.float32)
    confs = torch.zeros((B, k), device=x.device, dtype=torch.float32)
    valid = torch.zeros((B, k), device=x.device, dtype=torch.bool)

    for i, di in enumerate(dets):
        if di is None or len(di) == 0:
            continue
        n = min(k, di.shape[0])
        boxes[i, :n] = di[:n, :4].float()
        confs[i, :n] = di[:n, 4].float()
        valid[i, :n] = True

    return feats, boxes, confs, valid


class DetectorRunner:
    def __init__(self, weights, device, half, p3_idx, p4_idx, p5_idx):
        self.yolo = YOLO(weights)
        self.det_model = self.yolo.model
        self.det_model.eval()
        for p in self.det_model.parameters():
            p.requires_grad_(False)
        self.det_model.to(device)
        if half and device.type == "cuda":
            self.det_model.half()
        else:
            self.det_model.float()

        self.hook = FeatureHook()
        self.hook.register(self.det_model, p3_idx=p3_idx, p4_idx=p4_idx, p5_idx=p5_idx)

    @torch.no_grad()
    def __call__(self, x, conf_thres, iou_thres, k, feats_force_fp32=True):
        return yolo_forward_tensor(
            det_model=self.det_model, hook=self.hook, x=x,
            conf_thres=conf_thres, iou_thres=iou_thres, k=k,
            feats_force_fp32=feats_force_fp32
        )

    def close(self):
        self.hook.remove()
