# SurgAtt-Tracker

**Anchor-Guided Multi-Scale Reranking for Surgical Attention Tracking**

---

## 🔧 Environment Setup

### 1. Create Conda Environment

```bash
conda create -n surgatt python=3.10
conda activate surgatt
```

### 2. Install Dependencies

```bash
pip install torch torchvision
pip install ultralytics
pip install tqdm
```


---

## 📂 Dataset Format

Expected directory structure:

```
images/
  train/
    video_01/
      xxx_frame_0001.jpg
      xxx_frame_0002.jpg
      ...
  val/
    video_02/
      ...

labels/
  train/
    video_01/
      xxx_frame_0001.txt
      xxx_frame_0002.txt
      ...
  val/
    video_02/
      ...
```



## 🚀 Training

Example training command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --weights path/to/yolo.pt \
  --img_root path/to/images/train \
  --lbl_root path/to/labels/train \
  --input_h 384 --input_w 640 \
  --k 10 \
  --topk 3 \
  --lambda_topk 1.0 \
  --lambda_ce1 0.2 \
  --lambda_soft_center 1.0 \
  --lambda_size 0.3 \
  --use_geo_bias \
  --lr 1e-4 \
  --wd 1e-2 \
  --epochs 50 \
  --save_dir ./ckpt
```

Model checkpoints will be saved to:

```
./ckpt
```

---

## 🔎 Inference

Run inference:

```bash
python scripts/test.py \
  --weights path/to/tracker.pt \
  --yolo path/to/yolo.pt \
  --source path/to/video_or_folder \
  --output output.txt
```


