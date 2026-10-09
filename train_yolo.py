"""
Step 4 of Project 1: train a YOLO model on your labeled landing-pad photos.

Before running:
  1. Export your dataset from Roboflow in "YOLOv11" format and unzip it
     into this project, in a folder named  pad_dataset
     (it should contain data.yaml plus train/, valid/ and test/ folders).
  2. Install the YOLO library once:   pip install ultralytics

Run:  python train_yolo.py
Results go to:  runs/detect/pad_v1/
"""

import os
import sys

import yaml
import torch
from ultralytics import YOLO

DATASET_DIR = "pad_dataset"   # folder you unzipped from Roboflow
EPOCHS = 50                   # how many times YOLO looks at all photos
IMG_SIZE = 640
RUN_NAME = "pad_v1"

# 1. Find data.yaml and rewrite its paths so they always work.
#    (Roboflow's file uses paths like ../train/images that often break.)
src_yaml = os.path.join(DATASET_DIR, "data.yaml")
if not os.path.exists(src_yaml):
    sys.exit(f"Could not find {src_yaml}. Did you unzip the Roboflow export into '{DATASET_DIR}'?")

with open(src_yaml) as f:
    data = yaml.safe_load(f)

root = os.path.abspath(DATASET_DIR)
data["path"] = root
data["train"] = "train/images"
data["val"] = "valid/images"
if os.path.isdir(os.path.join(root, "test", "images")):
    data["test"] = "test/images"
else:
    data.pop("test", None)

fixed_yaml = os.path.join(DATASET_DIR, "data_fixed.yaml")
with open(fixed_yaml, "w") as f:
    yaml.safe_dump(data, f)

for split in ("train", "valid"):
    n = len(os.listdir(os.path.join(root, split, "images")))
    print(f"{split}: {n} images")
print(f"Classes: {data.get('names')}")

# 2. Use the Mac's GPU (Apple "mps") if available, otherwise the CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training on: {device}")

# 3. Start from a small pretrained YOLO model and teach it your pad.
model = YOLO("yolo11n.pt")   # 'n' = nano, the smallest and fastest version
model.train(
    data=fixed_yaml,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=16,
    device=device,
    name=RUN_NAME,
    exist_ok=True,
    workers=2,
)

# 4. Grade the trained model on the validation photos it never trained on.
best = os.path.join("runs", "detect", RUN_NAME, "weights", "best.pt")
metrics = YOLO(best).val(data=fixed_yaml, imgsz=IMG_SIZE, device=device)
print("\n========== RESULTS ==========")
print(f"mAP50     : {metrics.box.map50:.3f}   (above 0.8 is good)")
print(f"mAP50-95  : {metrics.box.map:.3f}")
print(f"Precision : {metrics.box.mp:.3f}   (how often a detection is really a pad)")
print(f"Recall    : {metrics.box.mr:.3f}   (how many pads it found)")
print(f"Model saved at: {best}")
