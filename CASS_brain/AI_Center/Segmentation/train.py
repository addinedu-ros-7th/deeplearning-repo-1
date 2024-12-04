import os
from ultralytics import YOLO


DATA_BASE = 'datasets/Lane/lane.yaml'
CHECKPOINT = 'runs/segment/train11/weights/best.pt'
# CHECKPOINT ='yolov8n-seg.pt'

model = YOLO(CHECKPOINT)

model.train(
    data=DATA_BASE,
    epochs=1000,
    imgsz=640,
    batch=-1,
    device=0,
    # lr0 = 1e-3,
    # warmup_epochs=5,
    # cos_lr=True,
    # mask_ratio=2
)