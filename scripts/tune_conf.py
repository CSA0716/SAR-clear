import json
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = '/data/hdd2/clearsar/runs/yolov8x_p2_640/weights/best.pt'
model = YOLO(WEIGHTS)

for conf in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    metrics = model.val(
        data='/data/hdd2/clearsar/scripts/clearsar.yaml',
        imgsz=640,
        conf=conf,
        iou=0.5,
        verbose=False,
        device='0',
    )
    p  = metrics.box.mp
    r  = metrics.box.mr
    m50   = metrics.box.map50
    m5095 = metrics.box.map
    print(f"conf={conf:.2f}  P={p:.3f}  R={r:.3f}  mAP50={m50:.3f}  mAP50-95={m5095:.3f}")
