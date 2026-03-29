import json
from pathlib import Path
from ultralytics import YOLO

# ── 配置 ──────────────────────────────────────────
WEIGHTS  = '/data/hdd2/clearsar/runs/yolov8x_p2_640/weights/best.pt'
TEST_DIR = '/data/hdd2/clearsar/ClearSAR/data/images/test'
OUT_FILE = '/data/hdd2/clearsar/scripts/submission.json'
CONF     = 0.40   # 初始阈值，提交前可调
IOU      = 0.5    # NMS IoU
IMGSZ    = 640
# ──────────────────────────────────────────────────

model = YOLO(WEIGHTS)
test_images = sorted(Path(TEST_DIR).glob('*.png'))
print(f"推理图像数: {len(test_images)}")

detections = []
for img_path in test_images:
    results = model.predict(
        source=str(img_path),
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        verbose=False,
        device='0',
    )
    image_id = int(img_path.stem)
    for r in results:
        boxes  = r.boxes.xyxy.cpu().tolist()
        scores = r.boxes.conf.cpu().tolist()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            detections.append({
                "image_id":   image_id,
                "category_id": 1,
                "bbox":  [x1, y1, x2 - x1, y2 - y1],  # COCO [x,y,w,h]
                "score": round(score, 6),
            })

with open(OUT_FILE, 'w') as f:
    json.dump(detections, f)

print(f"总检测数: {len(detections)}")
print(f"平均每图: {len(detections)/len(test_images):.1f}")
print(f"已保存 → {OUT_FILE}")
