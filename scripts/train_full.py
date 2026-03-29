from ultralytics import YOLO

model = YOLO('yolov8x-p2.yaml')
model.load('yolov8x.pt')

model.train(
    data='/data/hdd2/clearsar/scripts/clearsar.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    device='0,1',
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
    patience=0,        # 关掉 early stopping，跑满
    box=9.0,
    cls=0.5,
    fliplr=0.5,
    flipud=0.5,
    degrees=90.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.4,
    translate=0.1,
    scale=0.3,
    project='/data/hdd2/clearsar/runs',
    name='yolov8x_p2_full_150',
    save_period=10,
    exist_ok=False,
)
