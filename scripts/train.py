from ultralytics import YOLO

model = YOLO('yolov8x-p2.yaml')
model.load('yolov8x.pt')  # 迁移预训练权重

model.train(
    data='/media/rtx3090tix2/24f5d972-c4e4-41cb-91ac-75a7e1ea6762/clearsar/yaml/clearsar.yaml',
    epochs=150,
    imgsz=1024,
    batch=8,
    device='0,1',
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5,
    patience=20,
    # box loss 加权，对细长小目标更敏感
    box=9.0,
    cls=0.5,
    # augmentation
    fliplr=0.5,
    flipud=0.5,
    degrees=90.0,
    mosaic=0.0,      # 关！条纹会被截断
    mixup=0.0,
    copy_paste=0.0,
    hsv_h=0.0,       # SAR 假彩色，色调/饱和度无意义
    hsv_s=0.0,
    hsv_v=0.4,
    translate=0.1,
    scale=0.3,
    project='/data/hdd2/clearsar/runs',
    name='yolov8x_p2_640',
    save_period=10,
    exist_ok=False,
)
