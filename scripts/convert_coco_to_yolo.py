# convert_coco_to_yolo.py
import json, shutil
from pathlib import Path

src = Path('/data/hdd2/clearsar/ClearSAR/data')
dst = Path('/data/hdd2/clearsar/yolo_dataset')

with open(src / 'annotations/instances_train.json') as f:
    coco = json.load(f)

# 建 val split (10%)
import random; random.seed(42)
all_imgs = coco['images']
random.shuffle(all_imgs)
val_imgs = set(x['id'] for x in all_imgs[:315])   # ~10%

img_map = {img['id']: img for img in coco['images']}

for split in ['train', 'val']:
    (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
    (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)

ann_by_img = {}
for ann in coco['annotations']:
    ann_by_img.setdefault(ann['image_id'], []).append(ann)

for img_info in coco['images']:
    img_id = img_info['id']
    fname = img_info['file_name']
    split = 'val' if img_id in val_imgs else 'train'
    
    # 复制图片
    src_img = src / 'images/train' / fname
    shutil.copy(src_img, dst / 'images' / split / fname)
    
    # 写 label
    w, h = img_info['width'], img_info['height']
    anns = ann_by_img.get(img_id, [])
    label_path = dst / 'labels' / split / fname.replace('.png', '.txt')
    with open(label_path, 'w') as f:
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            cx = (x + bw/2) / w
            cy = (y + bh/2) / h
            f.write(f"0 {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}\n")

print("done")