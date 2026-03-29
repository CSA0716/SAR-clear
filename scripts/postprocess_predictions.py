"""Post-process ClearSAR COCO predictions with NMS / box voting.

Key features:
- Merge one or multiple COCO prediction JSON files.
- Optional per-file weighting (useful for model ensemble).
- Per-image per-class NMS or score-weighted box voting.
- Optional clipping to image bounds using COCO annotation metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(dets: List[dict], iou_thr: float) -> List[dict]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda x: x["score"], reverse=True)
    kept: List[dict] = []

    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou_xywh(best["bbox"], d["bbox"]) < iou_thr]
    return kept


def box_voting(dets: List[dict], iou_thr: float) -> List[dict]:
    if not dets:
        return []

    pool = sorted(dets, key=lambda x: x["score"], reverse=True)
    merged: List[dict] = []

    while pool:
        seed = pool.pop(0)
        cluster = [seed]
        remain = []
        for d in pool:
            if iou_xywh(seed["bbox"], d["bbox"]) >= iou_thr:
                cluster.append(d)
            else:
                remain.append(d)
        pool = remain

        ws = [max(c["score"], 1e-6) for c in cluster]
        wsum = sum(ws)
        x = sum(c["bbox"][0] * w for c, w in zip(cluster, ws)) / wsum
        y = sum(c["bbox"][1] * w for c, w in zip(cluster, ws)) / wsum
        w = sum(c["bbox"][2] * w for c, w in zip(cluster, ws)) / wsum
        h = sum(c["bbox"][3] * w for c, w in zip(cluster, ws)) / wsum
        score = max(c["score"] for c in cluster)

        out = dict(seed)
        out["bbox"] = [x, y, w, h]
        out["score"] = float(score)
        merged.append(out)

    return merged


def load_predictions(paths: List[Path], input_weights: List[float] | None = None) -> List[dict]:
    if input_weights is None:
        input_weights = [1.0] * len(paths)

    if len(paths) != len(input_weights):
        raise ValueError("--inputs and --input-weights must have the same length")

    merged: List[dict] = []
    for p, w in zip(paths, input_weights):
        with p.open("r", encoding="utf-8") as f:
            preds = json.load(f)
        for d in preds:
            item = dict(d)
            item["score"] = float(item["score"]) * float(w)
            merged.append(item)
    return merged


def load_image_shapes(coco_ann_path: Path) -> Dict[int, Tuple[int, int]]:
    with coco_ann_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    image_shapes: Dict[int, Tuple[int, int]] = {}
    for im in ann.get("images", []):
        image_shapes[int(im["id"])] = (int(im["width"]), int(im["height"]))
    return image_shapes


def clip_bbox_xywh(bbox: List[float], width: int, height: int) -> List[float] | None:
    x, y, w, h = bbox
    x1 = max(0.0, min(float(x), float(width)))
    y1 = max(0.0, min(float(y), float(height)))
    x2 = max(0.0, min(float(x + w), float(width)))
    y2 = max(0.0, min(float(y + h), float(height)))
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0.0 or nh <= 0.0:
        return None
    return [x1, y1, nw, nh]


def group_by_image_and_class(preds: List[dict]) -> Dict[Tuple[int, int], List[dict]]:
    grouped: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for d in preds:
        grouped[(int(d["image_id"]), int(d.get("category_id", 1)))].append(d)
    return grouped


def run_postprocess(
    preds: List[dict],
    conf: float,
    iou_thr: float,
    max_det: int,
    method: str,
    image_shapes: Dict[int, Tuple[int, int]] | None = None,
) -> List[dict]:
    preds = [d for d in preds if float(d["score"]) >= conf]
    grouped = group_by_image_and_class(preds)
    out: List[dict] = []

    for (image_id, _), dets in grouped.items():
        if method == "nms":
            pp = nms(dets, iou_thr=iou_thr)
        elif method == "vote":
            pp = box_voting(dets, iou_thr=iou_thr)
        else:
            raise ValueError(f"Unsupported method: {method}")

        pp = sorted(pp, key=lambda x: x["score"], reverse=True)[:max_det]

        if image_shapes and image_id in image_shapes:
            iw, ih = image_shapes[image_id]
            clipped = []
            for d in pp:
                box = clip_bbox_xywh(d["bbox"], width=iw, height=ih)
                if box is None:
                    continue
                nd = dict(d)
                nd["bbox"] = box
                clipped.append(nd)
            pp = clipped

        for d in pp:
            d["score"] = round(float(d["score"]), 6)
        out.extend(pp)

    return out


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Input COCO prediction json file(s)")
    p.add_argument("--output", required=True, help="Output COCO prediction json")
    p.add_argument("--input-weights", default="", help="Optional comma-separated weights per input file")
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--iou", type=float, default=0.55)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--method", choices=["nms", "vote"], default="vote")
    p.add_argument("--clip-with-ann", default="", help="Optional COCO ann file to clip bbox into image bounds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(x) for x in args.inputs]
    input_weights = parse_float_list(args.input_weights) if args.input_weights else None
    preds = load_predictions(input_paths, input_weights=input_weights)

    image_shapes = None
    if args.clip_with_ann:
        image_shapes = load_image_shapes(Path(args.clip_with_ann))

    out = run_postprocess(
        preds=preds,
        conf=args.conf,
        iou_thr=args.iou,
        max_det=args.max_det,
        method=args.method,
        image_shapes=image_shapes,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"input files: {len(input_paths)}")
    print(f"input detections: {len(preds)}")
    print(f"output detections: {len(out)}")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
