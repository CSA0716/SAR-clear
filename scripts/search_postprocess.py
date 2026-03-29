"""Grid-search post-processing params for ClearSAR mAP50-95 on local val split."""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from postprocess_predictions import load_image_shapes, load_predictions, parse_float_list, run_postprocess


def coco_map5095(coco_gt: COCO, preds: list, quiet: bool = True) -> float:
    if len(preds) == 0:
        return 0.0

    coco_dt = coco_gt.loadRes(preds)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.summarize()
    else:
        evaluator.summarize()
    return float(evaluator.stats[0])


def parse_int_list(raw: str):
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Input prediction json(s)")
    p.add_argument("--input-weights", default="", help="Optional comma-separated weights per input file")
    p.add_argument("--ann", required=True, help="COCO val annotation file")
    p.add_argument("--methods", default="vote,nms", help="Comma-separated: vote,nms")
    p.add_argument("--confs", default="0.05,0.1,0.15,0.2,0.25,0.3")
    p.add_argument("--ious", default="0.45,0.5,0.55,0.6,0.65")
    p.add_argument("--max-dets", default="100,200,300")
    p.add_argument("--save-best", default="./scripts/submission_best.json")
    p.add_argument("--save-meta", default="./scripts/submission_best_meta.json")
    p.add_argument("--verbose-coco", action="store_true", help="Print full COCO summary for each setting")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    inputs = [Path(x) for x in args.inputs]
    input_weights = parse_float_list(args.input_weights) if args.input_weights else None
    ann_file = Path(args.ann)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    confs = parse_float_list(args.confs)
    ious = parse_float_list(args.ious)
    max_dets = parse_int_list(args.max_dets)

    preds = load_predictions(inputs, input_weights=input_weights)
    coco_gt = COCO(str(ann_file))
    image_shapes = load_image_shapes(ann_file)

    print(f"Loaded detections: {len(preds)} from {len(inputs)} file(s)")

    best = {"map": -1.0}

    for method, conf, iou_thr, max_det in itertools.product(methods, confs, ious, max_dets):
        pp = run_postprocess(
            preds,
            conf=conf,
            iou_thr=iou_thr,
            max_det=max_det,
            method=method,
            image_shapes=image_shapes,
        )

        score = coco_map5095(coco_gt=coco_gt, preds=pp, quiet=not args.verbose_coco)
        print(
            f"method={method:>4} conf={conf:.2f} iou={iou_thr:.2f} max_det={max_det:>3} => mAP50-95={score:.5f}"
        )

        if score > best["map"]:
            best = {
                "map": score,
                "method": method,
                "conf": conf,
                "iou": iou_thr,
                "max_det": max_det,
                "n_dets": len(pp),
                "inputs": [str(x) for x in inputs],
                "input_weights": input_weights if input_weights else [1.0] * len(inputs),
                "preds": pp,
            }

    print("\nBest setting:")
    print({k: v for k, v in best.items() if k != "preds"})

    save_best = Path(args.save_best)
    save_best.parent.mkdir(parents=True, exist_ok=True)
    with save_best.open("w", encoding="utf-8") as f:
        json.dump(best["preds"], f)
    print(f"Saved best predictions to: {save_best}")

    save_meta = Path(args.save_meta)
    with save_meta.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in best.items() if k != "preds"}, f, indent=2)
    print(f"Saved best config to: {save_meta}")


if __name__ == "__main__":
    main()
