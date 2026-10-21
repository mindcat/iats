"""YOLOv8 glyph/cartouche detector on the synthetic scenes, plus end-to-end prediction dump.

    uv run python train_yolo.py train --model yolov8n.pt --epochs 60
    uv run python train_yolo.py predict --weights runs/detect/glyphs/weights/best.pt --split test
    uv run python evaluate_grouping.py data/scenes/meta/test.jsonl --pred data/scenes/pred_test.jsonl
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.join(HERE, "data", "scenes")


def train(args):
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=os.path.join(SCENES, "dataset.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=os.path.join(HERE, "runs", "detect"), name=args.name, exist_ok=True,
                # mirrored glyphs are not sitelen pona; scenes are already perspective-augmented
                fliplr=0.0, degrees=5.0, mosaic=1.0, close_mosaic=10, max_det=600, workers=16,
                patience=20, plots=True)
    metrics = model.val(data=os.path.join(SCENES, "dataset.yaml"), split="test", imgsz=args.imgsz, max_det=600)
    print(json.dumps({"test_map50": float(metrics.box.map50), "test_map50_95": float(metrics.box.map),
                      "per_class_map50": [float(v) for v in metrics.box.ap50]}))


def iou_matrix(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


def predict(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)
    scenes = [json.loads(line) for line in open(os.path.join(SCENES, "meta", f"{args.split}.jsonl"))]
    out = open(os.path.join(SCENES, f"pred_{args.split}.jsonl"), "w")
    for k in range(0, len(scenes), 32):
        batch = scenes[k:k + 32]
        paths = [os.path.join(SCENES, "images", args.split, s["image"] + ".jpg") for s in batch]
        for scene, res in zip(batch, model.predict(paths, imgsz=args.imgsz, conf=args.conf, max_det=600,
                                                    verbose=False)):
            boxes = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            dets = [{"x": float(b[0] + b[2]) / 2, "y": float(b[1] + b[3]) / 2, "w": float(b[2] - b[0]),
                     "h": float(b[3] - b[1]), "cartouche": bool(c == 1), "glyph_id": None} for b, c in zip(boxes, cls)]
            gt = scene["dets"]
            gt_index = [-1] * len(dets)
            if dets and gt:
                gt_boxes = [[d["x"] - d["w"] / 2, d["y"] - d["h"] / 2, d["x"] + d["w"] / 2, d["y"] + d["h"] / 2]
                            for d in gt]
                iou = iou_matrix(boxes, gt_boxes)
                same = np.array([[p["cartouche"] == g["cartouche"] for g in gt] for p in dets])
                cost = -(iou * same)
                for i, j in zip(*linear_sum_assignment(cost)):
                    if iou[i, j] * same[i, j] >= 0.5:
                        gt_index[i] = int(j)
                        dets[i]["glyph_id"] = gt[j]["glyph_id"]
            out.write(json.dumps({"image": scene["image"], "dets": dets, "gt_index": gt_index}) + "\n")
    out.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", default="yolov8n.pt")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--name", default="glyphs")
    p = sub.add_parser("predict")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()
    {"train": train, "predict": predict}[args.cmd](args)


if __name__ == "__main__":
    main()
