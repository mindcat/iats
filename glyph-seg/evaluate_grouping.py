"""Scores heuristic_grouping.py against generate_seg_dataset.py ground truth.

  region F1        a predicted region counts only if its glyph set equals a GT region's
  type accuracy    text_box / short_sentence / single_glyph label of matched regions
  line exact       GT lines reproduced with identical glyphs and order
  Kendall's tau    reading order of each GT region's glyphs in the prediction (mean over regions)

    uv run python evaluate_grouping.py data/scenes/meta/test.jsonl              # oracle boxes
    uv run python evaluate_grouping.py data/scenes/meta/test.jsonl --pred preds.jsonl
"""

import argparse
import json
from collections import Counter

import numpy as np
from scipy.stats import kendalltau

from heuristic_grouping import group


def gt_structure(scene):
    regions = []
    for r in scene["regions"]:
        lines = [line for para in r["paragraphs"] for line in para]
        regions.append({"type": r["type"], "glyphs": [g for line in lines for g in line], "lines": lines,
                        "n_paras": len(r["paragraphs"])})
    return regions


def pred_structure(result, dets, to_gt=None):
    """to_gt maps detector indices to GT indices (-1 = false positive, dropped)."""
    to_gt = to_gt or list(range(len(dets)))
    regions = []
    for r in result["regions"]:
        lines = [[to_gt[g["det"]] for g in line if not dets[g["det"]].get("cartouche") and to_gt[g["det"]] >= 0]
                 for p in r["paragraphs"] for line in p["lines"]]
        lines = [line for line in lines if line]
        regions.append({"type": r["type"], "glyphs": [g for line in lines for g in line], "lines": lines,
                        "n_paras": len(r["paragraphs"])})
    return [r for r in regions if r["glyphs"]]


def score_scene(gt, pred, stats):
    gt_sets = {frozenset(r["glyphs"]): r for r in gt}
    pred_sets = {frozenset(r["glyphs"]): r for r in pred}
    stats["gt_regions"] += len(gt)
    stats["pred_regions"] += len(pred)
    for key, g in gt_sets.items():
        p = pred_sets.get(key)
        if p is None:
            stats[f"missed_{g['type']}"] += 1
            continue
        stats["matched"] += 1
        stats[f"matched_{g['type']}"] += 1
        stats["type_correct"] += p["type"] == g["type"]
        stats["para_count_correct"] += p["n_paras"] == g["n_paras"]
    for g in gt:
        stats[f"gt_{g['type']}"] += 1
    pred_lines = {tuple(line) for r in pred for line in r["lines"]}
    for g in gt:
        for line in g["lines"]:
            stats["gt_lines"] += 1
            stats["lines_exact"] += tuple(line) in pred_lines
    # reading order: position of each glyph in the flattened prediction
    order = {d: k for k, d in enumerate(d for r in pred for d in r["glyphs"])}
    taus = []
    for g in gt:
        if len(g["glyphs"]) >= 2 and all(d in order for d in g["glyphs"]):
            tau = kendalltau(range(len(g["glyphs"])), [order[d] for d in g["glyphs"]]).statistic
            taus.append(1.0 if np.isnan(tau) else tau)
    return taus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meta")
    ap.add_argument("--pred", help="jsonl of {image, dets} from a detector (default: GT boxes)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    scenes = [json.loads(line) for line in open(args.meta)]
    if args.limit:
        scenes = scenes[:args.limit]
    preds = {}
    if args.pred:
        preds = {r["image"]: r for r in map(json.loads, open(args.pred))}

    stats, taus = Counter(), []
    for scene in scenes:
        dets, to_gt = scene["dets"], None
        if args.pred:
            # detector boxes were IoU-matched to GT glyphs (train_yolo.py predict)
            dets, to_gt = preds[scene["image"]]["dets"], preds[scene["image"]]["gt_index"]
            stats["gt_glyphs"] += sum(1 for d in scene["dets"] if not d["cartouche"])
            stats["detected_glyphs"] += sum(1 for d, j in zip(dets, to_gt) if j >= 0 and not d["cartouche"])
        result = group(dets)
        taus += score_scene(gt_structure(scene), pred_structure(result, dets, to_gt), stats)

    p = stats["matched"] / max(stats["pred_regions"], 1)
    r = stats["matched"] / max(stats["gt_regions"], 1)
    report = {
        "scenes": len(scenes),
        "region_precision": round(p, 4),
        "region_recall": round(r, 4),
        "region_f1": round(2 * p * r / max(p + r, 1e-9), 4),
        "type_accuracy_on_matched": round(stats["type_correct"] / max(stats["matched"], 1), 4),
        "paragraph_count_accuracy": round(stats["para_count_correct"] / max(stats["matched"], 1), 4),
        "line_exact_rate": round(stats["lines_exact"] / max(stats["gt_lines"], 1), 4),
        "kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else None,
        "recall_by_type": {t: round(stats[f"matched_{t}"] / max(stats[f"gt_{t}"], 1), 4)
                           for t in ("text_box", "short_sentence", "single_glyph")},
    }
    if args.pred:
        report["glyph_detection_recall"] = round(stats["detected_glyphs"] / max(stats["gt_glyphs"], 1), 4)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
