"""Geometric layout parser for detected sitelen pona glyphs and cartouches.

Input: detections [{"x", "y", "w", "h", "cartouche": bool, "glyph_id": int | None}, ...]
with (x, y) the box center. Output: regions -> paragraphs -> lines -> glyphs in
reading order (see `group`).

Steps
  1. cartouche consolidation: glyphs inside a cartouche box become one macro node
  2. macro-block separation: scale-normalized anisotropic proximity graph + size
     compatibility -> connected components (text boxes / short sentences / single glyphs)
  3. line tracing per region: dominant baseline angle from horizontal neighbours,
     de-rotate, sweep left-to-right attaching each node to the row whose nearest
     member overlaps it vertically (follows gentle page curvature)
  4. paragraphs: split rows at unusually large vertical gaps (or indent + short line)
"""

import json
import math
import sys
from dataclasses import dataclass, field

import numpy as np

GAMMA_X = 1.8
GAMMA_Y = 1.9  # plan suggests ~1.6; 1.9 bridges 1.7x line spacing (size gating keeps captions apart)
MAX_SIZE_RATIO = 1.75
ROW_OVERLAP = 0.4
PARA_GAP = 1.4
# paragraphs of one text box: column-aligned blocks separated by at most this many glyph heights
PARA_MAX_GAP = 3.0
PARA_MIN_OVERLAP = 0.6


@dataclass
class Node:
    x: float
    y: float
    w: float
    h: float
    s: float  # characteristic scale
    members: list = field(default_factory=list)  # detection indices in reading order
    cartouche: int | None = None  # detection index of the cartouche box


def _inside(det, box):
    return abs(det["x"] - box["x"]) <= box["w"] / 2 and abs(det["y"] - box["y"]) <= box["h"] / 2


def consolidate(dets):
    """Step 1: fold glyphs that sit inside a cartouche into one macro node."""
    carts = [i for i, d in enumerate(dets) if d.get("cartouche")]
    owner = {}
    for c in sorted(carts, key=lambda i: dets[i]["w"] * dets[i]["h"]):
        for i, d in enumerate(dets):
            if not d.get("cartouche") and i not in owner and _inside(d, dets[c]):
                owner[i] = c
    nodes = []
    for c in carts:
        box = dets[c]
        inner = [i for i, o in owner.items() if o == c]
        vertical = box["h"] > 1.5 * box["w"]
        inner.sort(key=lambda i: dets[i]["y"] if vertical else dets[i]["x"])
        # a horizontal cartouche is one glyph tall: its height is the line scale
        s = min(box["w"], box["h"])
        nodes.append(Node(box["x"], box["y"], box["w"], box["h"], s, inner, cartouche=c))
    for i, d in enumerate(dets):
        if not d.get("cartouche") and i not in owner:
            nodes.append(Node(d["x"], d["y"], d["w"], d["h"], max(d["w"], d["h"]), [i]))
    return nodes


def _gap_distance(a, b):
    """Anisotropic layout distance; edge gaps + mean scale equals center distance for square glyphs."""
    s = (a.s + b.s) / 2
    gx = max(0.0, abs(a.x - b.x) - (a.w + b.w) / 2) + s
    gy = max(0.0, abs(a.y - b.y) - (a.h + b.h) / 2) + s
    # a pure vertical neighbour (overlapping in x) only pays its vertical gap
    if abs(a.x - b.x) < (a.w + b.w) / 2:
        gx = min(gx, abs(a.x - b.x))
    if abs(a.y - b.y) < (a.h + b.h) / 2:
        gy = min(gy, abs(a.y - b.y))
    return math.hypot(gx / (GAMMA_X * s), gy / (GAMMA_Y * s))


def components(nodes):
    """Step 2: union-find over compatible, nearby nodes."""
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if max(a.s, b.s) / max(1e-6, min(a.s, b.s)) >= MAX_SIZE_RATIO:
                continue
            if _gap_distance(a, b) <= 1.0:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _bbox(nodes):
    return (min(n.x - n.w / 2 for n in nodes), min(n.y - n.h / 2 for n in nodes),
            max(n.x + n.w / 2 for n in nodes), max(n.y + n.h / 2 for n in nodes))


def _is_block(nodes):
    """Large enough to be (part of) a text box rather than a caption or stray glyph."""
    return len(nodes) > 6


def _aligned(a, b, scale):
    """Shared left edge, shared center (centered text), or large horizontal overlap."""
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    overlap = min(ax1, bx1) - max(ax0, bx0)
    return (abs(ax0 - bx0) < scale or abs((ax0 + ax1) - (bx0 + bx1)) / 2 < scale
            or overlap > PARA_MIN_OVERLAP * min(ax1 - ax0, bx1 - bx0))


def merge_paragraph_blocks(nodes, comps):
    """Re-join paragraphs that the proximity graph split at blank lines: one block must be
    text-box sized, the other at least two glyphs; they must share a column and sit close."""
    comps = [list(c) for c in comps]
    merged = True
    while merged:
        merged = False
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                na, nb = [nodes[i] for i in comps[a]], [nodes[i] for i in comps[b]]
                if not ((_is_block(na) and len(nb) >= 2) or (_is_block(nb) and len(na) >= 2)):
                    continue
                sa, sb = np.median([n.s for n in na]), np.median([n.s for n in nb])
                if max(sa, sb) / min(sa, sb) >= MAX_SIZE_RATIO:
                    continue
                ba, bb = _bbox(na), _bbox(nb)
                vgap = max(bb[1] - ba[3], ba[1] - bb[3])
                if _aligned(ba, bb, (sa + sb) / 2) and vgap <= PARA_MAX_GAP * (sa + sb) / 2:
                    comps[a] += comps.pop(b)
                    merged = True
                    break
            if merged:
                break
    return comps


def baseline_angle(nodes):
    """Median angle of left-to-right horizontal-neighbour vectors."""
    angles = []
    for a in nodes:
        best = None
        for b in nodes:
            if b is a:
                continue
            s = (a.s + b.s) / 2
            dx = b.x - a.x
            dy = b.y - a.y
            gap = dx - (a.w + b.w) / 2 + s  # edge gap based, so long cartouches qualify
            if dx > 0 and abs(dy) < 0.6 * s and 0.5 * s <= gap <= 2.2 * s:
                if best is None or dx < best[0]:
                    best = (dx, dy)
        if best:
            angles.append(math.atan2(best[1], best[0]))
    return float(np.median(angles)) if angles else 0.0


def trace_rows(nodes, theta):
    """Step 3: de-rotate and sweep left-to-right, chaining nodes into rows."""
    c, s_ = math.cos(-theta), math.sin(-theta)
    rot = [(n.x * c - n.y * s_, n.x * s_ + n.y * c) for n in nodes]
    rows = []  # each: list of node indices, kept in x' order
    for i in sorted(range(len(nodes)), key=lambda k: rot[k][0]):
        xi, yi = rot[i]
        hi = 0.4 * nodes[i].s
        best, best_score = None, 0.0
        for r, row in enumerate(rows):
            j = row[-1]  # nearest member so far along the sweep
            xj, yj = rot[j]
            hj = 0.4 * nodes[j].s
            overlap = min(yi + hi, yj + hj) - max(yi - hi, yj - hj)
            frac = overlap / (2 * min(hi, hj))
            # rows must not reach across large horizontal gaps or backwards
            reach = xi - xj - (nodes[i].w + nodes[j].w) / 2
            if frac > ROW_OVERLAP and reach < 3.0 * max(nodes[i].s, nodes[j].s) and frac > best_score:
                best, best_score = r, frac
        if best is None:
            rows.append([i])
        else:
            rows[best].append(i)
    rows.sort(key=lambda row: np.mean([rot[i][1] for i in row]))
    return rows, rot


def split_paragraphs(rows, rot, nodes):
    """Step 4: break at large inter-row gaps, or indent after a short line."""
    if len(rows) < 2:
        return [rows]
    ys = [np.mean([rot[i][1] for i in row]) for row in rows]
    gaps = np.diff(ys)
    med = float(np.median(gaps))
    scale = float(np.median([n.s for n in nodes]))
    lefts = [min(rot[i][0] - nodes[i].w / 2 for i in row) for row in rows]
    rights = [max(rot[i][0] + nodes[i].w / 2 for i in row) for row in rows]
    width = max(rights) - min(lefts)
    paras, cur = [], [rows[0]]
    for r in range(1, len(rows)):
        big_gap = len(rows) >= 3 and gaps[r - 1] > PARA_GAP * med
        indent = lefts[r] - min(lefts) > 1.0 * scale and rights[r - 1] - min(lefts) < 0.7 * width
        if big_gap or indent:
            paras.append(cur)
            cur = []
        cur.append(rows[r])
    paras.append(cur)
    return paras


def group(dets):
    nodes = consolidate(dets)
    regions = []
    for comp in merge_paragraph_blocks(nodes, components(nodes)):
        sub = [nodes[i] for i in comp]
        theta = baseline_angle(sub)
        rows, rot = trace_rows(sub, theta)
        paras = split_paragraphs(rows, rot, sub)
        n_glyph_nodes = len(sub)
        if n_glyph_nodes == 1:
            kind = "single_glyph"
        elif len(rows) == 1 and n_glyph_nodes <= 6:
            kind = "short_sentence"
        else:
            kind = "text_box"
        out_paras = []
        for para in paras:
            lines = []
            for row in para:
                line = []
                for k in row:
                    node = sub[k]
                    for m in node.members:
                        d = dets[m]
                        line.append({"det": m, "glyph_id": d.get("glyph_id"), "x": d["x"], "y": d["y"],
                                     "in_cartouche": node.cartouche is not None,
                                     **({"cartouche": node.cartouche} if node.cartouche is not None else {})})
                    if node.cartouche is not None and not node.members:
                        line.append({"det": node.cartouche, "glyph_id": None, "x": node.x, "y": node.y,
                                     "in_cartouche": True, "cartouche": node.cartouche})
                lines.append(line)
            out_paras.append({"lines": lines})
        regions.append({"type": kind, "bbox": list(_bbox(sub)),
                        "angle_deg": math.degrees(theta), "paragraphs": out_paras})

    # region reading order: top-to-bottom bands of one glyph height, then left-to-right
    band = float(np.median([n.s for n in nodes])) if nodes else 1.0
    regions.sort(key=lambda r: (round(r["bbox"][1] / band), r["bbox"][0]))
    for k, r in enumerate(regions):
        r["region_id"] = k
    return {"regions": regions}


def reading_order(result):
    """Flattened detection indices in reading order."""
    return [g["det"] for r in result["regions"] for p in r["paragraphs"] for line in p["lines"] for g in line]


if __name__ == "__main__":
    print(json.dumps(group(json.load(sys.stdin)), indent=1))
