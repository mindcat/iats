"""Synthetic sitelen pona scenes for YOLO glyph/cartouche detection and layout grouping.

Each scene mixes
  - 1-2 multi-line text boxes (2-6 lines from tp-dataset/corpus/mono.jsonl, word-wrapped,
    names drawn as acrophonic cartouches, optional card / chat-bubble background)
  - 1-4 independent stray glyphs or short 2-6 glyph captions (other size/angle/place)
  - Latin / CJK / icon distractors (unlabeled background)
  - camera augmentation: perspective, rotation, shadow, blur, noise, JPEG

Outputs a YOLO dataset (class 0 glyph, 1 cartouche) plus meta/<split>.jsonl with the
ground-truth region -> paragraph -> line hierarchy used to score heuristic_grouping.py.

    uv run python generate_seg_dataset.py --train 20000 --val 1000 --test 1000
"""

import argparse
import json
import math
import os
import random
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
OCR_DEMO = os.path.join(HERE, "..", "ocr-demo")
CORPUS = os.path.join(HERE, "..", "tp-dataset", "corpus", "mono.jsonl")
sys.path.insert(0, OCR_DEMO)

import generate_dataset as gd  # noqa: E402  (font audit + negative fonts are shared with Task 1)

WORD_RE = re.compile(r"[A-Za-z]+")
LETTERS = set("aeijklmnopstuw")


# ------------------------------------------------------------------ text sources

def load_texts():
    long_texts, short_texts = [], []
    with open(CORPUS) as f:
        for line in f:
            m = json.loads(line)
            if m["source"] == "tatoeba":
                if 2 <= m["n_words"] <= 6:
                    short_texts.append(m["text"])
            elif m["n_words"] >= 12:
                long_texts.append(m["text"])
    return long_texts, short_texts


def to_items(text, word_ids, words_by_letter, rng, max_name=6):
    """Text -> render items: ("glyph", gid) or ("cartouche", [gid, ...]) with sentence breaks."""
    items = []
    for tok in WORD_RE.findall(text):
        low = tok.lower()
        if low in word_ids:
            items.append(("glyph", word_ids[low]))
        elif tok[0].isupper() and set(low) <= LETTERS:
            # acrophony: each glyph contributes its first letter (Sonja = soweli olin nasin jan a);
            # consecutive vowels after a consonant collapse into the syllable, as in practice
            sylls = re.findall(r"[jklmnpstw]?[aeiou]n?(?![aeiou])|n", low) or [low[0]]
            gids = [rng.choice(words_by_letter[s[0]]) for s in sylls[:max_name] if s[0] in words_by_letter]
            if gids:
                items.append(("cartouche", gids))
    return items


# ------------------------------------------------------------------ rendering

class GlyphRenderer:
    def __init__(self, support_list, rng):
        self.support = support_list
        self.rng = rng
        self.cache = {}

    def mask(self, gid, font_path, size):
        """Tight-cropped glyph ink mask (uint8) at roughly `size` px tall/wide."""
        key = (gid, font_path, size)
        if key not in self.cache:
            text = dict(self.support[gid])[font_path]
            font = ImageFont.truetype(font_path, int(size * 1.4))
            img = Image.new("L", (int(size * 3), int(size * 3)), 0)
            ImageDraw.Draw(img).text((size * 0.5, size * 0.5), text, font=font, fill=255)
            bbox = img.getbbox()
            img = img.crop(bbox) if bbox else Image.new("L", (2, 2), 0)
            scale = size / max(img.size)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                             Image.Resampling.LANCZOS)
            if len(self.cache) > 4000:
                self.cache.clear()
            self.cache[key] = np.array(img)
        return self.cache[key]

    def fonts_for(self, gids):
        sets = [set(dict(self.support[g])) for g in gids]
        common = set.intersection(*sets) if sets else set()
        return sorted(common)


class Layer:
    """A group of glyphs drawn into its own canvas so it can be rotated as a unit."""

    def __init__(self, w, h):
        self.ink = np.zeros((h, w), np.uint8)
        self.dets = []   # {"box": [x0, y0, x1, y1], "cartouche": bool, "glyph_id": int|None}
        self.lines = []  # paragraphs -> lines -> det indices (layer-local)

    def paste(self, m, x, y):
        h, w = m.shape
        x, y = int(round(x)), int(round(y))
        region = self.ink[y:y + h, x:x + w]
        np.maximum(region, m[:region.shape[0], :region.shape[1]], out=region)
        ys, xs = np.nonzero(m > 64)
        if len(xs) == 0:
            return [x, y, x + w, y + h]
        return [x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1]


def layout_lines(items, renderer, font, size, max_width, spacing, rng):
    """Greedy word wrap. Returns lines of placed item specs with widths."""
    def item_width(it):
        if it[0] == "glyph":
            return size
        inner = size * 0.78
        return len(it[1]) * inner * (1 + spacing * 0.5) + size * 0.5
    lines, cur, cur_w = [], [], 0.0
    gap = size * spacing
    for it in items:
        w = item_width(it)
        if cur and cur_w + gap + w > max_width:
            lines.append(cur)
            cur, cur_w = [], 0.0
        cur.append((it, w))
        cur_w += (gap if len(cur) > 1 else 0) + w
    if cur:
        lines.append(cur)
    return lines


def draw_items_line(layer, line, renderer, font, size, x0, y_center, spacing):
    """Draws one line; returns det indices in reading order."""
    x = x0
    gap = size * spacing
    idx = []
    for it, w in line:
        if it[0] == "glyph":
            m = renderer.mask(it[1], font, size)
            box = layer.paste(m, x + (size - m.shape[1]) / 2, y_center - m.shape[0] / 2)
            layer.dets.append({"box": box, "cartouche": False, "glyph_id": it[1]})
            idx.append(len(layer.dets) - 1)
        else:
            inner = int(size * 0.78)
            pad = size * 0.25
            gx = x + pad
            inner_idx = []
            for gid in it[1]:
                m = renderer.mask(gid, font, inner)
                box = layer.paste(m, gx + (inner - m.shape[1]) / 2, y_center - m.shape[0] / 2)
                layer.dets.append({"box": box, "cartouche": False, "glyph_id": gid})
                inner_idx.append(len(layer.dets) - 1)
                gx += inner * (1 + spacing * 0.5)
            # cartouche outline: rounded rectangle around the name glyphs
            x1 = x + w
            half = size * 0.55
            stroke = max(1, int(size / 14))
            img = Image.fromarray(layer.ink)
            ImageDraw.Draw(img).rounded_rectangle([x, y_center - half, x1, y_center + half],
                                                  radius=int(half * 0.9), outline=255, width=stroke)
            layer.ink = np.array(img)
            layer.dets.append({"box": [x, y_center - half, x1, y_center + half], "cartouche": True,
                               "glyph_id": None})
            idx.extend(inner_idx)
        x += w + gap
    return idx


def build_text_block(items, renderer, font, size, max_glyphs_per_line, spacing, line_gap, rng,
                     center=False, para_break=None):
    lines = layout_lines(items, renderer, font, size, max_glyphs_per_line * size * (1 + spacing), spacing, rng)
    widths = [sum(w for _, w in ln) + size * spacing * (len(ln) - 1) for ln in lines]
    W = int(max(widths) + size)
    extra = [0.0] * len(lines)
    paras = [[0]]
    for k in range(1, len(lines)):
        if para_break and k in para_break and 2 <= k <= len(lines) - 2:
            extra[k] = size * rng.uniform(0.9, 1.6)
            paras.append([])
        paras[-1].append(k)
    H = int(size * 0.5 + sum(size * line_gap + e for e in extra) + size)
    layer = Layer(W + int(size), H + int(size))
    y = size
    line_dets = []
    for k, ln in enumerate(lines):
        y += extra[k]
        x0 = size * 0.5 + ((W - size - widths[k]) / 2 if center else 0)
        line_dets.append(draw_items_line(layer, ln, renderer, font, size, x0, y, spacing))
        y += size * line_gap
    layer.lines = [[line_dets[k] for k in p] for p in paras]
    return layer


# ------------------------------------------------------------------ scene

def rotate_layer(layer, angle):
    h, w = layer.ink.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    layer.ink = cv2.warpAffine(layer.ink, M, (nw, nh), flags=cv2.INTER_LINEAR)
    for d in layer.dets:
        d["box"] = _transform_box(d["box"], lambda p: M @ np.array([p[0], p[1], 1.0]))
    return layer


def _transform_box(box, f):
    x0, y0, x1, y1 = box
    pts = np.array([f(p) for p in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))])
    # rotated glyph boxes: shrink the AABB of rotated corners halfway back toward the
    # rotated center so the label hugs the ink instead of the rotated square
    cx, cy = pts.mean(0)
    w0, h0 = x1 - x0, y1 - y0
    bw, bh = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
    w, h = (bw + w0) / 2, (bh + h0) / 2
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def place(occupied, w, h, W, H, rng, margin, tries=60):
    for _ in range(tries):
        x = rng.uniform(margin, max(margin, W - w - margin))
        y = rng.uniform(margin, max(margin, H - h - margin))
        r = (x - margin, y - margin, x + w + margin, y + h + margin)
        if x + w <= W and y + h <= H and all(r[2] <= o[0] or r[0] >= o[2] or r[3] <= o[1] or r[1] >= o[3]
                                             for o in occupied):
            occupied.append((x, y, x + w, y + h))
            return x, y
    return None


def background(W, H, rng, dark):
    base = rng.uniform(25, 70) if dark else rng.uniform(190, 255)
    tint = np.array([rng.uniform(-12, 12) for _ in range(3)])
    img = np.ones((H, W, 3), np.float32) * (base + tint)
    gx = np.linspace(-1, 1, W)[None, :, None] * rng.uniform(-25, 25)
    gy = np.linspace(-1, 1, H)[:, None, None] * rng.uniform(-25, 25)
    img += gx + gy
    if rng.random() < 0.3:  # ruled paper
        step = rng.randint(24, 60)
        col = base - 40 if not dark else base + 30
        for y in range(rng.randint(0, step), H, step):
            img[y:y + 1] = col
    return img


def ink_colour(rng, dark):
    if dark:
        return np.array([rng.uniform(200, 255)] * 3) + np.array([rng.uniform(-20, 0) for _ in range(3)])
    if rng.random() < 0.2:
        return np.array([rng.uniform(0, 120), rng.uniform(0, 80), rng.uniform(60, 160)])
    return np.array([rng.uniform(0, 60)] * 3)


def composite(img, layer, x, y, colour, alpha=1.0):
    h, w = layer.ink.shape
    H, W = img.shape[:2]
    x, y = int(x), int(y)
    w2, h2 = min(w, W - x), min(h, H - y)
    a = layer.ink[:h2, :w2, None].astype(np.float32) / 255 * alpha
    img[y:y + h2, x:x + w2] = img[y:y + h2, x:x + w2] * (1 - a) + colour * a


def draw_panel(img, box, rng, dark):
    """Card / chat bubble behind a text box."""
    x0, y0, x1, y1 = [int(v) for v in box]
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    fill = tuple(int(v) for v in (np.array([rng.uniform(150, 255)] * 3) if not dark
                                   else np.array([rng.uniform(40, 90)] * 3)))
    outline = tuple(int(v) for v in np.array([rng.uniform(60, 160)] * 3)) if rng.random() < 0.6 else None
    d.rounded_rectangle([x0, y0, x1, y1], radius=rng.randint(4, 30), fill=fill, outline=outline,
                        width=rng.randint(1, 3))
    return np.array(pil).astype(np.float32)


def draw_distractor(img, occupied, W, H, rng, neg_fonts, size, dark):
    kind = rng.random()
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    col = tuple(int(c) for c in ink_colour(rng, dark))
    if kind < 0.7 and neg_fonts:
        if rng.random() < 0.5:
            text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                           for _ in range(rng.randint(2, 9)))
        else:
            text = "".join(chr(rng.randint(0x4E00, 0x9FA5)) for _ in range(rng.randint(1, 5)))
        fonts = [fp for fp, cmap in neg_fonts if all(ord(c) in cmap for c in text)]
        if not fonts:
            return img
        font = ImageFont.truetype(rng.choice(fonts), int(size * rng.uniform(0.6, 1.4)))
        bb = d.textbbox((0, 0), text, font=font)
        pos = place(occupied, bb[2] - bb[0], bb[3] - bb[1], W, H, rng, size * 0.6)
        if pos:
            d.text((pos[0] - bb[0], pos[1] - bb[1]), text, font=font, fill=col)
    else:
        s = int(size * rng.uniform(0.8, 1.6))
        pos = place(occupied, s, s, W, H, rng, size * 0.6)
        if pos:
            x, y = pos
            shape = rng.choice(["star", "arrow", "rect", "heart"])
            if shape == "rect":
                d.rectangle([x, y, x + s, y + s * 0.7], outline=col, width=2)
            elif shape == "arrow":
                d.line([x, y + s / 2, x + s, y + s / 2], fill=col, width=3)
                d.polygon([(x + s, y + s / 2), (x + s * 0.7, y + s * 0.25), (x + s * 0.7, y + s * 0.75)], fill=col)
            else:
                n = 5 if shape == "star" else 24
                pts = []
                for k in range(2 * n):
                    r = s / 2 if k % 2 == 0 else (s / 4.5 if shape == "star" else s / 2.2)
                    t = math.pi * k / n
                    pts.append((x + s / 2 + r * math.sin(t), y + s / 2 - r * math.cos(t)))
                d.polygon(pts, fill=col)
    return np.array(pil).astype(np.float32)


def camera(img, dets, rng):
    H, W = img.shape[:2]
    j = 0.06 * min(W, H)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    angle = math.radians(rng.uniform(-6, 6))
    c, s = math.cos(angle), math.sin(angle)
    dst = (dst - [W / 2, H / 2]) @ np.float32([[c, s], [-s, c]]) + [W / 2, H / 2]
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    border = tuple(float(v) for v in img[0, 0])
    img = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    def f(p):
        v = M @ np.array([p[0], p[1], 1.0])
        return v[:2] / v[2]
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        pts = np.array([f(p) for p in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))])
        d["box"] = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]

    if rng.random() < 0.5:  # soft shadow
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        yy, xx = np.mgrid[0:H, 0:W]
        r = rng.uniform(0.3, 0.8) * max(W, H)
        shade = 1 - rng.uniform(0.15, 0.45) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
        img *= shade[..., None]
    if rng.random() < 0.25:
        k = rng.choice([3, 5, 7])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1 / k
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1),
                                (k, k))
        img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
    elif rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.3))
    img += np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, rng.uniform(2, 9), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 95)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), dets


# ------------------------------------------------------------------ worker

_W = {}


def _init(support_list, word_ids, long_texts, short_texts, neg_fonts):
    words_by_letter = {}
    for w, gid in word_ids.items():
        words_by_letter.setdefault(w[0], []).append(gid)
    _W.update(support=support_list, word_ids=word_ids, long=long_texts, short=short_texts,
              neg_fonts=neg_fonts, by_letter=words_by_letter)


def make_scene(seed):
    rng = random.Random(seed)
    W = H = rng.choice([640, 640, 1024])
    dark = rng.random() < 0.15
    renderer = GlyphRenderer(_W["support"], rng)
    img = background(W, H, rng, dark)
    occupied, dets, regions = [], [], []
    main_size = rng.uniform(0.028, 0.06) * W

    def add_layer(layer, x, y, kind, colour):
        base = len(dets)
        for d in layer.dets:
            x0, y0, x1, y1 = d["box"]
            dets.append({**d, "box": [x0 + x, y0 + y, x1 + x, y1 + y]})
        regions.append({"type": kind, "paragraphs": [[[base + i for i in line] for line in para]
                                                      for para in layer.lines]})
        composite(img_ref[0], layer, x, y, colour)

    img_ref = [img]
    # 1) structured text boxes
    for _ in range(rng.choice([1, 1, 2])):
        size = main_size * rng.uniform(0.85, 1.15)
        for _attempt in range(10):
            items = to_items(rng.choice(_W["long"]), _W["word_ids"], _W["by_letter"], rng)
            per_line = rng.randint(5, 12)
            n_lines = rng.randint(2, 6)
            items = items[:per_line * n_lines]
            if len(items) >= per_line + 2:
                break
        fonts = renderer.fonts_for([g for it in items for g in ([it[1]] if it[0] == "glyph" else it[1])])
        if not fonts:
            continue
        spacing, line_gap = rng.uniform(0.1, 0.35), rng.uniform(1.25, 1.7)
        # a paragraph break needs two lines on each side, else it is just two captions
        breaks = {rng.randint(2, n_lines - 2)} if n_lines >= 4 and rng.random() < 0.35 else None
        layer = build_text_block(items, renderer, rng.choice(fonts), int(size), per_line, spacing, line_gap,
                                 rng, center=rng.random() < 0.25, para_break=breaks)
        if len(layer.lines) == 1 and len(layer.lines[0]) < 2:
            continue
        if rng.random() < 0.3:
            layer = rotate_layer(layer, rng.uniform(-12, 12))
        h, w = layer.ink.shape
        pos = place(occupied, w, h, W, H, rng, size * 1.5)
        if not pos:
            continue
        if rng.random() < 0.4:
            img_ref[0] = draw_panel(img_ref[0], (pos[0] - size * 0.2, pos[1] - size * 0.2,
                                                 pos[0] + w + size * 0.2, pos[1] + h + size * 0.2), rng, dark)
        n_lines_drawn = sum(len(p) for p in layer.lines)
        kind = "text_box" if n_lines_drawn > 1 or len(layer.dets) > 6 else "short_sentence"
        add_layer(layer, pos[0], pos[1], kind, ink_colour(rng, dark))

    # 2) independent stray glyphs / short captions
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.45:
            gid = rng.randrange(len(_W["support"]))
            items = [("glyph", gid)]
            size = main_size * rng.choice([rng.uniform(0.6, 0.9), rng.uniform(1.9, 3.0)])
        else:
            items = to_items(rng.choice(_W["short"]), _W["word_ids"], _W["by_letter"], rng)[:6]
            if not items:
                continue
            # captions differ in scale from body text, so they never join a text box
            size = main_size * rng.choice([rng.uniform(0.5, 0.6), rng.uniform(1.8, 2.4)])
        fonts = renderer.fonts_for([g for it in items for g in ([it[1]] if it[0] == "glyph" else it[1])])
        if not fonts:
            continue
        layer = build_text_block(items, renderer, rng.choice(fonts), max(12, int(size)), 99,
                                 rng.uniform(0.1, 0.3), 1.4, rng)
        if rng.random() < 0.5:
            layer = rotate_layer(layer, rng.uniform(-25, 25))
        h, w = layer.ink.shape
        pos = place(occupied, w, h, W, H, rng, max(size, main_size) * 1.6)
        if not pos:
            continue
        kind = "single_glyph" if len(layer.dets) == 1 else "short_sentence"
        add_layer(layer, pos[0], pos[1], kind, ink_colour(rng, dark))

    # 3) distractors
    for _ in range(rng.randint(0, 4)):
        img_ref[0] = draw_distractor(img_ref[0], occupied, W, H, rng, _W["neg_fonts"], main_size, dark)

    img, dets = camera(img_ref[0], dets, rng)
    return seed, img, dets, regions


# ------------------------------------------------------------------ main

def write_split(pool, split, n, seed0, out):
    img_dir = os.path.join(out, "images", split)
    lbl_dir = os.path.join(out, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    meta = open(os.path.join(out, "meta", f"{split}.jsonl"), "w")
    for seed, img, dets, regions in tqdm(pool.imap_unordered(make_scene, range(seed0, seed0 + n), chunksize=8),
                                         total=n, desc=split):
        name = f"{split}_{seed:07d}"
        H, W = img.shape[:2]
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines, out_dets = [], []
        for d in dets:
            x0, y0, x1, y1 = (max(0, d["box"][0]), max(0, d["box"][1]), min(W, d["box"][2]), min(H, d["box"][3]))
            out_dets.append({"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "w": x1 - x0, "h": y1 - y0,
                             "cartouche": d["cartouche"], "glyph_id": d["glyph_id"]})
            lines.append(f"{int(d['cartouche'])} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                         f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        meta.write(json.dumps({"image": name, "width": W, "height": H, "dets": out_dets,
                               "regions": regions}) + "\n")
    meta.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "scenes"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    words = gd.load_wordlist()
    infos = {w: gd.load_word_info(w) for w in words}
    font_paths = sorted(os.path.join(gd.FONT_DIR, f) for f in os.listdir(gd.FONT_DIR) if f.endswith((".ttf", ".otf")))
    support, _ = gd.audit_fonts(words, infos, font_paths)
    classes = [w for w in words if support[w]]
    support_list = [support[w] for w in classes]
    word_ids = {w: i for i, w in enumerate(classes)}
    long_texts, short_texts = load_texts()
    neg_fonts = gd.negative_fonts()

    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)
    with open(os.path.join(args.out, "dataset.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                "names:\n  0: glyph\n  1: cartouche\n")
    with Pool(args.workers, initializer=_init,
              initargs=(support_list, word_ids, long_texts, short_texts, neg_fonts)) as pool:
        for split, n, seed0 in (("train", args.train, 0), ("val", args.val, 10_000_000),
                                ("test", args.test, 20_000_000)):
            if n:
                write_split(pool, split, n, seed0, args.out)


if __name__ == "__main__":
    main()
