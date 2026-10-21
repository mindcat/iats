"""Synthetic 28x28 sitelen pona glyph dataset generator.

Builds the canonical class list from tp-dataset/categories/typst.toml, audits
which fonts actually render each word (UCSUR codepoint or ligature), adds a
`not_sitelen_pona` reject class, and writes train/val .npz archives plus
class_mapping.json.

    uv run python generate_dataset.py --samples-per-class 5000 --val-samples-per-class 500
"""

import argparse
import hashlib
import json
import os
import random
import tomllib
from multiprocessing import Pool

import numpy as np
import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
TP_DATASET = os.path.join(HERE, "..", "tp-dataset")
TOML_PATH = os.path.join(TP_DATASET, "categories", "typst.toml")
WORDS_DIR = os.path.join(TP_DATASET, "words")
FONT_DIR = os.path.join(HERE, "fonts")

IMG_SIZE = 28
BASE_SIZE = 112
NEG_LABEL = "not_sitelen_pona"

# system fonts for the reject class (first existing path of each group is used)
NEGATIVE_FONT_CANDIDATES = [
    # latin
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    # cjk + hangul
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]

# shapes that are (near) identical to a sitelen pona glyph; never use as negatives
CONFUSABLE_CHARS = set("oO0QDCcUu()[]{}<>«»‹›^vV∨∧xX×+#|lI1iLJ7T⊥!.:·•-–—_=~≈∼∩∪⊂⊃◯○●△▽□■◇↓↑→←"
                       # seme is a question mark, lete an asterisk
                       "?¿*✱✳⁎∗"
                       # Greek / Cyrillic homoglyphs of the excluded Latin letters, and circled forms
                       "οΟσθΘоОсСиІіιΙνυΥχΧхХτΤтТуУ⊕⊗⊙⊖◐◑◒◓⊚⊛")


# ---------------------------------------------------------------- vocabulary

def load_wordlist(toml_path=TOML_PATH):
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return [name.removesuffix(".toml") for name in data["wordlist"]["list"]]


def load_word_info(word):
    with open(os.path.join(WORDS_DIR, f"{word}.toml"), "rb") as f:
        data = tomllib.load(f)
    sp = data.get("sitelenpona", [])
    ucsur = sp[0] if sp and len(sp[0]) == 1 and 0xF1900 <= ord(sp[0]) <= 0xF19FF else None
    return {
        "ucsur": ucsur,
        "semantic": data.get("categories", {}).get("semantic", []),
        "definition_en": data.get("definitions", {}).get("en", ""),
    }


# ---------------------------------------------------------------- font audit

class FontProbe:
    """Decides how (if at all) a font renders a given toki pona word."""

    def __init__(self, path):
        self.path = path
        self.tt = TTFont(path, fontNumber=0)
        self.cmap = self.tt.getBestCmap() or {}
        self.order = self.tt.getGlyphOrder()
        self.glyphset = self.tt.getGlyphSet()
        self.hb_font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))

    def has_ink(self, glyph_name):
        pen = BoundsPen(self.glyphset)
        self.glyphset[glyph_name].draw(pen)
        return pen.bounds is not None

    def shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {})
        return [self.order[g.codepoint] for g in buf.glyph_infos]

    def render_text_for(self, word, ucsur):
        """Returns the string to draw for `word`, or None if unsupported."""
        if ucsur and ord(ucsur) in self.cmap:
            name = self.cmap[ord(ucsur)]
            if name != ".notdef" and self.has_ink(name):
                return ucsur
        # ligature fallback: the latin spelling must shape to one inked, non-letter glyph
        glyphs = self.shape(word)
        if len(glyphs) == 1 and glyphs[0] != ".notdef" and self.has_ink(glyphs[0]):
            if not (len(word) == 1 and glyphs[0] == self.cmap.get(ord(word))
                    and glyphs[0] in (word, word.upper())):
                return word
        return None


def clean_render(text, font_path, size=64):
    """Untransformed render, cropped and fitted into IMG_SIZE (inference-style)."""
    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (BASE_SIZE * 2, BASE_SIZE * 2), 0)
    ImageDraw.Draw(img).text((BASE_SIZE // 2, BASE_SIZE // 2), text, font=font, fill=255)
    return fit_to_canvas(img, 24, center=True)


def fit_to_canvas(img, max_dim, center=False, rng=None):
    bbox = img.getbbox()
    if not bbox:
        return None
    img = img.crop(bbox)
    scale = max_dim / max(img.size)
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    img = img.resize((w, h), Image.Resampling.BILINEAR)
    out = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    if center:
        ox, oy = (IMG_SIZE - w) // 2, (IMG_SIZE - h) // 2
    else:
        ox, oy = rng.randint(0, IMG_SIZE - w), rng.randint(0, IMG_SIZE - h)
    out.paste(img, (ox, oy))
    return np.array(out)


def audit_fonts(words, infos, font_paths):
    """support[word] = [(font_path, text), ...] with duplicate/blank renders removed."""
    support = {w: [] for w in words}
    report = {}
    for fp in font_paths:
        probe = FontProbe(fp)
        hashes = {}
        texts = {}
        for w in words:
            text = probe.render_text_for(w, infos[w]["ucsur"])
            if text is None:
                continue
            arr = clean_render(text, fp)
            if arr is None or (arr > 64).sum() < 6:
                continue
            texts[w] = text
            hashes.setdefault(hashlib.md5(arr.tobytes()).hexdigest(), []).append(w)
        # identical bitmaps for different words => placeholder/tofu glyph
        dupes = {w for ws in hashes.values() if len(ws) > 1 for w in ws}
        for w, text in texts.items():
            if w not in dupes:
                support[w].append((fp, text))
        report[os.path.basename(fp)] = {
            "supported": len(texts) - len(dupes),
            "missing": [w for w in words if w not in texts],
            "duplicate_renders": sorted(dupes),
        }
    return support, report


# ---------------------------------------------------------------- negatives

def negative_char_pool():
    latin = [chr(c) for c in range(0x21, 0x7F)]
    cjk = [chr(c) for c in range(0x4E00, 0x9FA6)]
    hangul = [chr(c) for c in range(0xAC00, 0xD7A4)]
    symbols = [chr(c) for c in list(range(0x2200, 0x2300)) + list(range(0x25A0, 0x2600))
               + list(range(0x2190, 0x21FF)) + list(range(0x0391, 0x03CA)) + list(range(0x0410, 0x0450))]
    pools = {
        "latin": [c for c in latin if c not in CONFUSABLE_CHARS],
        "cjk": cjk,
        "hangul": hangul,
        "symbol": [c for c in symbols if c not in CONFUSABLE_CHARS],
    }
    return pools


def negative_fonts():
    fonts = []
    for fp in NEGATIVE_FONT_CANDIDATES:
        if os.path.exists(fp):
            tt = TTFont(fp, fontNumber=0, lazy=True)
            fonts.append((fp, set(tt.getBestCmap() or {})))
    return fonts


def draw_scribble(rng, size=BASE_SIZE):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    width = rng.randint(3, 8)
    for _ in range(rng.randint(2, 5)):
        pts = [(rng.uniform(15, size - 15), rng.uniform(15, size - 15)) for _ in range(4)]
        t = np.linspace(0, 1, 40)[:, None]
        p = np.array(pts)
        curve = ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1]
                 + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3])
        draw.line([tuple(c) for c in curve], fill=255, width=width, joint="curve")
    return img


class ConfusableFilter:
    """Rejects negative samples that look like any clean sitelen pona template."""

    def __init__(self, support, threshold):
        self.threshold = threshold
        temps = []
        for entries in support.values():
            for fp, text in entries:
                arr = clean_render(text, fp)
                if arr is not None:
                    temps.append(self._vec(arr))
        self.templates = np.stack(temps)

    @staticmethod
    def _vec(arr):
        v = np.asarray(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32).ravel()
        v -= v.mean()
        return v / (np.linalg.norm(v) + 1e-6)

    def max_similarity(self, img):
        arr = fit_to_canvas(img, 24, center=True)
        if arr is None:
            return 1.0
        return float((self.templates @ self._vec(arr)).max())

    def ok(self, img):
        return self.max_similarity(img) < self.threshold


# ---------------------------------------------------------------- augmentation

def render_base(text, font_path, rng):
    font = ImageFont.truetype(font_path, rng.randint(40, 60))
    img = Image.new("L", (BASE_SIZE, BASE_SIZE), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    if w > BASE_SIZE - 8 or h > BASE_SIZE - 8:
        img = Image.new("L", (w + 16, h + 16), 0)
        draw = ImageDraw.Draw(img)
        draw.text((8 - bbox[0], 8 - bbox[1]), text, font=font, fill=255)
    else:
        draw.text(((BASE_SIZE - w) / 2 - bbox[0], (BASE_SIZE - h) / 2 - bbox[1]), text, font=font, fill=255)
    return img


def augment(img, rng, nrng, invert_p):
    # stroke width jitter
    r = rng.random()
    if r < 0.25:
        img = img.filter(ImageFilter.MaxFilter(rng.choice([3, 5])))
    elif r < 0.40:
        eroded = img.filter(ImageFilter.MinFilter(3))
        ink = np.count_nonzero(np.asarray(img) > 128)
        if np.count_nonzero(np.asarray(eroded) > 128) >= 0.6 * ink:
            img = eroded

    # affine: rotation +-15 deg, shear, anisotropic scale
    w, h = img.size
    pad = max(w, h) // 2
    canvas = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    canvas.paste(img, (pad, pad))
    theta = np.deg2rad(rng.uniform(-15, 15))
    shx, shy = rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1)
    sx, sy = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    m = rot @ np.array([[1, shx], [shy, 1]]) @ np.diag([sx, sy])
    inv = np.linalg.inv(m)
    cx, cy = canvas.width / 2, canvas.height / 2
    off = np.array([cx, cy]) - inv @ np.array([cx, cy])
    canvas = canvas.transform(canvas.size, Image.Transform.AFFINE,
                              (inv[0, 0], inv[0, 1], off[0], inv[1, 0], inv[1, 1], off[1]),
                              resample=Image.Resampling.BILINEAR, fillcolor=0)

    arr = fit_to_canvas(canvas, IMG_SIZE - rng.randint(2, 10), rng=rng)
    if arr is None:
        return None
    out = Image.fromarray(arr)
    if rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))

    arr = np.asarray(out, dtype=np.float32)
    arr = arr * rng.uniform(0.85, 1.0)  # contrast / ink darkness
    arr = arr + nrng.normal(0.0, rng.uniform(5, 15), size=arr.shape)
    arr[nrng.random(arr.shape) < 0.02] = 0
    arr = np.clip(arr, 0, 255)
    if np.count_nonzero(arr > 128) < 12:
        return None  # faded to near-nothing; not a usable training sample
    if rng.random() < invert_p:
        arr = 255 - arr
    return arr.astype(np.uint8)


# ---------------------------------------------------------------- workers

_W = {}


def _init_worker(support_list, neg_fonts, neg_pools, filt_threshold, support, invert_p):
    _W.update(support_list=support_list, neg_fonts=neg_fonts, neg_pools=neg_pools,
              invert_p=invert_p)
    _W["filter"] = ConfusableFilter(support, filt_threshold)


def _negative_base(rng):
    kind = rng.choices(["latin", "cjk", "hangul", "symbol", "scribble"], weights=[3, 3, 2, 1.5, 1.0])[0]
    if kind == "scribble":
        return draw_scribble(rng)
    pool = _W["neg_pools"][kind]
    for _ in range(20):
        ch = rng.choice(pool)
        fonts = [fp for fp, cmap in _W["neg_fonts"] if ord(ch) in cmap]
        if fonts:
            return render_base(ch, rng.choice(fonts), rng)
    return None


def _generate_chunk(args):
    label, count, seed = args
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        if label < len(_W["support_list"]):
            fp, text = rng.choice(_W["support_list"][label])
            base = render_base(text, fp, rng)
        else:
            base = _negative_base(rng)
            if base is not None and not _W["filter"].ok(base):
                continue
        if base is None:
            continue
        arr = augment(base, rng, nrng, _W["invert_p"])
        if arr is not None:
            out.append(arr)
    return label, np.stack(out)


def generate_split(pool, n_classes, per_class, neg_count, seed, chunk=250):
    jobs = []
    for label in range(n_classes + 1):
        total = neg_count if label == n_classes else per_class
        for i in range(0, total, chunk):
            jobs.append((label, min(chunk, total - i), seed * 1_000_003 + label * 10_007 + i))
    X, y = [], []
    for label, arr in tqdm(pool.imap_unordered(_generate_chunk, jobs), total=len(jobs), desc=f"seed {seed}"):
        X.append(arr)
        y.append(np.full(len(arr), label, dtype=np.int64))
    X, y = np.concatenate(X), np.concatenate(y)
    perm = np.random.default_rng(seed).permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------- main

def build_class_mapping(words, infos, support):
    classes = [w for w in words if support[w]]
    mapping = {
        str(i): {
            "word": w,
            "ucsur_hex": f"U+{ord(infos[w]['ucsur']):05X}" if infos[w]["ucsur"] else None,
            "category": infos[w]["semantic"],
            "definition_en": infos[w]["definition_en"],
            "fonts": [os.path.basename(fp) for fp, _ in support[w]],
        }
        for i, w in enumerate(classes)
    }
    mapping[str(len(classes))] = {"word": NEG_LABEL, "ucsur_hex": None, "category": ["reject"],
                                  "definition_en": "not a sitelen pona glyph", "fonts": []}
    return classes, mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-per-class", type=int, default=5000)
    ap.add_argument("--val-samples-per-class", type=int, default=500)
    ap.add_argument("--neg-multiplier", type=float, default=4.0)
    ap.add_argument("--confusable-threshold", type=float, default=0.80)
    # inference normalizes to white-on-black first, so inverted samples only cost capacity
    ap.add_argument("--invert-p", type=float, default=0.0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    words = load_wordlist()
    infos = {w: load_word_info(w) for w in words}
    font_paths = sorted(os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith((".ttf", ".otf")))
    support, report = audit_fonts(words, infos, font_paths)

    print(f"{len(words)} words in typst.toml, {len(font_paths)} fonts")
    for name, r in report.items():
        print(f"  {name:58s} {r['supported']:4d} ok  missing={r['missing']}  dup={r['duplicate_renders']}")
    unsupported = [w for w in words if not support[w]]
    if unsupported:
        print(f"WARNING: no font renders {unsupported}; dropped from classes")

    classes, mapping = build_class_mapping(words, infos, support)
    os.makedirs(args.out_dir, exist_ok=True)
    for path in (os.path.join(HERE, "class_mapping.json"), os.path.join(args.out_dir, "class_mapping.json")):
        with open(path, "w") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "font_audit.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"{len(classes)} glyph classes + 1 reject class -> class_mapping.json")
    if args.audit_only:
        return

    neg_fonts = negative_fonts()
    print(f"{len(neg_fonts)} negative-class system fonts")
    support_list = [support[w] for w in classes]
    n = len(classes)
    with Pool(args.workers, initializer=_init_worker,
              initargs=(support_list, neg_fonts, negative_char_pool(), args.confusable_threshold,
                        support, args.invert_p)) as pool:
        for split, per_class, seed in (("train", args.samples_per_class, args.seed),
                                       ("val", args.val_samples_per_class, args.seed + 1)):
            X, y = generate_split(pool, n, per_class, int(per_class * args.neg_multiplier), seed)
            path = os.path.join(args.out_dir, f"{split}_data.npz")
            np.savez_compressed(path, X=X, y=y)
            print(f"saved {path}: X={X.shape}")


if __name__ == "__main__":
    main()
