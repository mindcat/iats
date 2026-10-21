"""Comparison table over every runs/*/eval.json: full gold test split and a contamination-safe
subset whose toki pona sentence first appeared on Tatoeba after the newest baseline checkpoint
(2023-05-02), so neither M2M-100 nor GPT-2 Alpaca could have trained on it.

    uv run python compare.py
"""

import bz2
import csv
import glob
import json
import os
import re
import sys

from eval_mt import score

HERE = os.path.dirname(os.path.abspath(__file__))
DETAILED = os.path.join(HERE, "..", "tp-dataset", "corpus", "raw", "tatoeba", "tok_sentences_detailed.tsv.bz2")
CUTOFF = "2023-05-02"


def norm(s):
    return re.sub(r"\s+", " ", s.strip())


def first_seen():
    csv.field_size_limit(sys.maxsize)
    first = {}
    with bz2.open(DETAILED, "rt") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            date = r[4] if r[4][:4].isdigit() and not r[4].startswith("0000") else "0000"
            first[norm(r[2])] = min(first.get(norm(r[2]), "9999"), date)
    return first


def main():
    first = first_seen()
    table = {}
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "*", "eval.json"))):
        name = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            samples = json.load(f)["samples"]
        for key, rows in samples.items():
            src, tgt = key.split("-")
            tok_side = [r["src"] if src == "tok" else r["ref"] for r in rows]
            post = [r for r, t in zip(rows, tok_side) if first.get(norm(t), "0000") > CUTOFF]
            for subset, rs in (("all", rows), ("post2023", post)):
                if rs:
                    m = score([r["hyp"] for r in rs], [r["ref"] for r in rs], tgt)
                    table.setdefault(name, {})[f"{key}/{subset}"] = m
    with open(os.path.join(HERE, "runs", "comparison.json"), "w") as f:
        json.dump(table, f, indent=1)
    cols = sorted({c for v in table.values() for c in v})
    print(f"{'run':28s}" + "".join(f"{c:>22s}" for c in cols))
    for name, v in table.items():
        cells = []
        for c in cols:
            m = v.get(c)
            cells.append(f"{m['bleu']:6.1f}/{m['chrf++']:5.1f}" + (f"/{100 * m['grammatical_rate']:4.1f}" if m and 'grammatical_rate' in m else "") if m else "-")
        print(f"{name:28s}" + "".join(f"{x:>22s}" for x in cells))
    print("cells: BLEU/chrF++[/grammatical %]")


if __name__ == "__main__":
    main()
