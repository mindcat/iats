"""Unsupervised word alignment (eflomal) of all tok-en pairs (no labels are used, so
including held-out text is harmless; only train-split links supervise training).

These links supervise the attention linker (guided alignment loss); they are statistical,
not dictionary-derived, so evaluating the linker against dictionary links stays fair.
Also reports eflomal's own top-1 agreement with the dictionary links on the test split.

    uv run python align_corpus.py     # -> cache/align_tok_en.jsonl
"""

import json
import os
import re
import tempfile
from collections import defaultdict

from eflomal import Aligner

import mt_data
from linker import load_gloss

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = re.compile(r"[A-Za-z0-9']+")


def words(text, lower=True):
    return [w.lower() if lower else w for w in WORD.findall(text)]


def read_links(path):
    with open(path) as f:
        return [{tuple(map(int, p.split("-"))) for p in line.split()} for line in f]


def symmetrize(fwd, rev):
    """Intersection, then any union link that covers a still-unaligned word (grow-final style)."""
    out = set(fwd & rev)
    src_done = {i for i, _ in out}
    tgt_done = {j for _, j in out}
    for i, j in sorted(fwd | rev):
        if i not in src_done or j not in tgt_done:
            out.add((i, j))
            src_done.add(i)
            tgt_done.add(j)
    return sorted(out)


def main():
    rows = [r for r in mt_data.load_jsonl("parallel.jsonl") if r["en"] and r["unit"] != "document"]
    src = [" ".join(words(r["tok"])) or "_" for r in rows]
    tgt = [" ".join(words(r["en"])) or "_" for r in rows]
    with tempfile.TemporaryDirectory() as tmp:
        fwd_p, rev_p = os.path.join(tmp, "fwd"), os.path.join(tmp, "rev")
        Aligner().align(src, tgt, links_filename_fwd=fwd_p, links_filename_rev=rev_p, quiet=True)
        fwd, rev = read_links(fwd_p), read_links(rev_p)

    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    gloss = load_gloss()
    hits = total = 0
    with open(os.path.join(HERE, "cache", "align_tok_en.jsonl"), "w") as f:
        for r, a, b in zip(rows, fwd, rev):
            links = symmetrize(a, b)
            f.write(json.dumps({"tok": r["tok"], "en": r["en"], "split": r["split"], "links": links}) + "\n")
            if r["split"] != "test":
                continue
            # agreement with dictionary links: target word -> unique dictionary source word
            sw, tw = words(r["tok"]), words(r["en"])
            by_t = defaultdict(set)
            for i, j in links:
                by_t[j].add(i)
            for j, t in enumerate(tw):
                cands = {i for i, s in enumerate(sw) if t in gloss.get(s, ()) or t.rstrip("s") in gloss.get(s, ())}
                if len(cands) == 1 and sw.count(sw[next(iter(cands))]) == 1:
                    total += 1
                    hits += next(iter(cands)) in by_t.get(j, ())
    print(f"aligned {len(rows)} pairs; eflomal vs dictionary links on test: {hits}/{total} = {hits / max(total, 1):.3f}")


if __name__ == "__main__":
    main()
