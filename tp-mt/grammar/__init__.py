"""Grammaticality via Huang & Joo's (2025) nearley CFG (cubedhuang/ilo-nasin-sin).

A text is grammatical when every sentence in it has at least one Earley parse.
Build the bundle once with: cd grammar && npm install && npm run build
"""

import json
import os
import re
import subprocess
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "nasin.cjs")
SENT_RE = re.compile(r"[.!?:;\"“”()]+")
# Earley ambiguity grows explosively on run-ons; longer sentences count as unparseable
MAX_WORDS = 40


def sentences(text):
    return [s.strip() for s in SENT_RE.split(text) if re.search(r"[A-Za-z]", s)]


def parse_counts(sents):
    if not sents:
        return []
    short = [s for s in sents if len(s.split()) <= MAX_WORDS]
    out = subprocess.run(["node", "--max-old-space-size=8192", BUNDLE],
                         input="\n".join(json.dumps(s) for s in short) + "\n",
                         capture_output=True, text=True, check=True).stdout.split()
    counts = iter(int(n) for n in out)
    return [next(counts) if len(s.split()) <= MAX_WORDS else 0 for s in sents]


def grammatical_rate(texts):
    """Fraction of texts whose sentences all parse; one node process for the whole batch."""
    per_text = [sentences(t) for t in texts]
    counts = iter(parse_counts([s for ss in per_text for s in ss]))
    # materialize each text's counts before testing, so the shared iterator stays aligned
    ok = [bool(ss) and all(n > 0 for n in [next(counts) for _ in ss]) for ss in per_text]
    return sum(ok) / max(len(ok), 1)


@lru_cache(maxsize=100_000)
def is_grammatical(text):
    ss = sentences(text)
    return bool(ss) and all(n > 0 for n in parse_counts(ss))
