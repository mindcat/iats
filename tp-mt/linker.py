"""Attention linker: which source words each generated target word attended to.

a[j, i] = sum over layers in [0.6L, 0.8L) and all heads of Attn(t_j, s_i), restricted to
source-word tokens (template / <bos> attention sinks excluded) and renormalized.

Evaluation uses dictionary-anchored silver links (no gold toki pona alignments exist):
a target English word w links to source word s when w is one of s's lipu Linku glosses;
only target words with exactly one such source word are scored (top-1 accuracy).

    uv run python linker.py runs/qwen2.5-0.5b --limit 400
"""

import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn

import mt_data
from eval_mt import load_run, test_pairs, translate

WORD_RE = re.compile(r"<tp:[a-z]+>|<cartouche>.*?</cartouche>|[A-Za-z0-9']+|[一-鿿]")


class AttentiveTranslatorWrapper(nn.Module):
    """Wraps a causal LM; forward returns (logits, alignment over source positions) per step.

    `source_mask` (B, T) marks prompt tokens that belong to source words, which keeps the
    exported graph shape-static: CoreML gets the alignment as a second named output.
    """

    def __init__(self, model, lo=0.6, hi=0.8):
        super().__init__()
        self.model = model
        n = model.config.num_hidden_layers
        self.layers = list(range(int(lo * n), max(int(lo * n) + 1, int(hi * n))))

    def forward(self, input_ids, attention_mask, source_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attn = torch.stack([out.attentions[k] for k in self.layers]).sum(dim=(0, 2))  # (B, T, T)
        attn = attn * source_mask[:, None, :]
        align = attn / attn.sum(-1, keepdim=True).clamp_min(1e-9)
        return out.logits, align


def char_words(text):
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(text)]


def token_word_index(offsets, words, base):
    """Maps each token (by char offset) to the index of the word it falls in, or -1."""
    idx = []
    for s, e in offsets:
        k = -1
        for w, (ws, we, _) in enumerate(words):
            if s - base < we and e - base > ws and e > s:
                k = w
                break
        idx.append(k)
    return idx


@torch.no_grad()
def align_pair(wrapper, encoder, src_lang, tgt_lang, src_text, hyp_text):
    """Word-level alignment matrix (target words x source words) for one translated pair."""
    tok = encoder.tok
    prompt = mt_data.translate_prompt(src_lang, tgt_lang)
    payload = encoder.tp.mark(src_text) if src_lang == "tok" else src_text
    target = encoder.tp.mark(hyp_text) if tgt_lang == "tok" else hyp_text
    msgs = encoder.messages(prompt.replace("{text}", payload), target)
    rendered = tok.apply_chat_template(msgs, tokenize=False)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = torch.tensor([enc["input_ids"]], device=wrapper.model.device)

    src_base = rendered.index(payload, rendered.index(prompt.split("{text}")[0]))
    tgt_base = rendered.rindex(target)
    src_words, tgt_words = char_words(payload), char_words(target)
    src_idx = token_word_index(enc["offset_mapping"], src_words, src_base)
    tgt_idx = token_word_index(enc["offset_mapping"], tgt_words, tgt_base)
    # a token is source only if it lies inside the payload span (not the template)
    in_src = [src_base <= s < src_base + len(payload) and k >= 0 for (s, _), k in zip(enc["offset_mapping"], src_idx)]
    source_mask = torch.tensor([in_src], dtype=torch.float32, device=ids.device)
    _, align = wrapper(ids, torch.ones_like(ids), source_mask)
    align = align[0].float().cpu().numpy()

    A = np.zeros((len(tgt_words), len(src_words)))
    counts = np.zeros(len(tgt_words))
    for t, tw in enumerate(tgt_idx):
        # token t is predicted from position t-1
        if tw < 0 or t == 0 or not (tgt_base <= enc["offset_mapping"][t][0] < tgt_base + len(target)):
            continue
        for s, sw in enumerate(src_idx):
            if in_src[s]:
                A[tw, sw] += align[t - 1, s]
        counts[tw] += 1
    A[counts > 0] /= counts[counts > 0, None]
    return [w for _, _, w in src_words], [w for _, _, w in tgt_words], A


def silver_links(src_words, tgt_words, gloss):
    """Target index -> unique source index according to the dictionary."""
    words = [w[4:-1] if w.startswith("<tp:") else None for w in src_words]
    links = {}
    for j, tw in enumerate(tgt_words):
        tw = tw.lower()
        cands = {i for i, w in enumerate(words) if w and (tw in gloss.get(w, ()) or tw.rstrip("s") in gloss.get(w, ()))}
        # same toki pona word twice in the source is ambiguous
        if len(cands) == 1 and words.count(words[next(iter(cands))]) == 1:
            links[j] = next(iter(cands))
    return links


def load_gloss():
    gloss = {}
    for d in mt_data.load_jsonl("dictionary.jsonl"):
        words = set()
        for g in list(d["ku_en"]) + re.split(r"[;,()]", d["en"]):
            words |= {w.lower() for w in re.findall(r"[A-Za-z]+", g) if len(w) > 2}
        gloss[d["word"]] = words - {"the", "and", "for", "with", "something", "thing", "noun", "verb"}
    return gloss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--bands", default="0.0-0.2,0.2-0.4,0.4-0.6,0.6-0.8,0.8-1.0,0.7-0.7")  # 0.7-0.7 = the guided layer alone
    args = ap.parse_args()

    model, encoder = load_run(args.run_dir)
    model.config._attn_implementation = "eager"  # attention weights are needed
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    srcs, _ = test_pairs("tok", "en")
    srcs = srcs[:args.limit]
    hyps = translate(model, encoder, "tok", "en", srcs)
    gloss = load_gloss()

    report = {}
    for band in args.bands.split(","):
        lo, hi = map(float, band.split("-"))
        wrapper = AttentiveTranslatorWrapper(model, lo, hi)
        hits = total = diag = first = 0
        for src, hyp in zip(srcs, hyps):
            sw, tw, A = align_pair(wrapper, encoder, "tok", "en", src, hyp)
            for j, i in silver_links(sw, tw, gloss).items():
                total += 1
                hits += int(A[j].argmax() == i)
                diag += int(round(j * (len(sw) - 1) / max(1, len(tw) - 1)) == i)
                first += int(i == 0)
        report[band] = {"layers": wrapper.layers, "top1": hits / max(total, 1), "n_links": total,
                        "diagonal_baseline": diag / max(total, 1), "first_word_baseline": first / max(total, 1)}
        print(band, report[band], flush=True)
    with open(os.path.join(args.run_dir, "linker.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
