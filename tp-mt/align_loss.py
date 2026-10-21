"""Guided alignment loss: KL(eflomal links || attention of one layer over source words).

The supervised layer keeps SDPA for its real forward; a pre-hook stashes its inputs and the
attention probabilities are recomputed only for supervised query rows, so memory stays small.
"""

import json
import os
import re

import torch
import torch.nn.functional as F

from linker import WORD_RE, char_words, token_word_index

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = re.compile(r"[A-Za-z0-9']+")


def _key(tok, en):
    return re.sub(r"\s+", " ", tok.strip()) + "\t" + re.sub(r"\s+", " ", en.strip())


def load_links(path=os.path.join(HERE, "cache", "align_tok_en.jsonl")):
    links = {}
    with open(path) as f:
        for row in map(json.loads, f):
            if row["split"] == "train":
                links[_key(row["tok"], row["en"])] = row["links"]
    return links


def example_alignment(encoder, ex, links, ids):
    """{"rows": [(query_row, [linked source positions]), ...], "src": [all source positions]} or None."""
    if ex["task"] == "tok-en":
        tok_text, en_text, tok_is_src = ex["text"], ex["assistant"], True
    elif ex["task"] == "en-tok":
        tok_text, en_text, tok_is_src = ex["assistant"], ex["text"], False
    else:
        return []
    pairs = links.get(_key(tok_text, en_text))
    if not pairs:
        return None
    tok = encoder.tok
    user = encoder.user_text(ex)
    target = encoder.tp.mark(ex["assistant"]) if ex["tp_assistant"] else ex["assistant"]
    rendered = tok.apply_chat_template(encoder.messages(user, target), tokenize=False)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"][:len(ids)] != ids[:len(enc["input_ids"])]:
        return None
    payload = encoder.tp.mark(ex["text"]) if ex["tp_user"] else ex["text"]
    src_base = rendered.index(payload, rendered.index(ex["prompt"].split("{text}")[0]))
    tgt_base = rendered.rindex(target)
    src_words, tgt_words = char_words(payload), char_words(target)
    # eflomal indexed plain words; the marked spans must line up one-to-one
    n_tok = len(WORD.findall(tok_text))
    n_en = len(WORD.findall(en_text))
    if (len(src_words), len(tgt_words)) != ((n_tok, n_en) if tok_is_src else (n_en, n_tok)):
        return None
    offs = enc["offset_mapping"]
    src_idx = token_word_index(offs, src_words, src_base)
    tgt_idx = token_word_index(offs, tgt_words, tgt_base)
    src_tokens = {}
    for p, (k, (s, _)) in enumerate(zip(src_idx, offs)):
        if k >= 0 and src_base <= s < src_base + len(payload):
            src_tokens.setdefault(k, []).append(p)
    by_target = {}
    for i, j in pairs:  # (tok word, en word)
        s, t = (i, j) if tok_is_src else (j, i)
        by_target.setdefault(t, set()).add(s)
    out = []
    for p, (k, (s, _)) in enumerate(zip(tgt_idx, offs)):
        if p == 0 or k < 0 or not (tgt_base <= s < tgt_base + len(target)) or k not in by_target:
            continue
        positions = sorted(q for w in by_target[k] for q in src_tokens.get(w, []))
        if positions:
            out.append((p - 1, positions))  # row p-1 predicts token p
    if not out:
        return None
    return {"rows": out, "src": sorted(q for ps in src_tokens.values() for q in ps)}


class AttentionProbe:
    """Recomputes one decoder layer's attention rows for the alignment loss."""

    def __init__(self, model, layer_frac=0.7):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        layers = base.model.layers
        self.layer_index = int(layer_frac * len(layers))
        self.attn = layers[self.layer_index].self_attn
        self.stash = None
        self.handle = self.attn.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        self.stash = (hidden, kwargs.get("position_embeddings"))

    def loss(self, supervision):
        """supervision: per batch row, None or example_alignment() output."""
        hidden, (cos, sin) = self.stash
        a = self.attn
        B, T, _ = hidden.shape
        hd = a.head_dim
        q = a.q_proj(hidden).view(B, T, -1, hd).transpose(1, 2)
        k = a.k_proj(hidden).view(B, T, -1, hd).transpose(1, 2)
        cos_, sin_ = cos.unsqueeze(1), sin.unsqueeze(1)
        rot = lambda x: torch.cat((-x[..., hd // 2:], x[..., :hd // 2]), dim=-1)  # noqa: E731
        q = q * cos_ + rot(q) * sin_
        k = k * cos_ + rot(k) * sin_
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        losses = []
        for b, item in enumerate(supervision):
            if not item:
                continue
            sup = item["rows"]
            rows = torch.tensor([r for r, _ in sup], device=hidden.device)
            scores = (q[b, :, rows] @ k[b].transpose(-1, -2)).float() / hd ** 0.5  # (H, n, T)
            causal = torch.arange(T, device=hidden.device)[None] > rows[:, None]
            scores = scores.masked_fill(causal[None], float("-inf"))
            probs = F.softmax(scores, dim=-1).mean(0)  # (n, T), heads averaged
            src = torch.zeros(T, device=hidden.device)
            src[item["src"]] = 1.0
            m = probs * src[None]
            m = m / m.sum(-1, keepdim=True).clamp_min(1e-9)
            target = torch.zeros_like(m)
            for n, (_, positions) in enumerate(sup):
                target[n, positions] = 1.0 / len(positions)
            losses.append((target * (torch.log(target.clamp_min(1e-9)) - torch.log(m.clamp_min(1e-9)))).sum(-1).mean())
        self.stash = None
        return torch.stack(losses).mean() if losses else None
