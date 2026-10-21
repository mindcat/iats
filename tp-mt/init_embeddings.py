"""Semantic warm-start for <tp:WORD> rows (no random vectors).

Each new row = weighted mean of the base model's embeddings of the word's lipu Linku
English glosses (ku_data weights) and its Chinese definition. If a standalone
toki pona word2vec space exists (tp-dataset/embeddings), a ridge map from that space
into the LLM space is fitted on the dictionary vectors and blended in, so words with
thin dictionary entries borrow geometry from corpus co-occurrence.
"""

import json
import os
import re

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TP_DATASET = os.path.join(HERE, "..", "tp-dataset")
W2V_PATH = os.path.join(TP_DATASET, "embeddings", "hw_full_d24.npz")  # Huang & Wu full model (7.95M tokens)


def load_dictionary():
    with open(os.path.join(TP_DATASET, "corpus", "dictionary.jsonl")) as f:
        return {d["word"]: d for d in map(json.loads, f)}


def _phrase_vec(matrix, tokenizer, phrase):
    ids = tokenizer(" " + phrase.strip(), add_special_tokens=False)["input_ids"]
    ids = [i for i in ids if i < matrix.shape[0]]
    return matrix[ids].float().mean(0) if ids else None


def _clean_definition(text):
    text = re.sub(r"\([^)]*\)|（[^）]*）", " ", text)
    return [p.strip() for p in re.split(r"[;,，、；]", text) if p.strip()]


def dictionary_vectors(matrix, tokenizer, words, zh_weight=0.3):
    dictionary = load_dictionary()
    vecs = []
    for w in words:
        d = dictionary.get(w, {})
        parts, weights = [], []
        ku = d.get("ku_en") or {}
        glosses = list(ku.items()) or [(g, 50) for g in _clean_definition(d.get("en", ""))]
        en = [(v, wt) for g, wt in glosses if (v := _phrase_vec(matrix, tokenizer, g)) is not None]
        if en:
            total = sum(wt for _, wt in en)
            parts.append(sum(v * (wt / total) for v, wt in en))
            weights.append(1 - zh_weight)
        zh = [v for g in _clean_definition(d.get("zh", "")) if (v := _phrase_vec(matrix, tokenizer, g)) is not None]
        if zh:
            parts.append(torch.stack(zh).mean(0))
            weights.append(zh_weight)
        if not parts:
            # last resort: the latin spelling itself
            parts, weights = [_phrase_vec(matrix, tokenizer, w)], [1.0]
        vecs.append(sum(p * (wt / sum(weights)) for p, wt in zip(parts, weights)))
    return torch.stack(vecs)


def ridge_refine(dict_vecs, words, alpha=0.5, ridge=1.0):
    """Blend dictionary vectors with a ridge projection of the word2vec space into the LLM space."""
    if not os.path.exists(W2V_PATH):
        return dict_vecs, False
    w2v = np.load(W2V_PATH)
    vocab = {w: i for i, w in enumerate(w2v["words"])}
    have = [k for k, w in enumerate(words) if w in vocab]
    if len(have) < 20:
        return dict_vecs, False
    X = np.stack([w2v["vectors"][vocab[words[k]]] for k in have])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Y = dict_vecs[have].cpu().numpy()
    mu = Y.mean(0)
    W = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ (Y - mu))
    out = dict_vecs.clone()
    out[have] = torch.from_numpy(alpha * Y + (1 - alpha) * (X @ W + mu)).float().to(out.device)
    return out, True


@torch.no_grad()
def warm_start(model, tp_tokenizer, extra_ids=()):
    """Writes warm-start vectors into input (and untied output) embedding rows."""
    base_tok = tp_tokenizer.base
    words = tp_tokenizer.words
    ids = tp_tokenizer.tp_ids
    emb = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings().weight if model.get_output_embeddings() is not None else emb

    # only the original vocabulary may serve as source rows
    src = emb[:tp_tokenizer.tp_offset]
    vecs = dictionary_vectors(src, base_tok, words)
    vecs, refined = ridge_refine(vecs, words)
    # match the typical norm of real vocabulary rows
    target_norm = src.float().norm(dim=1).median()
    vecs = vecs / vecs.norm(dim=1, keepdim=True) * target_norm
    emb[ids] = vecs.to(emb.dtype)
    if out_emb.data_ptr() != emb.data_ptr():
        out_src = out_emb[:tp_tokenizer.tp_offset]
        ovecs = dictionary_vectors(out_src, base_tok, words)
        ovecs = ovecs / ovecs.norm(dim=1, keepdim=True) * out_src.float().norm(dim=1).median()
        out_emb[ids] = ovecs.to(out_emb.dtype)

    # cartouche markers: start from the embedding of a quote / name-ish token
    for marker_id, phrase in extra_ids:
        v = _phrase_vec(src, base_tok, phrase)
        emb[marker_id] = (v / v.norm() * target_norm).to(emb.dtype)
        if out_emb.data_ptr() != emb.data_ptr():
            out_emb[marker_id] = _phrase_vec(out_emb[:tp_tokenizer.tp_offset], base_tok, phrase).to(out_emb.dtype)
    return refined
