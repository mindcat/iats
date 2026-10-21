"""Evaluates a train_mt.py run on the held-out gold test split.

Directions: tok->en, en->tok (gold English), tok->zh, zh->tok (gold Chinese).
Writes hypotheses and BLEU / chrF++ to <run>/eval.json.

    uv run python eval_mt.py runs/qwen2.5-0.5b
"""

import argparse
import json
import os

import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import mt_data
from grammar import grammatical_rate

DIRECTIONS = [("tok", "en"), ("en", "tok"), ("tok", "zh"), ("zh", "tok")]


def load_run(run_dir, device="cuda"):
    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    tok = AutoTokenizer.from_pretrained(run["args"]["model"])
    encoder = mt_data.ChatEncoder(tok)
    model = AutoModelForCausalLM.from_pretrained(run["args"]["model"], dtype=torch.bfloat16).to(device)
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok), mean_resizing=False)
    rows = torch.load(os.path.join(run_dir, "tp_rows.pt"))
    with torch.no_grad():
        model.get_input_embeddings().weight[rows["ids"]] = rows["embed"].to(device)
        if "lm_head" in rows:
            model.get_output_embeddings().weight[rows["ids"]] = rows["lm_head"].to(device)
    adapter = os.path.join(run_dir, "adapter")
    if os.path.exists(adapter):
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, encoder


@torch.no_grad()
def translate(model, encoder, src, tgt, texts, batch_size=48, max_new_tokens=192):
    tok = encoder.tok
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prompt = mt_data.translate_prompt(src, tgt)
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    results = {}
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        prompts = [encoder.prompt_ids(encoder.user_text(
            {"prompt": prompt, "text": texts[i], "tp_user": src == "tok"})) for i in idx]
        n = max(map(len, prompts))
        ids = torch.tensor([[pad] * (n - len(p)) + p for p in prompts], device=model.device)
        attn = (torch.arange(n, device=model.device)[None] >= torch.tensor(
            [n - len(p) for p in prompts], device=model.device)[:, None]).long()
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=pad)
        for i, row in zip(idx, gen[:, n:].tolist()):
            stop = [k for k, t in enumerate(row) if t in encoder.stop_ids]
            row = row[:stop[0]] if stop else row
            results[i] = encoder.tp.decode(row) if tgt == "tok" else tok.decode(row, skip_special_tokens=True).strip()
    return [results[i] for i in range(len(texts))]


def score(hyps, refs, tgt):
    tokenize = "zh" if tgt == "zh" else "13a"
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    out = {"bleu": round(bleu.score, 2), "chrf++": round(chrf.score, 2), "n": len(refs)}
    if tgt == "tok":
        out["grammatical_rate"] = round(grammatical_rate(hyps), 4)
        out["reference_grammatical_rate"] = round(grammatical_rate(refs), 4)
    return out


def test_pairs(src, tgt):
    rows = [r for r in mt_data.load_jsonl("parallel.jsonl") if r["split"] == "test"]
    lang = "zh" if "zh" in (src, tgt) else "en"
    rows = [r for r in rows if r[f"{lang}_quality"] == "gold"]
    return [r[src] for r in rows], [r[tgt] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, encoder = load_run(args.run_dir)
    report, samples = {}, {}
    for src, tgt in DIRECTIONS:
        srcs, refs = test_pairs(src, tgt)
        if args.limit:
            srcs, refs = srcs[:args.limit], refs[:args.limit]
        hyps = translate(model, encoder, src, tgt, srcs)
        key = f"{src}-{tgt}"
        report[key] = score(hyps, refs, tgt)
        samples[key] = [{"src": s, "ref": r, "hyp": h} for s, r, h in zip(srcs, refs, hyps)]
        print(key, report[key], flush=True)
    with open(os.path.join(args.run_dir, "eval.json"), "w") as f:
        json.dump({"metrics": report, "samples": samples}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
