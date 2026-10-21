"""Prior-work baselines on the same gold test split and metrics as eval_mt.py:

  m2m100      m2m/translate.ipynb: Jayyydyyy/m2m100_418m_tokipona ('tl' = toki pona)
  gpt2-alpaca tp-gpt2 (McDaniel 2023): adamthekiwi/toki-pona-gpt2-alpaca-best, its Alpaca prompt

Caveat: both were trained on public Tatoeba data, so some test pairs may have been seen
in their training; their scores are an upper bound.

    uv run python eval_baselines.py [m2m100|gpt2-alpaca]
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from eval_mt import score, test_pairs

HERE = os.path.dirname(os.path.abspath(__file__))


ALPACA = ("Below is an instruction that describes a task. Write a response that appropriately completes "
          "the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n")
ALPACA_TASK = {("en", "tok"): "Translate the following English text into Toki Pona:\n{text}",
               ("tok", "en"): "Translate the following Toki Pona text into English:\n{text}"}


def clean_gpt2(text):
    # the model never learned to stop: it runs straight into the next Alpaca prompt
    return text.split("Below is an instruction")[0].split("###")[0].split("\n\n")[0].strip()


@torch.no_grad()
def translate_gpt2(model, tok, texts, src, tgt, batch_size=64):
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    out = []
    for i in range(0, len(texts), batch_size):
        prompts = [ALPACA.format(instruction=ALPACA_TASK[(src, tgt)].format(text=t)) for t in texts[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        for row in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            out.append(clean_gpt2(row))
    return out


@torch.no_grad()
def translate(model, tok, texts, src, tgt, batch_size=64):
    tok.src_lang = src
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(model.device)
        gen = model.generate(**enc, forced_bos_token_id=tok.get_lang_id(tgt), max_new_tokens=192, num_beams=1)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return [o.strip() for o in out]


def run(name, translate_fn):
    report, samples = {}, {}
    for src, tgt in (("tok", "en"), ("en", "tok")):
        srcs, refs = test_pairs(src, tgt)
        hyps = translate_fn(srcs, src, tgt)
        report[f"{src}-{tgt}"] = score(hyps, refs, tgt)
        samples[f"{src}-{tgt}"] = [{"src": s, "ref": r, "hyp": h} for s, r, h in zip(srcs, refs, hyps)]
        print(name, f"{src}-{tgt}", report[f"{src}-{tgt}"], flush=True)
    out = os.path.join(HERE, "runs", f"baseline-{name}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval.json"), "w") as f:
        json.dump({"metrics": report, "samples": samples}, f, ensure_ascii=False, indent=1)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    code = {"tok": "tl", "en": "en"}
    if only in (None, "m2m100"):
        tok = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
        model = AutoModelForSeq2SeqLM.from_pretrained("Jayyydyyy/m2m100_418m_tokipona").cuda().eval()
        run("m2m100", lambda texts, s, t: translate(model, tok, texts, code[s], code[t]))
        del model
    if only not in (None, "gpt2-alpaca"):
        return

    # the checkpoint ships no tokenizer files; it was finetuned from vicgalle/gpt2-alpaca-gpt4
    tok = AutoTokenizer.from_pretrained("vicgalle/gpt2-alpaca-gpt4")
    model = AutoModelForCausalLM.from_pretrained("adamthekiwi/toki-pona-gpt2-alpaca-best").cuda().eval()
    run("gpt2-alpaca", lambda texts, s, t: translate_gpt2(model, tok, texts, s, t))


if __name__ == "__main__":
    main()
