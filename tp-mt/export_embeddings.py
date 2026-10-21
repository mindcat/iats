"""Dumps <tp:WORD> embedding rows as {words, vectors} .npz for tp-dataset/embeddings/evaluate.py.

    uv run python export_embeddings.py --run runs/qwen2.5-0.5b             # trained rows
    uv run python export_embeddings.py --warm-start Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mt_data
from init_embeddings import warm_start

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tp-dataset", "embeddings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--warm-start")
    ap.add_argument("--name")
    args = ap.parse_args()

    if args.run:
        with open(os.path.join(args.run, "run.json")) as f:
            run = json.load(f)
        tok = AutoTokenizer.from_pretrained(run["args"]["model"])
        tp = mt_data.ChatEncoder(tok).tp
        rows = torch.load(os.path.join(args.run, "tp_rows.pt"))
        pos = {i: k for k, i in enumerate(rows["ids"])}
        vecs = np.stack([rows["embed"][pos[i]].float().numpy() for i in tp.tp_ids])
        name = args.name or f"llm_{os.path.basename(os.path.normpath(args.run))}"
    else:
        tok = AutoTokenizer.from_pretrained(args.warm_start)
        tp = mt_data.ChatEncoder(tok).tp
        model = AutoModelForCausalLM.from_pretrained(args.warm_start, dtype=torch.float32)
        if len(tok) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok), mean_resizing=False)
        warm_start(model, tp)
        vecs = model.get_input_embeddings().weight[tp.tp_ids].detach().numpy()
        name = args.name or f"warmstart_{args.warm_start.split('/')[-1]}"
    path = os.path.join(OUT, f"{name}.npz")
    np.savez(path, words=np.array(tp.words), vectors=vecs)
    print(f"saved {path} {vecs.shape}")


if __name__ == "__main__":
    main()
