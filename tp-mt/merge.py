"""Merges a train_mt.py run (LoRA + <tp:*> rows) into a standalone HF checkpoint.

    uv run python merge.py runs/qwen2.5-0.5b        # -> runs/qwen2.5-0.5b/merged
"""

import argparse
import json
import os

from eval_mt import load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    model, encoder = load_run(args.run_dir, device="cpu")
    out = os.path.join(args.run_dir, "merged")
    model.save_pretrained(out)
    encoder.tok.save_pretrained(out)
    tp = encoder.tp
    # linker layers for the export: the best band measured by linker.py, if it has been run
    align_layers = None
    if os.path.exists(os.path.join(args.run_dir, "linker.json")):
        with open(os.path.join(args.run_dir, "linker.json")) as f:
            align_layers = max(json.load(f).values(), key=lambda b: b["top1"])["layers"]
    with open(os.path.join(out, "tp_vocab.json"), "w") as f:
        # glyph_id -> token id, so the iOS app can feed CNN output straight into the LLM
        json.dump({"tp_offset": tp.tp_offset, "words": tp.words, "cartouche_open": tp.cart_open_id,
                   "cartouche_close": tp.cart_close_id, "align_layers": align_layers}, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
