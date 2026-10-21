"""Converts a merged tp-mt checkpoint into a stateful, int4-palettized CoreML model.

Inputs : inputIds (1, q) int32, causalMask (1, 1, q, end) fp16, sourceMask (1, end) fp16
Outputs: logits (1, 1, V), alignment (1, end)  -- the attention-linker vector per step
States : keyCache / valueCache (KV cache lives on device between calls)

    uv run python export_llm.py ../tp-mt/runs/qwen2.5-0.5b/merged --context 512
"""

import argparse
import glob
import json
import os
import shutil

import coremltools as ct
import numpy as np
import torch

from llm_module import Qwen2Export

HERE = os.path.dirname(os.path.abspath(__file__))


def dir_size(path):
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(path, "**"), recursive=True) if os.path.isfile(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_dir")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "models", "TokiPonaTranslator.mlpackage"))
    args = ap.parse_args()

    model = Qwen2Export.from_pretrained(args.merged_dir, max_context=args.context, dtype=torch.float16)
    # tracing with a 2-token query and 5 cached positions; real shapes are flexible below
    ids = torch.zeros((1, 2), dtype=torch.int32)
    mask = torch.zeros((1, 1, 2, 5), dtype=torch.float16)
    src = torch.ones((1, 5), dtype=torch.float16)
    traced = torch.jit.trace(model, (ids, mask, src))

    q = ct.RangeDim(lower_bound=1, upper_bound=args.context, default=1)
    end = ct.RangeDim(lower_bound=1, upper_bound=args.context, default=1)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="inputIds", shape=(1, q), dtype=np.int32),
                ct.TensorType(name="causalMask", shape=(1, 1, q, end), dtype=np.float16),
                ct.TensorType(name="sourceMask", shape=(1, end), dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16),
                 ct.TensorType(name="alignment", dtype=np.float16)],
        states=[ct.StateType(wrapped_type=ct.TensorType(shape=model.k_cache.shape, dtype=np.float16), name="k_cache"),
                ct.StateType(wrapped_type=ct.TensorType(shape=model.v_cache.shape, dtype=np.float16), name="v_cache")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    if args.nbits:
        cfg = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpPalettizerConfig(
            nbits=args.nbits, mode="kmeans", granularity="per_grouped_channel", group_size=16))
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, cfg)

    with open(os.path.join(args.merged_dir, "tp_vocab.json")) as f:
        vocab = json.load(f)
    mlmodel.short_description = "toki pona <-> English/Chinese translator with attention-linker output"
    mlmodel.user_defined_metadata["tp_offset"] = str(vocab["tp_offset"])
    mlmodel.user_defined_metadata["align_layers"] = json.dumps(sorted(model.align_layers))
    mlmodel.user_defined_metadata["context"] = str(args.context)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    mlmodel.save(args.out)
    tok_dir = args.out.removesuffix(".mlpackage") + "-tokenizer"  # tp_vocab.json differs per model
    os.makedirs(tok_dir, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "tp_vocab.json", "chat_template.jinja"):
        if os.path.exists(os.path.join(args.merged_dir, name)):
            shutil.copy(os.path.join(args.merged_dir, name), tok_dir)
    print(f"Exported {args.out} ({dir_size(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
