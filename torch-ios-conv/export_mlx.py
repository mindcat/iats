"""4-bit MLX folder from a merged tp-mt checkpoint (mlx-swift / mlx-lm speed benchmarking).

Unlike export_llm.py, stock mlx-swift runners do not expose attention weights; the
attention linker needs a patched TransformerBlock that returns scores before SDPA.

    uv run --no-project --python 3.12 --with "mlx[cpu]" --with mlx-lm \
        python export_mlx.py ../tp-mt/runs/qwen2.5-0.5b/merged
"""

import argparse
import glob
import os
import shutil

from mlx_lm import convert

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged_dir")
    ap.add_argument("--out", default=os.path.join(HERE, "models", "TokiPonaTranslator-mlx-4bit"))
    ap.add_argument("--q-bits", type=int, default=4)
    args = ap.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    convert(args.merged_dir, mlx_path=args.out, quantize=True, q_bits=args.q_bits, q_group_size=64)
    shutil.copy(os.path.join(args.merged_dir, "tp_vocab.json"), args.out)
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(args.out, "*")))
    print(f"Exported {args.out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
