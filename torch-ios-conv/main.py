"""Exports an ocr-demo SitelenPonaCNN checkpoint to SitelenPonaClassifier.mlpackage.

    uv run python main.py                       # newest checkpoint in ../ocr-demo/models
    uv run python main.py --model "98.90%-135c-10-20-09:00"
"""

import argparse
import glob
import json
import os
import shutil
import sys

import coremltools as ct
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OCR_DEMO = os.path.join(HERE, "..", "ocr-demo")
OCR_MODELS = os.path.join(OCR_DEMO, "models")
sys.path.insert(0, OCR_DEMO)

from train import ARCHS  # noqa: E402  (dropout is identity in eval mode, so tracing is export-safe)


def dir_size(path):
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(path, "**"), recursive=True) if os.path.isfile(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="checkpoint stem in ocr-demo/models (default: newest)")
    ap.add_argument("--out", default=os.path.join(HERE, "models", "SitelenPonaClassifier.mlpackage"))
    args = ap.parse_args()

    if args.model:
        ckpt = os.path.join(OCR_MODELS, f"{args.model}.pth")
    else:
        ckpt = max(glob.glob(os.path.join(OCR_MODELS, "*.pth")), key=os.path.getmtime)
    meta_path = ckpt.removesuffix(".pth") + ".json"
    with open(meta_path) as f:
        meta = json.load(f)
    mapping = meta["class_mapping"]
    labels = [mapping[str(i)]["word"] for i in range(len(mapping))]
    print(f"{os.path.basename(ckpt)}: {len(labels)} classes")

    model = ARCHS[meta.get("arch", "v1")](len(labels))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    traced = torch.jit.trace(model, torch.rand(1, 1, 28, 28))

    # iOS feeds 0-255 pixels; the model was trained on 0.0-1.0
    image_input = ct.ImageType(name="image", shape=(1, 1, 28, 28),
                               color_layout=ct.colorlayout.GRAYSCALE, scale=1 / 255.0)
    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels=labels),
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.short_description = "sitelen pona glyph classifier (28x28 white-on-black); last class is not_sitelen_pona"
    mlmodel.user_defined_metadata["source_checkpoint"] = os.path.basename(ckpt)
    mlmodel.user_defined_metadata["val_accuracy"] = f"{meta['metrics']['val_accuracy']:.4f}"
    mlmodel.user_defined_metadata["reject_label"] = labels[-1]
    mlmodel.user_defined_metadata["arch"] = meta.get("arch", "v1")
    mlmodel.user_defined_metadata["class_mapping"] = json.dumps(
        {i: {k: mapping[i][k] for k in ("word", "ucsur_hex")} for i in mapping}, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    mlmodel.save(args.out)
    with open(os.path.join(os.path.dirname(args.out), "class_mapping.json"), "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)
    print(f"Exported {args.out} ({dir_size(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
