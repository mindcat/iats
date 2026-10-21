"""Trains SitelenPonaCNN on data/{train,val}_data.npz produced by generate_dataset.py.

    uv run python train.py --epochs 15
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


class SitelenPonaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class SitelenPonaCNNv2(nn.Module):
    """Three conv blocks with BatchNorm; fewer parameters than v1 but a much better reject class."""

    def __init__(self, num_classes):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        # 28x28 => 14x14 => 7x7 => 3x3
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.flatten(self.features(x), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


ARCHS = {"v1": SitelenPonaCNN, "v2": SitelenPonaCNNv2}


def load_split(path, device):
    data = np.load(path)
    X = torch.from_numpy(data["X"]).to(device)  # uint8, normalized per batch
    y = torch.from_numpy(data["y"]).long().to(device)
    return X, y


def batches(X, y, batch_size, shuffle):
    idx = torch.randperm(len(X), device=X.device) if shuffle else torch.arange(len(X), device=X.device)
    for i in range(0, len(X), batch_size):
        j = idx[i:i + batch_size]
        yield X[j].unsqueeze(1).float() / 255.0, y[j]


@torch.no_grad()
def evaluate(model, X, y, n_classes, batch_size=4096):
    model.eval()
    preds, loss = [], 0.0
    for xb, yb in batches(X, y, batch_size, shuffle=False):
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        preds.append(out.argmax(1))
    preds = torch.cat(preds)
    conf = torch.bincount(y * n_classes + preds, minlength=n_classes ** 2).reshape(n_classes, n_classes).cpu().numpy()
    return loss / len(X), conf


def summarize(conf, labels):
    neg = len(labels) - 1
    recall = conf.diagonal() / conf.sum(1).clip(min=1)
    acc = conf.diagonal().sum() / conf.sum()
    # reject class: negatives accepted as a glyph / glyphs thrown away as reject
    neg_fpr = 1 - recall[neg]
    false_reject = conf[:neg, neg].sum() / conf[:neg].sum()
    return {
        "val_accuracy": float(acc),
        "reject_false_positive_rate": float(neg_fpr),
        "glyph_false_reject_rate": float(false_reject),
        "worst_recall": sorted(((labels[i], float(r)) for i, r in enumerate(recall)), key=lambda t: t[1])[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="v2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    with open(os.path.join(args.data_dir, "class_mapping.json")) as f:
        mapping = json.load(f)
    labels = [mapping[str(i)]["word"] for i in range(len(mapping))]
    n = len(labels)

    X_train, y_train = load_split(os.path.join(args.data_dir, "train_data.npz"), device)
    X_val, y_val = load_split(os.path.join(args.data_dir, "val_data.npz"), device)
    print(f"train {tuple(X_train.shape)}  val {tuple(X_val.shape)}  classes {n}")

    model = ARCHS[args.arch](n).to(device)
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters())} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * ((len(X_train) + args.batch_size - 1) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    for epoch in range(args.epochs):
        model.train()
        running, nb = 0.0, 0
        for xb, yb in batches(X_train, y_train, args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            nb += 1
        val_loss, conf = evaluate(model, X_val, y_val, n)
        s = summarize(conf, labels)
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | Train Loss: {running / nb:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {100 * s['val_accuracy']:.2f}% | reject FPR {100 * s['reject_false_positive_rate']:.2f}% | "
              f"false reject {100 * s['glyph_false_reject_rate']:.2f}%", flush=True)

    print("lowest per-class recall:", ", ".join(f"{w} {100 * r:.1f}%" for w, r in s["worst_recall"]))
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    stem = f"{100 * s['val_accuracy']:.2f}%-{args.arch}-{n}c-{datetime.now().strftime('%m-%d-%H:%M')}"
    torch.save(model.state_dict(), os.path.join(HERE, "models", f"{stem}.pth"))
    with open(os.path.join(HERE, "models", f"{stem}.json"), "w") as f:
        json.dump({"arch": args.arch, "class_mapping": mapping, "metrics": s, "val_loss": val_loss, "args": vars(args),
                   "confusion": conf.tolist()}, f, ensure_ascii=False)
    print(f"Model saved to models/{stem}.pth")


if __name__ == "__main__":
    main()
