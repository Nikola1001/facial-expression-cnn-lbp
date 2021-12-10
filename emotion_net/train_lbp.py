"""Train an LBP-only classifier for controlled late fusion experiments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import CLASSES
from .data import read_fer
from .experiment import set_seed
from .features import lbp_histogram
from .metrics import classification_metrics
from .model import LBPExpert


def main():
    parser = argparse.ArgumentParser(description="Train a spatial LBP histogram classifier")
    parser.add_argument("--fer-csv", type=Path, default=Path("data/raw/fer2013.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lbp_expert"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    args.output.mkdir(parents=True, exist_ok=True)
    splits = read_fer(args.fer_csv)
    features = {name: torch.from_numpy(np.stack([lbp_histogram(image) for image in splits[name][0]]))
                for name in ("train", "val")}
    targets = {name: torch.from_numpy(splits[name][1]) for name in ("train", "val")}
    train = DataLoader(TensorDataset(features["train"], targets["train"]),
                       batch_size=512, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LBPExpert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    history, best_score = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(None, x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_logits = model(None, features["train"].to(device)).cpu()
            val_logits = model(None, features["val"].to(device)).cpu()
        row = {"epoch": epoch}
        for name, logits in (("train", train_logits), ("val", val_logits)):
            row[name] = classification_metrics(targets[name].numpy(), logits.argmax(1).numpy())
            row[name]["loss"] = float(nn.functional.cross_entropy(logits, targets[name]))
        history.append(row)
        score = row["val"]["accuracy"]
        print(f"epoch {epoch}/{args.epochs}: val accuracy={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "architecture": "lbp_expert", "epoch": epoch, "seed": args.seed},
                       args.output / "best.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("accuracy", "loss")):
        for split in ("train", "val"):
            ax.plot([row[split][key] for row in history], label=split)
        ax.set(title=key, xlabel="Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"selected_epoch": torch.load(args.output / "best.pt", weights_only=True)["epoch"],
                      "val_accuracy": best_score}), flush=True)


if __name__ == "__main__":
    main()
