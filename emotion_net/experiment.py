"""Training and evaluation entry point."""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import CLASSES
from .data import FaceDataset, read_ck, read_fer, read_fer_archive, split_summary
from .metrics import classification_metrics
from .model import EmotionNet, ResNet18LBP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(items, batch_size, training, strong_augment=False):
    return DataLoader(FaceDataset(items[0], items[1], augment=training,
                                  strong_augment=strong_augment),
                      batch_size=batch_size, shuffle=training, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              amp_dtype=torch.float16):
    training = optimizer is not None
    model.train(training)
    total_loss, actual, predicted = 0.0, [], []
    for image, lbp, target in loader:
        image, lbp, target = image.to(device), lbp.to(device), target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                logits = model(image, lbp)
                loss = criterion(logits, target)
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        total_loss += loss.item() * len(target)
        actual.extend(target.cpu().tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
    metrics = classification_metrics(actual, predicted)
    metrics["loss"] = total_loss / len(actual)
    return metrics


def save_plots(history, matrix, out_dir):
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key, ax, title in (("accuracy", axes[0], "Accuracy"),
                           ("loss", axes[1], "Cross-entropy loss"),
                           ("macro_f1", axes[2], "Macro F1")):
        ax.plot(epochs, [row["train"][key] for row in history], label="train")
        ax.plot(epochs, [row["val"][key] for row in history], label="validation")
        ax.set(xlabel="Epoch", title=title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    save_confusion(matrix, out_dir / "confusion_matrix.png")


def save_confusion(matrix, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    counts = np.asarray(matrix)
    ax.imshow(counts, cmap="Blues")
    ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(7):
        for col in range(7):
            color = "white" if counts[row, col] > counts.max() * 0.5 else "black"
            ax.text(col, row, str(counts[row, col]), ha="center", va="center",
                    fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train FER2013 or subject-held-out CK+ experiment")
    parser.add_argument("--dataset", choices=("fer", "ck"), required=True)
    parser.add_argument("--fer-csv", type=Path, default=Path("data/raw/fer2013.csv"))
    parser.add_argument("--fer-archive", type=Path,
                        help="FER2013 image ZIP with train/test folders and split prefixes")
    parser.add_argument("--ck-root", type=Path, default=Path("data/raw/ckplus"))
    parser.add_argument("--init", type=Path, help="Optional pretrained model checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weights", choices=("sqrt_inverse", "none"),
                        default="sqrt_inverse")
    parser.add_argument("--selection", choices=("macro_f1", "accuracy"),
                        default="macro_f1")
    parser.add_argument("--architecture", choices=("compact", "resnet18_lbp"),
                        default="compact")
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--disable-lbp", action="store_true",
                        help="Train a control run without LBP information")
    parser.add_argument("--strong-augment", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 30 if args.dataset == "fer" else 12
    if args.lr is None:
        args.lr = (1e-4 if args.dataset == "fer" else 2e-5) if args.architecture == "resnet18_lbp" else (1e-3 if args.dataset == "fer" else 2e-4)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not 0 <= args.label_smoothing < 1 or args.patience < 1:
        parser.error("label-smoothing must be in [0, 1) and patience must be positive")
    if args.architecture == "resnet18_lbp" and not args.init and not args.pretrained_weights:
        parser.error("resnet18_lbp requires --pretrained-weights or --init")
    if args.disable_lbp and args.architecture != "resnet18_lbp":
        parser.error("--disable-lbp is only supported for resnet18_lbp")
    out_dir = args.output or Path("artifacts") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.dataset == "fer":
        splits = read_fer_archive(args.fer_archive) if args.fer_archive else read_fer(args.fer_csv)
    else:
        splits = read_ck(args.ck_root, args.seed)
    (out_dir / "split_summary.json").write_text(
        json.dumps(split_summary(splits), indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (ResNet18LBP(pretrained_path=args.pretrained_weights if not args.init else None,
                         use_lbp=not args.disable_lbp)
             if args.architecture == "resnet18_lbp" else EmotionNet()).to(device)
    if args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=True)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError("Checkpoint class order does not match")
        if checkpoint.get("architecture", "compact") != args.architecture:
            raise ValueError("Checkpoint architecture does not match")
        model.load_state_dict(checkpoint["model"])
    counts = np.bincount(splits["train"][1], minlength=len(CLASSES))
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1)) if args.class_weights == "sqrt_inverse" else np.ones(len(CLASSES))
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                    label_smoothing=args.label_smoothing)
    if args.architecture == "resnet18_lbp":
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr},
            {"params": list(model.lbp_projection.parameters()) + list(model.classifier.parameters()),
             "lr": args.lr * 5},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = (torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
              if args.architecture == "compact" else None)
    amp_dtype = torch.bfloat16 if args.architecture == "resnet18_lbp" else torch.float16
    loaders = {key: make_loader(value, args.batch_size, key == "train", args.strong_augment)
               for key, value in splits.items()}
    history, best_score, stale = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, amp_dtype)
        val = run_epoch(model, loaders["val"], criterion, device, amp_dtype=amp_dtype)
        scheduler.step()
        history.append({"epoch": epoch, "train": train, "val": val})
        print(f"{args.dataset} epoch {epoch}/{args.epochs}: "
              f"train loss={train['loss']:.4f} f1={train['macro_f1']:.4f}; "
              f"val loss={val['loss']:.4f} f1={val['macro_f1']:.4f}", flush=True)
        if val[args.selection] > best_score:
            best_score, stale = val[args.selection], 0
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "dataset": args.dataset, "epoch": epoch, "seed": args.seed,
                        "class_weights": args.class_weights, "selection": args.selection,
                        "architecture": args.architecture, "use_lbp": not args.disable_lbp,
                        "strong_augment": args.strong_augment,
                        "label_smoothing": args.label_smoothing},
                       out_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test = run_epoch(model, loaders["test"], criterion, device, amp_dtype=amp_dtype)
    report = {"dataset": args.dataset, "selected_epoch": checkpoint["epoch"],
              "seed": args.seed, "device": str(device),
              "class_weights": args.class_weights, "selection": args.selection,
              "architecture": args.architecture, "use_lbp": not args.disable_lbp,
              "strong_augment": args.strong_augment,
              "label_smoothing": args.label_smoothing, "learning_rate": args.lr,
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "test": test}
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plots(history, test["confusion_matrix"], out_dir)
    print(json.dumps({"selected_epoch": checkpoint["epoch"], "test_accuracy": test["accuracy"],
                      "test_macro_f1": test["macro_f1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
