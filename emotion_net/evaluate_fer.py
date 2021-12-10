"""Evaluate a saved model or ensemble on a fixed FER2013 split."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .data import read_fer
from .ensemble import FlipEnsemble, LBPFusion
from .experiment import make_loader, run_epoch, save_confusion
from .infer import load_model
from .model import ResNet18LBP


def main():
    parser = argparse.ArgumentParser(description="Evaluate FER2013 validation or test split")
    parser.add_argument("--fer-csv", type=Path, default=Path("data/raw/fer2013.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ensemble"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    split = read_fer(args.fer_csv)[args.split]
    model, device = load_model(args.checkpoint)
    amp_dtype = (torch.bfloat16 if isinstance(model, (ResNet18LBP, FlipEnsemble, LBPFusion))
                 else torch.float16)
    metrics = run_epoch(model, make_loader(split, args.batch_size, False),
                        nn.CrossEntropyLoss(), device, amp_dtype=amp_dtype)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "FER2013", "split": args.split, "samples": len(split[1]),
              "checkpoint": str(args.checkpoint), "metrics": metrics}
    (args.output / f"{args.split}_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    save_confusion(metrics["confusion_matrix"],
                   args.output / f"{args.split}_confusion_matrix.png")
    print(json.dumps({"split": args.split, "samples": len(split[1]),
                      "accuracy": metrics["accuracy"],
                      "macro_f1": metrics["macro_f1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
