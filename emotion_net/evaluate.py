"""Independent evaluation of a trained checkpoint on CK+ CSV."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import FaceDataset, read_ck_csv_archive
from .experiment import run_epoch, save_confusion
from .infer import load_model
from .ensemble import FlipEnsemble, LBPFusion
from .model import ResNet18LBP


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on CK+ CSV")
    parser.add_argument("--ck-archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/fer/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ck_external"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    images, labels = read_ck_csv_archive(args.ck_archive)
    model, device = load_model(args.checkpoint)
    loader = DataLoader(FaceDataset(images, labels), batch_size=args.batch_size, num_workers=0)
    amp_dtype = (torch.bfloat16 if isinstance(model, (ResNet18LBP, FlipEnsemble, LBPFusion))
                 else torch.float16)
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), device,
                        amp_dtype=amp_dtype)
    matrix = metrics["confusion_matrix"]
    non_neutral_total = sum(sum(row) for row in matrix[:6])
    non_neutral_correct = sum(matrix[index][index] for index in range(6))
    metrics["non_neutral_accuracy"] = non_neutral_correct / non_neutral_total
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "CK+ CSV", "evaluation": "external_fer_model_no_ck_training",
              "samples": len(labels), "excluded_contempt": True, "metrics": metrics}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion(metrics["confusion_matrix"], args.output / "confusion_matrix.png")
    print(json.dumps({"samples": len(labels), "accuracy": metrics["accuracy"],
                      "macro_f1": metrics["macro_f1"],
                      "non_neutral_accuracy": metrics["non_neutral_accuracy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
