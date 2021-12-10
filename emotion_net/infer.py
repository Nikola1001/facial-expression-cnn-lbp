"""Single-image prediction shared by the command line and desktop UI."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from . import CLASSES
from .ensemble import FlipEnsemble, LBPFusion
from .features import lbp_histogram, prepare_image
from .model import EmotionNet, ResNet18LBP, LBPExpert


def load_model(checkpoint_path: Path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path.suffix.lower() == ".json":
        manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if manifest.get("type") == "flip_ensemble":
            models = [load_model((checkpoint_path.parent / item).resolve(), device)[0]
                      for item in manifest["checkpoints"]]
            return FlipEnsemble(models, manifest["weights"]).to(device).eval(), device
        if manifest.get("type") == "lbp_fusion":
            cnn = load_model((checkpoint_path.parent / manifest["cnn"]).resolve(), device)[0]
            expert = load_model((checkpoint_path.parent / manifest["lbp_expert"]).resolve(), device)[0]
            return LBPFusion(cnn, expert, manifest["lbp_weight"]).to(device).eval(), device
        raise ValueError("Unknown ensemble manifest type")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(saved["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order does not match")
    architecture = saved.get("architecture", "compact")
    if architecture == "resnet18_lbp":
        model = ResNet18LBP(use_lbp=saved.get("use_lbp", True)).to(device)
    elif architecture == "compact":
        model = EmotionNet().to(device)
    elif architecture == "lbp_expert":
        model = LBPExpert().to(device)
    else:
        raise ValueError(f"Unknown checkpoint architecture: {architecture}")
    model.load_state_dict(saved["model"])
    model.eval()
    return model, device


def crop_face(image: Image.Image) -> Image.Image:
    """Use the largest detected face; small images are treated as existing crops."""
    image = ImageOps.exif_transpose(image)
    if image.width <= 64 and image.height <= 64:
        return image
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    if len(faces) == 0:
        raise ValueError("Face not detected. Supply a close, frontal face crop.")
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad = int(max(w, h) * 0.08)
    return image.crop((max(0, x - pad), max(0, y - pad),
                       min(image.width, x + w + pad), min(image.height, y + h + pad)))


@torch.inference_mode()
def predict(image: Image.Image, model, device, detect: bool = True):
    crop = crop_face(image) if detect else image
    gray = prepare_image(crop)
    image_tensor = torch.from_numpy(gray.astype(np.float32)[None, None] / 127.5 - 1).to(device)
    lbp_tensor = torch.from_numpy(lbp_histogram(gray)[None]).to(device)
    probabilities = model(image_tensor, lbp_tensor).softmax(1)[0].cpu().numpy()
    return {label: float(probabilities[i]) for i, label in enumerate(CLASSES)}, crop


def main():
    parser = argparse.ArgumentParser(description="Predict one face image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/fer/best.pt"))
    parser.add_argument("--no-detect", action="store_true", help="Image is already a face crop")
    args = parser.parse_args()
    model, device = load_model(args.checkpoint)
    with Image.open(args.image) as image:
        probabilities, _ = predict(image, model, device, detect=not args.no_detect)
    print(json.dumps(dict(sorted(probabilities.items(), key=lambda item: -item[1])), indent=2))


if __name__ == "__main__":
    main()
