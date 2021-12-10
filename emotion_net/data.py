"""FER2013 and CK+ readers with fixed, leakage-aware evaluation splits."""

import csv
import io
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import CLASSES
from .features import SIZE, lbp_histogram, prepare_image


FER_SPLITS = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
CK_LABELS = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}  # Contempt is excluded.
FRAME_ID = re.compile(r"(S\d{3})_(\d{3})_(\d+)", re.IGNORECASE)


def read_fer(csv_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"FER2013 CSV missing: {csv_path}")
    splits = {key: ([], []) for key in ("train", "val", "test")}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"emotion", "pixels", "Usage"}.issubset(reader.fieldnames or []):
            raise ValueError("FER2013 CSV needs emotion, pixels, Usage columns")
        for line, row in enumerate(reader, start=2):
            split = FER_SPLITS.get(row["Usage"])
            if split is None:
                raise ValueError(f"Unknown FER2013 Usage on line {line}: {row['Usage']}")
            label = int(row["emotion"])
            image = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
            if not 0 <= label < len(CLASSES) or image.size != SIZE * SIZE:
                raise ValueError(f"Invalid FER2013 label or pixels on line {line}")
            splits[split][0].append(image.reshape(SIZE, SIZE))
            splits[split][1].append(label)
    return {name: (np.stack(images), np.asarray(labels, dtype=np.int64))
            for name, (images, labels) in splits.items()}


def read_fer_archive(archive: Path):
    """Load an image-folder FER2013 ZIP retaining filename-encoded splits."""
    if not archive.is_file():
        raise FileNotFoundError(f"FER2013 archive missing: {archive}")
    class_to_id = {name: index for index, name in enumerate(CLASSES)}
    splits = {key: ([], []) for key in ("train", "val", "test")}
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir() or Path(member.filename).suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            parts = member.filename.replace("\\", "/").split("/")
            if len(parts) < 3 or parts[-2].lower() not in class_to_id:
                raise ValueError(f"Unexpected FER2013 archive path: {member.filename}")
            prefix = parts[-1].split("_", 1)[0]
            split = FER_SPLITS.get(prefix)
            if split is None:
                raise ValueError(f"Missing FER2013 split in filename: {member.filename}")
            with zip_file.open(member) as stream, Image.open(stream) as image:
                array = prepare_image(image)
            splits[split][0].append(array)
            splits[split][1].append(class_to_id[parts[-2].lower()])
    if any(not values[0] for values in splits.values()):
        raise ValueError("FER2013 archive must contain train, PublicTest and PrivateTest images")
    return {name: (np.stack(images), np.asarray(labels, dtype=np.int64))
            for name, (images, labels) in splits.items()}


def read_ck_csv_archive(archive: Path):
    """Read a CK+ CSV for external evaluation only (no subject identities)."""
    if not archive.is_file():
        raise FileNotFoundError(f"CK+ archive missing: {archive}")
    images, labels = [], []
    with ZipFile(archive) as zip_file:
        csv_members = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError("CK+ archive should contain exactly one CSV")
        with zip_file.open(csv_members[0]) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            if not {"emotion", "pixels"}.issubset(reader.fieldnames or []):
                raise ValueError("CK+ CSV needs emotion and pixels columns")
            for row in reader:
                label = int(row["emotion"])
                if label == 7:  # Contempt is outside the seven-class target.
                    continue
                image = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
                if not 0 <= label < len(CLASSES) or image.size != SIZE * SIZE:
                    raise ValueError("Invalid CK+ label or pixels")
                images.append(image.reshape(SIZE, SIZE))
                labels.append(label)
    if not images:
        raise ValueError("CK+ archive has no compatible images")
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def read_ck(root: Path, seed: int = 42):
    """Read labelled apex and initial neutral frame; split by subject identity."""
    if not root.is_dir():
        raise FileNotFoundError(f"CK+ directory missing: {root}")
    frames = defaultdict(list)
    for path in root.rglob("*"):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        match = FRAME_ID.search(path.stem)
        if match:
            frames[(match[1].upper(), match[2])].append((int(match[3]), path))
    records = []
    for label_path in root.rglob("*_emotion.txt"):
        match = FRAME_ID.search(label_path.stem)
        if not match:
            continue
        key = (match[1].upper(), match[2])
        sequence = sorted(frames.get(key, []))
        if len(sequence) < 2:
            continue
        raw_label = int(float(label_path.read_text(encoding="utf-8").strip()))
        if raw_label not in CK_LABELS:
            continue
        records.append((key[0], sequence[-1][1], CK_LABELS[raw_label]))
        records.append((key[0], sequence[0][1], 6))
    if not records:
        raise ValueError("No CK+ labelled sequences found. Include images and Emotion/*_emotion.txt files.")
    subjects = np.array(sorted({record[0] for record in records}))
    if len(subjects) < 10:
        raise ValueError("CK+ needs at least 10 subjects for a useful subject-held-out split")
    np.random.default_rng(seed).shuffle(subjects)
    n_test = max(1, round(len(subjects) * 0.15))
    n_val = max(1, round(len(subjects) * 0.15))
    test_ids = set(subjects[:n_test])
    val_ids = set(subjects[n_test:n_test + n_val])
    splits = {key: ([], [], []) for key in ("train", "val", "test")}
    for subject, path, label in records:
        split = "test" if subject in test_ids else "val" if subject in val_ids else "train"
        with Image.open(path) as image:
            array = prepare_image(image)
        splits[split][0].append(array)
        splits[split][1].append(label)
        splits[split][2].append(subject)
    output = {}
    for name, (images, labels, identities) in splits.items():
        output[name] = (np.stack(images), np.asarray(labels, dtype=np.int64), identities)
    return output


class FaceDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False,
                 strong_augment: bool = False):
        if len(images) != len(labels):
            raise ValueError("Image and label counts differ")
        self.images = images
        self.labels = labels
        self.augment = augment
        self.strong_augment = strong_augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            if np.random.random() < 0.5:
                image = np.fliplr(image)
            if self.strong_augment:
                import cv2

                transform = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2),
                                                     float(np.random.uniform(-12, 12)),
                                                     float(np.random.uniform(0.9, 1.1)))
                transform[:, 2] += np.random.uniform(-3, 3, size=2)
                image = cv2.warpAffine(np.ascontiguousarray(image), transform,
                                       (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT_101)
            elif np.random.random() < 0.5:
                angle = float(np.random.uniform(-8, 8))
                image = np.asarray(Image.fromarray(image).rotate(angle, resample=Image.Resampling.BILINEAR))
            if np.random.random() < 0.5:
                gain = np.random.uniform(0.85, 1.15)
                shift = np.random.uniform(-12, 12)
                image = np.clip(image.astype(np.float32) * gain + shift, 0, 255).astype(np.uint8)
            if self.strong_augment and np.random.random() < 0.35:
                image = np.array(image, copy=True)
                size = int(np.random.randint(4, 9))
                top = int(np.random.randint(0, SIZE - size + 1))
                left = int(np.random.randint(0, SIZE - size + 1))
                image[top:top + size, left:left + size] = int(image.mean())
        image = np.ascontiguousarray(image)
        lbp = lbp_histogram(image)
        tensor = torch.from_numpy(image.astype(np.float32)[None] / 127.5 - 1)
        return tensor, torch.from_numpy(lbp), int(self.labels[index])


def split_summary(splits):
    return {name: {"samples": len(items[1]),
                   "classes": {CLASSES[i]: Counter(items[1].tolist()).get(i, 0)
                               for i in range(len(CLASSES))},
                   **({"subjects": len(set(items[2])),
                       "subject_ids": sorted(set(items[2]))} if len(items) > 2 else {})}
            for name, items in splits.items()}
