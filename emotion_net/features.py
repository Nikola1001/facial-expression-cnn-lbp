"""Deterministic image preprocessing and uniform LBP histograms."""

import numpy as np
from PIL import Image, ImageOps


SIZE = 48
GRID = 4
BINS = 10


def prepare_image(image: Image.Image) -> np.ndarray:
    """Return a 48x48 grayscale uint8 face crop.

    The caller supplies a face crop; detection is intentionally outside this function.
    """
    gray = ImageOps.grayscale(ImageOps.exif_transpose(image))
    return np.asarray(gray.resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)


def _uniform_lut() -> np.ndarray:
    lut = np.full(256, 9, dtype=np.uint8)
    for code in range(256):
        bits = np.array([(code >> i) & 1 for i in range(8)], dtype=np.uint8)
        if np.count_nonzero(bits != np.roll(bits, 1)) <= 2:
            lut[code] = bits.sum()
    return lut


UNIFORM_LUT = _uniform_lut()
REGION_IDS = ((np.arange(SIZE)[:, None] // (SIZE // GRID)) * GRID
              + np.arange(SIZE)[None, :] // (SIZE // GRID)).astype(np.int32)


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """4x4 spatial histogram of rotation-invariant uniform 8-neighbor LBP."""
    if gray.shape != (SIZE, SIZE) or gray.dtype != np.uint8:
        raise ValueError("LBP expects a 48x48 uint8 grayscale image")
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1))
    code = np.zeros_like(center, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy:1 + dy + SIZE, 1 + dx:1 + dx + SIZE]
        code |= (neighbor >= center).astype(np.uint8) << bit
    mapped = UNIFORM_LUT[code]
    bins = (REGION_IDS * BINS + mapped).ravel()
    hist = np.bincount(bins, minlength=GRID * GRID * BINS).astype(np.float32)
    return hist / ((SIZE // GRID) ** 2)
