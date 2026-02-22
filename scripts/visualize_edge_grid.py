"""
Visualize all edge methods and mushy augmentations side-by-side.

Grid 1 — Edge Methods (rows) × Sample Images (cols):
  original | xdog:0.28 | canny:1.0 | canny:1.2 | canny:1.5 | canny:1.75 | canny:2.0 | canny:2.3

Grid 2 — Mushy Augmentations (rows) × Sample Images (cols):
  mixed_blend | binary | blur | dilate | erode | dropout

Run from repo root:
  python scripts/visualize_edge_grid.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datasets
import cv2
from PIL import Image

from data.celeba import canny_edges, xdog_edges, _ensure_white_edges_on_black, _gray_to_rgb_pil

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = (
    r"e:\cmu-e\courses\diffusion\homeworks\cmu-10799-diffusion"
    r"\data\celeba\electronickale___cmu-10799-celeba64-subset"
    r"\default\0.0.0\cea8d2312303971a09528db035498464cbb01e37"
)

NUM_IMAGES   = 5      # columns
SAMPLE_SEED  = 7
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")

EDGE_METHODS = [
    ("xdog:0.28",   "XDoG σ=0.28"),
    ("canny:1.5",   "Canny σ=1.5"),
    ("canny:1.75",  "Canny σ=1.75"),
    ("canny:2.0",   "Canny σ=2.0"),
    ("canny:2.3",   "Canny σ=2.3"),
]

CANNY_LOW, CANNY_HIGH = 80, 200

# Dirichlet alpha matching your config (uniform → each method ~14%)
DIRICHLET_ALPHA = [1.0] * len(EDGE_METHODS)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_images(n: int, seed: int) -> list[Image.Image]:
    print(f"Loading dataset from disk ...")
    arrow_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".arrow")]
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {DATASET_PATH}")
    arrow_path = os.path.join(DATASET_PATH, arrow_files[0])
    ds = datasets.Dataset.from_file(arrow_path)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=n, replace=False).tolist()
    images = [ds[int(i)]["image"].convert("RGB").resize((64, 64), Image.BILINEAR) for i in idxs]
    print(f"Loaded {n} images (indices {idxs})")
    return images


def apply_edge(image: Image.Image, method: str) -> Image.Image:
    base, _, sigma_str = method.partition(":")
    sigma = float(sigma_str) if sigma_str else None
    if base == "canny":
        return canny_edges(image, sigma=sigma or 1.2, low=CANNY_LOW, high=CANNY_HIGH)
    if base == "xdog":
        return xdog_edges(image, sigma=sigma or 0.29)
    raise ValueError(f"Unknown method: {method}")


def make_mix(image: Image.Image, alpha: list[float], seed: int | None = None) -> np.ndarray:
    """Weighted blend of all 7 edge methods. Returns float32 [0,1] array."""
    rng = np.random.default_rng(seed)
    maps = []
    for method, _ in EDGE_METHODS:
        emap = apply_edge(image, method)
        gray = np.array(emap.convert("L"), dtype=np.float32) / 255.0
        maps.append(gray)
    stacked = np.stack(maps, axis=0)
    weights = rng.dirichlet(alpha).astype(np.float32)
    mix = np.tensordot(weights, stacked, axes=(0, 0))
    return np.clip(mix, 0.0, 1.0)


def as_pil(arr01: np.ndarray) -> Image.Image:
    u8 = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    u8 = _ensure_white_edges_on_black(u8)
    return _gray_to_rgb_pil(u8)


# Mushy augmentations applied at 100% probability so the effect is always visible.

def aug_binary(x: np.ndarray) -> np.ndarray:
    thr = 0.5
    return (x >= thr).astype(np.float32)

def aug_blur(x: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), sigmaX=1.0, sigmaY=1.0)

def aug_dilate(x: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.dilate(x.astype(np.float32), kernel, iterations=1)

def aug_erode(x: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.erode(x.astype(np.float32), kernel, iterations=1)

def aug_dropout(x: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    keep = rng.random(x.shape) > 0.3
    return x * keep.astype(np.float32)

MUSHY_AUGS = [
    ("blend (no aug)", None),
    ("dropout 30%", aug_dropout),
]

# ── Plot helpers ───────────────────────────────────────────────────────────────

def show(ax, img, title="", fontsize=7):
    if isinstance(img, np.ndarray):
        img = as_pil(img)
    ax.imshow(img)
    ax.set_title(title, fontsize=fontsize, pad=2)
    ax.axis("off")


def plot_grid1(images: list[Image.Image]) -> plt.Figure:
    """Rows = edge methods (+original), cols = images."""
    rows = 1 + len(EDGE_METHODS)  # original + 7 methods
    cols = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    fig.suptitle("Grid 1 — Edge Methods", fontsize=10, fontweight="bold", y=1.01)

    for c, img in enumerate(images):
        show(axes[0, c], img, "original" if c == 0 else "")

    for r, (method, label) in enumerate(EDGE_METHODS, start=1):
        for c, img in enumerate(images):
            edge = apply_edge(img, method)
            show(axes[r, c], edge, label if c == 0 else "")

    plt.tight_layout()
    return fig


def plot_grid2(images: list[Image.Image]) -> plt.Figure:
    """Rows = mushy augmentations (applied to the mixed blend), cols = images."""
    rows = len(MUSHY_AUGS)
    cols = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    fig.suptitle("Grid 2 — Mushy Augmentations (on mixed blend)", fontsize=10, fontweight="bold", y=1.01)

    for c, img in enumerate(images):
        mix = make_mix(img, DIRICHLET_ALPHA, seed=42)
        for r, (label, fn) in enumerate(MUSHY_AUGS):
            result = fn(mix) if fn is not None else mix
            show(axes[r, c], result, label if c == 0 else "")

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = load_images(NUM_IMAGES, SAMPLE_SEED)

    print("Rendering Grid 1 (edge methods) ...")
    fig1 = plot_grid1(images)
    out1 = os.path.join(OUTPUT_DIR, "edge_methods_grid.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")

    print("Rendering Grid 2 (mushy augmentations) ...")
    fig2 = plot_grid2(images)
    out2 = os.path.join(OUTPUT_DIR, "mushy_aug_grid.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")

    plt.close("all")


if __name__ == "__main__":
    main()
