"""Generate Canny edge comparison grid for 10 CelebA samples with sigma=1.0."""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "celeba-subset"


def canny_edges(img_array, sigma=1.0, low=50, high=150):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    edges = cv2.Canny(gray, low, high)
    return edges


def main():
    random.seed(999)
    np.random.seed(999)

    # Load dataset from HuggingFace (uses cache)
    ds = load_dataset(
        "electronickale/cmu-10799-celeba64-subset",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"Loaded dataset: {len(ds)} images")

    # Pick 10 random indices
    indices = random.sample(range(len(ds)), 10)

    out_dir = REPO_ROOT / "outputs" / "canny_final_10"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings: our recommended sigma=1.0 with (50,150)
    rows = []
    for row_i, idx in enumerate(indices):
        item = ds[idx]
        img = item["image"].convert("RGB")
        arr = np.array(img)

        edges = canny_edges(arr, sigma=1.0, low=50, high=150)
        edges_rgb = Image.fromarray(np.stack([edges]*3, axis=-1))

        rows.append((img, edges_rgb))
        print(f"[{row_i+1}/10] idx={idx}")

    # Build grid: 10 rows x 2 cols (original | canny)
    w, h = rows[0][0].size
    grid = Image.new("RGB", (2 * w, 10 * h), color=(255, 255, 255))
    for r, (orig, edge) in enumerate(rows):
        grid.paste(orig, (0, r * h))
        grid.paste(edge, (w, r * h))

    grid_path = out_dir / "canny_s1.0_grid_10.png"
    grid.save(grid_path)
    print(f"\nSaved: {grid_path}")
    print("Columns: original | Canny (sigma=1.0, low=50, high=150)")


if __name__ == "__main__":
    main()
