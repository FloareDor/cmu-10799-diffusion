import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CelebADataset


def load_celeba_dataset(root: str) -> CelebADataset:
    try:
        ds = CelebADataset(root=root, split="train", augment=False, from_hub=False)
        ds.transform = None
        return ds
    except Exception:
        ds = CelebADataset(root=root, split="train", augment=False, from_hub=True)
        ds.transform = None
        return ds


def canny_edges_rgb(image: Image.Image, sigma: float, low: int, high: int) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(edges).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Canny with multiple sigma settings.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data/celeba-subset")
    parser.add_argument("--out-dir", type=str, default="outputs/canny_sigma_compare")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_celeba_dataset(args.data_root)
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    # Keep thresholds fixed first to isolate sigma effect.
    settings = [
        ("canny_s1.0_l50_h150", 1.0, 50, 150),
        ("canny_s1.6_l50_h150", 1.6, 50, 150),
        ("canny_s2.2_l50_h150", 2.2, 50, 150),
    ]

    # Columns: original + each canny setting
    rows = []
    for row_i, idx in enumerate(indices):
        img = dataset[idx].convert("RGB")
        row_imgs = [img]
        img.save(out_dir / f"sample_{row_i:02d}_orig.png")

        for name, sigma, low, high in settings:
            edge = canny_edges_rgb(img, sigma=sigma, low=low, high=high)
            edge.save(out_dir / f"sample_{row_i:02d}_{name}.png")
            row_imgs.append(edge)

        rows.append(row_imgs)
        print(f"[{row_i + 1}/{len(indices)}] wrote sample_{row_i:02d}_*")

    w, h = rows[0][0].size
    cols = 1 + len(settings)
    grid = Image.new("RGB", (cols * w, len(rows) * h), color=(255, 255, 255))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            grid.paste(im, (c * w, r * h))

    grid_path = out_dir / "canny_sigma_grid.png"
    grid.save(grid_path)
    print(f"Saved grid: {grid_path}")
    print("Columns: original | s1.0 | s1.6 | s2.2 (all low=50 high=150)")


if __name__ == "__main__":
    main()
