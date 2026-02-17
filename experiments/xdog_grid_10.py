"""Generate XDoG edge comparison grid for 10 CelebA samples."""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "celeba-subset"


def xdog_edges(image_array, sigma=0.5, k=1.6, gamma=0.98, epsilon=0.01, phi=10.0):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1 - gamma * g2
    dog = dog / (dog.max() + 1e-8)
    result = np.where(dog >= epsilon, np.ones_like(dog), 1.0 + np.tanh(phi * (dog - epsilon)))
    return np.clip(result, 0, 1)  # 0=edge(black), 1=background(white)


def canny_edges(image_array, sigma=1.0, low=50, high=150):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    edges = cv2.Canny(gray, low, high)
    return edges  # 0/255 uint8


def main():
    random.seed(999)
    np.random.seed(999)

    ds = load_dataset(
        "electronickale/cmu-10799-celeba64-subset",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"Loaded dataset: {len(ds)} images")

    indices = random.sample(range(len(ds)), 10)

    out_dir = REPO_ROOT / "outputs" / "xdog_vs_canny_10"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_i, idx in enumerate(indices):
        img = ds[idx]["image"].convert("RGB")
        arr = np.array(img)

        # Canny (our current pick)
        canny = canny_edges(arr, sigma=1.0, low=50, high=150)
        canny_rgb = Image.fromarray(np.stack([canny]*3, axis=-1))

        # XDoG variants
        xdog1 = xdog_edges(arr, sigma=0.5, k=1.6, gamma=0.98, epsilon=0.01, phi=10.0)
        xdog1_inv = (1.0 - xdog1)  # invert so edges are white
        xdog1_rgb = Image.fromarray((np.stack([xdog1_inv]*3, axis=-1) * 255).astype(np.uint8))

        xdog2 = xdog_edges(arr, sigma=0.5, k=1.6, gamma=0.98, epsilon=0.0, phi=200.0)
        xdog2_inv = (1.0 - xdog2)
        xdog2_rgb = Image.fromarray((np.stack([xdog2_inv]*3, axis=-1) * 255).astype(np.uint8))

        xdog3 = xdog_edges(arr, sigma=0.8, k=1.6, gamma=0.98, epsilon=0.01, phi=10.0)
        xdog3_inv = (1.0 - xdog3)
        xdog3_rgb = Image.fromarray((np.stack([xdog3_inv]*3, axis=-1) * 255).astype(np.uint8))

        rows.append((img, canny_rgb, xdog1_rgb, xdog2_rgb, xdog3_rgb))
        print(f"[{row_i+1}/10] idx={idx}")

    # Build grid: 10 rows x 5 cols
    w, h = rows[0][0].size
    cols = 5
    grid = Image.new("RGB", (cols * w, 10 * h), color=(255, 255, 255))
    for r, row_imgs in enumerate(rows):
        for c, im in enumerate(row_imgs):
            grid.paste(im, (c * w, r * h))

    grid_path = out_dir / "xdog_vs_canny_grid.png"
    grid.save(grid_path)
    print(f"\nSaved: {grid_path}")
    print("Columns: original | Canny(s=1.0) | XDoG(s=0.5,phi=10) | XDoG(s=0.5,phi=200) | XDoG(s=0.8,phi=10)")


if __name__ == "__main__":
    main()
