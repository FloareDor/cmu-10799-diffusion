"""Compare Canny, XDoG, and PiDiNet (soft safe) on 10 CelebA samples."""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset
from controlnet_aux import PidiNetDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "celeba-subset"


def canny_edges(image_array, sigma=1.0, low=50, high=150):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(np.stack([edges]*3, axis=-1))


def xdog_edges(image_array, sigma=0.5, k=1.6, gamma=0.98, epsilon=0.01, phi=10.0):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1 - gamma * g2
    dog = dog / (dog.max() + 1e-8)
    result = np.where(dog >= epsilon, np.ones_like(dog), 1.0 + np.tanh(phi * (dog - epsilon)))
    result = np.clip(result, 0, 1)
    # Invert so edges are white on black (like Canny)
    inv = (1.0 - result)
    return Image.fromarray((np.stack([inv]*3, axis=-1) * 255).astype(np.uint8))


def main():
    random.seed(999)
    np.random.seed(999)

    ds = load_dataset(
        "electronickale/cmu-10799-celeba64-subset",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"Loaded dataset: {len(ds)} images")

    # Load PiDiNet
    print("Loading PiDiNet...")
    pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    print("PiDiNet loaded.")

    indices = random.sample(range(len(ds)), 10)

    out_dir = REPO_ROOT / "outputs" / "full_edge_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_i, idx in enumerate(indices):
        img = ds[idx]["image"].convert("RGB")
        arr = np.array(img)

        # 1. Canny (sigma=1.0)
        canny_img = canny_edges(arr, sigma=1.0, low=50, high=150)

        # 2. XDoG (sigma=0.5, phi=10) - soft sketch
        xdog1_img = xdog_edges(arr, sigma=0.5, k=1.6, gamma=0.98, epsilon=0.01, phi=10.0)

        # 3. XDoG (sigma=0.8, phi=10) - coarser sketch
        xdog2_img = xdog_edges(arr, sigma=0.8, k=1.6, gamma=0.98, epsilon=0.01, phi=10.0)

        # 4. PiDiNet soft safe
        pidi_img = pidinet(img, safe=True)
        # PiDiNet returns a PIL image (grayscale-ish), convert to RGB for grid
        pidi_img = pidi_img.convert("RGB")

        rows.append((img, canny_img, xdog1_img, xdog2_img, pidi_img))
        print(f"[{row_i+1}/10] idx={idx}")

    # Build grid: 10 rows x 5 cols
    w, h = rows[0][0].size
    cols = 5
    grid = Image.new("RGB", (cols * w, 10 * h), color=(255, 255, 255))
    for r, row_imgs in enumerate(rows):
        for c, im in enumerate(row_imgs):
            # Resize if PiDiNet returns different size
            if im.size != (w, h):
                im = im.resize((w, h), Image.LANCZOS)
            grid.paste(im, (c * w, r * h))

    grid_path = out_dir / "full_edge_compare_grid.png"
    grid.save(grid_path)
    print(f"\nSaved: {grid_path}")
    print("Columns: Original | Canny(s=1.0) | XDoG(s=0.5,phi=10) | XDoG(s=0.8,phi=10) | PiDiNet(soft,safe)")


if __name__ == "__main__":
    main()
