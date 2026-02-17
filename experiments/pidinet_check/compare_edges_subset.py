import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from controlnet_aux import PidiNetDetector

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


def canny_edges_rgb(image: Image.Image, sigma: float = 1.0, low: int = 50, high: int = 150) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(edges, mode="L").convert("RGB")


def xdog_edges_rgb(
    image: Image.Image,
    sigma: float = 0.5,
    k: float = 1.6,
    gamma: float = 0.98,
    epsilon: float = 0.01,
    phi: float = 10.0,
) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1 - gamma * g2
    result = np.where(dog >= epsilon, 1.0, 1.0 + np.tanh(phi * (dog - epsilon)))
    result = np.clip(result, 0.0, 1.0)
    edges = (1.0 - result) * 255.0  # white edges on black
    return Image.fromarray(edges.astype(np.uint8), mode="L").convert("RGB")


def pidinet_maps(image: Image.Image, detector: PidiNetDetector, threshold: float = 0.5) -> tuple[Image.Image, Image.Image]:
    soft = detector(image.convert("RGB"), safe=True).convert("L")
    soft_np = np.array(soft, dtype=np.uint8)
    binary_np = (soft_np >= int(threshold * 255)).astype(np.uint8) * 255
    binary = Image.fromarray(binary_np, mode="L")
    return soft.convert("RGB"), binary.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PiDiNet, Canny, and XDoG on CelebA subset")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data/celeba-subset")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="outputs/edge_compare")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_celeba_dataset(args.data_root)
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print("Loading PiDiNet detector...")
    detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")

    # Columns: original | pidinet_soft_safe | pidinet_binary | canny | xdog
    rows = []
    for row_i, idx in enumerate(indices):
        img = dataset[idx].convert("RGB")
        pidi_soft, pidi_bin = pidinet_maps(img, detector, threshold=args.threshold)
        canny = canny_edges_rgb(img)
        xdog = xdog_edges_rgb(img)

        rows.append((img, pidi_soft, pidi_bin, canny, xdog))

        img.save(out_dir / f"sample_{row_i:02d}_orig.png")
        pidi_soft.save(out_dir / f"sample_{row_i:02d}_pidinet_soft_safe.png")
        pidi_bin.save(out_dir / f"sample_{row_i:02d}_pidinet_binary.png")
        canny.save(out_dir / f"sample_{row_i:02d}_canny.png")
        xdog.save(out_dir / f"sample_{row_i:02d}_xdog.png")
        print(f"[{row_i + 1}/{len(indices)}] wrote sample_{row_i:02d}_*")

    w, h = rows[0][0].size
    cols = 5
    grid = Image.new("RGB", (cols * w, len(rows) * h), color=(255, 255, 255))
    for r, imgs in enumerate(rows):
        for c, im in enumerate(imgs):
            grid.paste(im, (c * w, r * h))

    grid_path = out_dir / "edge_comparison_grid.png"
    grid.save(grid_path)
    print(f"Saved grid: {grid_path}")
    print("Columns: original | pidinet_soft_safe | pidinet_binary | canny | xdog")


if __name__ == "__main__":
    main()
