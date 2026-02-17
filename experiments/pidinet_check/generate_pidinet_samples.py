import argparse
import random
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from controlnet_aux import PidiNetDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CelebADataset


def load_celeba_dataset(root: str) -> CelebADataset:
    """Load CelebA, preferring local cache and falling back to HF Hub."""
    try:
        ds = CelebADataset(root=root, split="train", augment=False, from_hub=False)
        ds.transform = None
        return ds
    except Exception:
        ds = CelebADataset(root=root, split="train", augment=False, from_hub=True)
        ds.transform = None
        return ds


def make_binary_edge(edge_img: Image.Image, threshold: float) -> Image.Image:
    gray = np.array(edge_img.convert("L"), dtype=np.uint8)
    bin_img = (gray >= int(threshold * 255)).astype(np.uint8) * 255
    return Image.fromarray(bin_img, mode="L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PiDiNet samples from CelebA")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--data-root", type=str, default="./data/celeba-subset")
    parser.add_argument("--out-dir", type=str, default="outputs/pidinet_check")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_celeba_dataset(args.data_root)
    total = len(dataset)
    if total == 0:
        raise RuntimeError("CelebA dataset appears empty.")

    sample_count = min(args.num_samples, total)
    indices = random.sample(range(total), sample_count)

    print("Loading PiDiNet (lllyasviel/Annotators) ...")
    processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")

    rows = []
    for i, idx in enumerate(indices):
        img = dataset[idx]
        if not isinstance(img, Image.Image):
            raise RuntimeError(f"Expected PIL image at idx={idx}, got {type(img)}")
        img = img.convert("RGB")

        # Good practice from ControlNet preprocessing: use safe=True.
        edge_soft = processor(img, safe=True)
        edge_soft = edge_soft.convert("RGB")
        edge_bin = make_binary_edge(edge_soft, threshold=args.threshold)

        img.save(out_dir / f"sample_{i:02d}_orig.png")
        edge_soft.save(out_dir / f"sample_{i:02d}_pidinet_soft_safe.png")
        edge_bin.save(out_dir / f"sample_{i:02d}_pidinet_binary_t{args.threshold:.2f}.png")

        rows.append((img, edge_soft, edge_bin))
        print(f"[{i + 1}/{sample_count}] wrote sample_{i:02d}_*")

    w, h = rows[0][0].size
    grid = Image.new("RGB", (3 * w, sample_count * h), color=(255, 255, 255))
    for r, (orig, soft, binary) in enumerate(rows):
        y = r * h
        grid.paste(orig, (0, y))
        grid.paste(soft, (w, y))
        grid.paste(binary, (2 * w, y))

    grid_path = out_dir / "pidinet_comparison_grid.png"
    grid.save(grid_path)
    print(f"Saved grid: {grid_path}")
    print("Columns: original | pidinet_soft_safe | pidinet_binary")


if __name__ == "__main__":
    main()
