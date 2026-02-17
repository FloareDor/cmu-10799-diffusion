import argparse
import csv
import random
import sys
from pathlib import Path

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


def edge_density(binary_edge: np.ndarray) -> float:
    # Fraction of white edge pixels in [0, 1].
    return float((binary_edge > 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PiDiNet binary thresholds by edge density.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55])
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data/celeba-subset")
    parser.add_argument("--target-density", type=float, default=0.12)
    parser.add_argument("--out-dir", type=str, default="outputs/pidinet_sweep")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_celeba_dataset(args.data_root)
    n = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    print(f"Loaded dataset with {len(dataset)} images; evaluating {n} samples")
    print("Loading PiDiNet detector...")
    detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")

    per_threshold = {t: [] for t in args.thresholds}

    for i, idx in enumerate(indices):
        img: Image.Image = dataset[idx].convert("RGB")
        soft = detector(img, safe=True).convert("L")
        soft_np = np.array(soft, dtype=np.uint8)

        for t in args.thresholds:
            bin_np = (soft_np >= int(t * 255)).astype(np.uint8) * 255
            d = edge_density(bin_np)
            per_threshold[t].append(d)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[{i + 1}/{n}] processed")

    rows = []
    for t in args.thresholds:
        vals = np.array(per_threshold[t], dtype=np.float64)
        mean_d = float(vals.mean())
        std_d = float(vals.std())
        min_d = float(vals.min())
        max_d = float(vals.max())
        # Heuristic score: closeness to target + low variance.
        score = abs(mean_d - args.target_density) + 0.25 * std_d
        rows.append(
            {
                "threshold": t,
                "mean_density": mean_d,
                "std_density": std_d,
                "min_density": min_d,
                "max_density": max_d,
                "score": score,
            }
        )

    rows = sorted(rows, key=lambda r: r["score"])
    best = rows[0]

    csv_path = out_dir / "pidinet_threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "mean_density",
                "std_density",
                "min_density",
                "max_density",
                "score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = out_dir / "pidinet_threshold_recommendation.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("PiDiNet safe=True threshold sweep\n")
        f.write(f"num_samples={n}\n")
        f.write(f"target_density={args.target_density}\n\n")
        f.write("Sorted by score (lower is better):\n")
        for row in rows:
            f.write(
                f"t={row['threshold']:.2f} "
                f"mean={row['mean_density']:.4f} "
                f"std={row['std_density']:.4f} "
                f"min={row['min_density']:.4f} "
                f"max={row['max_density']:.4f} "
                f"score={row['score']:.4f}\n"
            )
        f.write("\n")
        f.write(f"RECOMMENDED_THRESHOLD={best['threshold']:.2f}\n")

    print(f"Saved sweep stats to: {csv_path}")
    print(f"Saved recommendation to: {txt_path}")
    print(
        "Best threshold: "
        f"{best['threshold']:.2f} "
        f"(mean_density={best['mean_density']:.4f}, std={best['std_density']:.4f})"
    )


if __name__ == "__main__":
    main()
