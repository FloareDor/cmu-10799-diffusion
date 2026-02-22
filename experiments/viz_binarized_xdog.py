"""
Visualize randomly binarized XDoG edge maps from local CelebA data.
Grid: original | raw XDoG | binarized XDoG (random tau per image)
"""
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")

import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "./data/celeba"   # HF cache dir
N = 8          # images to show
SEED = 42
SIGMA = 0.29   # XDoG sigma
K = 1.6
GAMMA = 0.98
EPSILON = 0.01
PHI = 10.0
TAU_RANGE = (0.90, 0.99)   # random binarization threshold per image
# ────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)

from datasets import load_dataset
dataset = load_dataset("electronickale/cmu-10799-celeba64-subset", split="train", cache_dir=DATA_PATH)
split = dataset
indices = random.sample(range(len(split)), N)

def xdog_raw(pil_img, sigma=SIGMA, k=K, gamma=GAMMA, epsilon=EPSILON, phi=PHI):
    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma * k)
    dog = g1 - gamma * g2
    dog = dog / (np.max(np.abs(dog)) + 1e-8)
    result = np.where(dog >= epsilon, np.ones_like(dog), 1.0 + np.tanh(phi * (dog - epsilon)))
    result = np.clip(result, 0.0, 1.0)
    edges = 1.0 - result   # white edges on black, float [0,1]
    if edges.mean() > 0.5:
        edges = 1.0 - edges
    return edges

fig, axes = plt.subplots(N, 3, figsize=(9, N * 3 + 0.5))
fig.suptitle("Original  |  Raw XDoG  |  Binarized XDoG (random τ)", fontsize=13)

for row, idx in enumerate(indices):
    item = split[idx]
    pil = item["image"].convert("RGB")

    raw = xdog_raw(pil)

    tau = random.uniform(*TAU_RANGE)
    binary = (raw >= tau).astype(np.float32)

    axes[row, 0].imshow(pil)
    axes[row, 0].axis("off")
    if row == 0:
        axes[row, 0].set_title("Original", fontsize=10)

    axes[row, 1].imshow(raw, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].axis("off")
    if row == 0:
        axes[row, 1].set_title("Raw XDoG", fontsize=10)

    axes[row, 2].imshow(binary, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].axis("off")
    if row == 0:
        axes[row, 2].set_title("Binarized XDoG", fontsize=10)
    axes[row, 2].set_ylabel(f"τ={tau:.2f}", fontsize=8, rotation=0, labelpad=40, va="center")

plt.tight_layout()
out = "outputs/viz_binarized_xdog.png"
import os; os.makedirs("outputs", exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
