"""Compare Canny (multiple sigmas), XDoG, PiDiNet, and HED on 10 CelebA samples."""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from controlnet_aux import PidiNetDetector, HEDdetector

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

    print("Loading PiDiNet...")
    pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    print("PiDiNet loaded.")
    print("Loading HED...")
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    print("HED loaded.")

    indices = random.sample(range(len(ds)), 10)

    out_dir = REPO_ROOT / "outputs" / "full_edge_compare_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_i, idx in enumerate(indices):
        img = ds[idx]["image"].convert("RGB")
        arr = np.array(img)

        # Canny (our pick) and Canny s1.3 for reference
        canny_s10 = canny_edges(arr, sigma=1.0, low=50, high=150)
        canny_s13 = canny_edges(arr, sigma=1.3, low=50, high=150)

        # XDoG sigma sweep (all phi=10)
        xdog_s025 = xdog_edges(arr, sigma=0.25, phi=10.0)
        xdog_s0275 = xdog_edges(arr, sigma=0.275, phi=10.0)
        xdog_s0288 = xdog_edges(arr, sigma=0.288, phi=10.0)
        xdog_s03 = xdog_edges(arr, sigma=0.3, phi=10.0)

        # PiDiNet nofilter
        pidi_nofilter = pidinet(img, safe=False, apply_filter=False).convert("RGB")

        # HED
        hed_img = hed(img).convert("RGB")

        rows.append((img, canny_s10, canny_s13, xdog_s025, xdog_s0275, xdog_s0288, xdog_s03, pidi_nofilter, hed_img))
        print(f"[{row_i+1}/10] idx={idx}")

    # Column labels (minimal)
    labels = [
        "Orig", "Canny 1.0", "Canny 1.3",
        "XDoG .25", "XDoG .28", "XDoG .29", "XDoG .30",
        "PiDi nofilter",
        "HED",
    ]
    w, h = rows[0][0].size
    cols = 9
    header_h = 28
    grid_h = 10 * h
    total_h = header_h + grid_h
    grid = Image.new("RGB", (cols * w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        try:
            font = ImageFont.truetype("Segoe UI.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
    for c, label in enumerate(labels):
        # center text in column
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), label, font=font)
        else:
            bbox = font.getbbox(label) if hasattr(font, "getbbox") else (0, 0, *draw.textsize(label, font=font))
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = c * w + (w - tw) // 2
        y = (header_h - th) // 2
        draw.text((x, y), label, fill=(64, 64, 64), font=font)
    for r, row_imgs in enumerate(rows):
        for c, im in enumerate(row_imgs):
            if im.size != (w, h):
                im = im.resize((w, h), Image.LANCZOS)
            grid.paste(im, (c * w, header_h + r * h))

    grid_path = out_dir / "full_compare_grid.png"
    grid.save(grid_path)
    print(f"\nSaved: {grid_path}")
    print("Columns: Orig | Canny s1.0 | Canny s1.3 | XDoG s0.25 | XDoG s0.275 | XDoG s0.288 | XDoG s0.3 | PiDiNet nofilter | HED")


if __name__ == "__main__":
    main()
