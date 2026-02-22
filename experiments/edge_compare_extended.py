"""Compare Canny (sparse variants), XDoG, and Smart Canny on CelebA samples."""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "celeba-subset"


def canny_edges(image_array, sigma=1.0, low=50, high=150):
    """Standard Canny with optional Gaussian pre-blur."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def xdog_edges(image_array, sigma=0.5, k=1.6, gamma=0.98, epsilon=0.01, phi=10.0):
    """XDoG (Extended Difference of Gaussians) for sketch-style edges."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1 - gamma * g2
    dog = dog / (dog.max() + 1e-8)
    result = np.where(dog >= epsilon, np.ones_like(dog), 1.0 + np.tanh(phi * (dog - epsilon)))
    result = np.clip(result, 0, 1)
    inv = 1.0 - result
    return Image.fromarray((np.stack([inv] * 3, axis=-1) * 255).astype(np.uint8))


def smart_canny_edges(image_array, d=9, sigma_color=75, sigma_space=75, low=50, high=150):
    """Smart Canny: Bilateral filter (smooth textures, keep edges) + Canny.
    Produces clean, structural lines without skin/fabric texture noise.
    """
    smoothed = cv2.bilateralFilter(image_array, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=low, threshold2=high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def pencil_sketch_edges(image_array, blur_ksize=21):
    """Ora-lytics pencil sketch: dodge blend for tonal shading."""
    grey = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    inv = cv2.bitwise_not(grey)
    blur = cv2.GaussianBlur(inv, (blur_ksize, blur_ksize), 0)
    invblur = cv2.bitwise_not(blur)
    sketch = cv2.divide(grey, invblur, scale=256.0)
    return Image.fromarray(np.stack([sketch] * 3, axis=-1))


def cartoon_edges(image_array, line_size=7, blur_value=7, k=5):
    """Ora-lytics cartoon: K-means posterization + adaptive-threshold edges."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value
    )
    imgf = np.float32(image_array).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(imgf, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()].reshape(image_array.shape)
    cartoon = cv2.bitwise_and(final_img, final_img, mask=edges)
    return Image.fromarray(cartoon)


def scribble_xdog_edges(image_array, thr_a=32):
    """ComfyUI Scribble XDoG Lines: simpler DoG with threshold."""
    img = image_array.astype(np.float32)
    g1 = cv2.GaussianBlur(img, (0, 0), 0.5)
    g2 = cv2.GaussianBlur(img, (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(image_array[:, :, 0], dtype=np.uint8)
    result[2 * (255 - dog) > thr_a] = 255
    return Image.fromarray(np.stack([result] * 3, axis=-1))


def main():
    random.seed(999)
    np.random.seed(999)

    ds = load_dataset(
        "electronickale/cmu-10799-celeba64-subset",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    print(f"Loaded dataset: {len(ds)} images")

    n_samples = 10
    indices = random.sample(range(len(ds)), n_samples)

    out_dir = REPO_ROOT / "outputs" / "edge_compare_extended"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row_i, idx in enumerate(indices):
        img = ds[idx]["image"].convert("RGB")
        arr = np.array(img)

        # Canny: sigma 2.5 -> 0.8 (sparsest to finest); thresholds 80/200
        canny_s25 = canny_edges(arr, sigma=2.5, low=80, high=200)
        canny_s23 = canny_edges(arr, sigma=2.3, low=80, high=200)
        canny_s2 = canny_edges(arr, sigma=2.0, low=80, high=200)
        canny_s175 = canny_edges(arr, sigma=1.75, low=80, high=200)
        canny_s15 = canny_edges(arr, sigma=1.5, low=80, high=200)
        canny_s12 = canny_edges(arr, sigma=1.2, low=80, high=200)
        canny_s10 = canny_edges(arr, sigma=1.0, low=80, high=200)
        canny_s08 = canny_edges(arr, sigma=0.8, low=80, high=200)
        # XDoG: sigma sweep
        xdog_022 = xdog_edges(arr, sigma=0.22, phi=10.0)
        xdog_025 = xdog_edges(arr, sigma=0.25, phi=10.0)
        xdog_028 = xdog_edges(arr, sigma=0.28, phi=10.0)
        xdog_030 = xdog_edges(arr, sigma=0.30, phi=10.0)
        # Scribble XDoG (ComfyUI-style): thr 24, 32, 40
        scribble_24 = scribble_xdog_edges(arr, thr_a=24)
        scribble_32 = scribble_xdog_edges(arr, thr_a=32)
        scribble_40 = scribble_xdog_edges(arr, thr_a=40)
        # Ora-lytics Pencil (blur 21) and Cartoon
        pencil = pencil_sketch_edges(arr, blur_ksize=21)
        cartoon = cartoon_edges(arr, line_size=7, blur_value=7, k=5)

        rows.append((
            img, canny_s25, canny_s23, canny_s2, canny_s175, canny_s15, canny_s12, canny_s10, canny_s08,
            xdog_022, xdog_025, xdog_028, xdog_030,
            scribble_24, scribble_32, scribble_40,
            pencil, cartoon,
        ))
        print(f"[{row_i+1}/{n_samples}] idx={idx}")

    labels = [
        "Orig",
        "Canny 2.5", "Canny 2.3", "Canny 2", "Canny 1.75", "Canny 1.5", "Canny 1.2", "Canny 1.0", "Canny 0.8",
        "XDoG .22", "XDoG .25", "XDoG .28", "XDoG .30",
        "ScribbleXDoG 24", "ScribbleXDoG 32", "ScribbleXDoG 40",
        "Pencil", "Cartoon",
    ]
    w, h = rows[0][0].size
    cols = 18
    header_h = 28
    n_rows = len(rows)
    grid_h = n_rows * h
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

    grid_path = out_dir / "edge_compare_extended_grid.png"
    grid.save(grid_path)
    print(f"\nSaved: {grid_path}")
    print("Columns: Orig | Canny 2.5->0.8 | XDoG .22-.30 | ScribbleXDoG 24/32/40 | Pencil | Cartoon")


if __name__ == "__main__":
    main()
