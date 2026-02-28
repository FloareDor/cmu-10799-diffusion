# Poster package asset builder (local)
import os
import re
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pkg = r"E:\cmu-e\courses\diffusion\homeworks\hw4\poster_package_2026-02-25"
raw = os.path.join(pkg, "raw_downloads", "modal_poster_results_dir", "poster")
figdir = os.path.join(pkg, "figures")
tabdir = os.path.join(pkg, "tables")
os.makedirs(figdir, exist_ok=True)
os.makedirs(tabdir, exist_ok=True)

kid_paths = {
    "Baseline": os.path.join(raw, "kid_baseline_1000_50steps_dataset.txt"),
    "HW4 Main": os.path.join(raw, "kid_hw4main_1000_50steps_dataset.txt"),
}
rows = []
for k,p in kid_paths.items():
    txt = open(p,'r',encoding='utf-8').read()
    mean = float(re.search(r"kernel_inception_distance_mean:\s*([0-9.]+)", txt).group(1))
    std = float(re.search(r"kernel_inception_distance_std:\s*([0-9.]+)", txt).group(1))
    rows.append((k, mean, std, p))

with open(os.path.join(tabdir, "kid_summary_modal_poster.csv"), "w", encoding="utf-8") as f:
    f.write("model,kid_mean,kid_std,source\\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]},\\\"{r[3]}\\\"\\n")


def make_two_panel(left_img_path, right_img_path, title, outpath, left_sub=None, right_sub=None):
    L = Image.open(left_img_path).convert("RGB")
    R = Image.open(right_img_path).convert("RGB")
    w = max(L.width, R.width)
    h = max(L.height, R.height)
    L = ImageOps.pad(L, (w,h), color=(255,255,255))
    R = ImageOps.pad(R, (w,h), color=(255,255,255))

    margin=24; gap=24; top=90
    canvas = Image.new("RGB", (w*2+gap+margin*2, h+top+margin*2), (247,247,245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((margin, margin), title, fill=(20,20,20), font=font)
    x1=margin; x2=margin+w+gap; y=margin+top
    draw.rectangle([x1, margin+22, x1+w, margin+top-6], fill=(230,236,248))
    draw.rectangle([x2, margin+22, x2+w, margin+top-6], fill=(227,240,229))
    draw.text((x1+10, margin+32), "Baseline", fill=(30,30,30), font=font)
    draw.text((x2+10, margin+32), "HW4 Main", fill=(30,30,30), font=font)
    if left_sub: draw.text((x1+10, margin+54), left_sub, fill=(50,50,50), font=font)
    if right_sub: draw.text((x2+10, margin+54), right_sub, fill=(50,50,50), font=font)

    canvas.paste(L, (x1,y)); canvas.paste(R, (x2,y))
    canvas.save(outpath)

kid_map = {r[0]: (r[1],r[2]) for r in rows}
make_two_panel(
    os.path.join(raw, "baseline_grid64_50steps_dataset.png"),
    os.path.join(raw, "hw4main_grid64_50steps_dataset.png"),
    "50-step Conditioned Sampling (64 images)",
    os.path.join(figdir, "fig_qualitative_50step_64_modal.png"),
    left_sub=f"KID {kid_map['Baseline'][0]:.4f} +/- {kid_map['Baseline'][1]:.4f}",
    right_sub=f"KID {kid_map['HW4 Main'][0]:.4f} +/- {kid_map['HW4 Main'][1]:.4f}",
)

make_two_panel(
    os.path.join(raw, "baseline_grid16_100steps_dataset.png"),
    os.path.join(raw, "hw4main_grid16_100steps_dataset.png"),
    "100-step Conditioned Sampling (16 images)",
    os.path.join(figdir, "fig_qualitative_100step_16_modal.png"),
)

labels=[r[0] for r in rows]
means=[r[1] for r in rows]
stds=[r[2] for r in rows]
plt.figure(figsize=(7.2,4.2))
plt.bar(labels,means,yerr=stds,capsize=4,color=["#4C78A8","#59A14F"])
plt.ylabel("KID (lower is better)")
plt.title("KID on 1000 conditioned samples (50 steps)")
plt.tight_layout()
plt.savefig(os.path.join(figdir, "fig_kid_bar_modal_poster.png"), dpi=220)
print("Poster assets rebuilt")
