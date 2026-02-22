"""
Generate 10 conditioned samples and save as a grid exactly like the training run.

Format: edge | generated | ground_truth (same as save_conditional_triplet_grid in train.py)

Usage:
    python experiments/hw4_sample_grid_10.py --checkpoint PATH --config configs/hw4_edge_adapter_mixed_eagf_modal_2gpu.yaml

    # With local data root (default uses ./data/celeba or HuggingFace cache):
    python experiments/hw4_sample_grid_10.py --checkpoint PATH --config configs/hw4_edge_adapter_mixed_eagf_modal_2gpu.yaml --data-root ./data/celeba
"""
import argparse
import math
import os
import sys
from pathlib import Path

import yaml
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import create_dataloader_from_config, unnormalize, save_image
from src.models import create_model_from_config
from src.methods import FlowMatching
from src.utils import EMA


def save_conditional_triplet_grid(condition, generated, target, save_path, nrow=None):
    """Save per-sample side-by-side panels: edge | generated | ground truth (same as train.py)."""
    panel = torch.cat([
        unnormalize(condition),
        unnormalize(generated),
        unnormalize(target),
    ], dim=3)
    if nrow is None:
        nrow = int(math.sqrt(panel.shape[0]))
    save_image(panel, save_path, nrow=nrow)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 10 samples with edge conditioning, grid format: edge | generated | target"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt file). Omit with --preview.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hw4_edge_adapter_mixed_eagf_modal_2gpu.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root for local runs (default: from config or ./data/celeba)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: outputs/hw4_sample_grid_10/hw4_sample_grid_10.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Skip generation: save condition | target | target grid (no checkpoint needed)",
    )
    args = parser.parse_args()

    if not args.preview and (args.checkpoint is None or not os.path.exists(args.checkpoint)):
        print("Checkpoint required for generation.")
        if args.checkpoint:
            print(f"  Not found: {args.checkpoint}")
        print("  Use --preview to see condition|target grid without a model.")
        print("  Or download a checkpoint: modal volume get cmu-10799-diffusion-data checkpoints/hw4_edge_adapter_mixed_eagf_modal_2gpu/hw4_edge_adapter_mixed_eagf_modal_2gpu_0020000.pt ./checkpoints/")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    elif config["data"].get("root") == "/data/celeba":
        config["data"]["root"] = str(REPO_ROOT / "data" / "celeba")

    n = args.num_samples

    print("Creating dataloader...")
    dataloader = create_dataloader_from_config(
        config, split="train"
    )
    batch = next(iter(dataloader))
    images, condition = batch
    images = images[:n].to(device)
    condition = condition[:n].to(device)

    if args.preview:
        print("Preview mode: saving condition | target | target (no generation)")
        out_path = args.output or str(REPO_ROOT / "outputs" / "hw4_sample_grid_10" / "hw4_sample_grid_10_preview.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_conditional_triplet_grid(
            condition=condition,
            generated=images,  # use target as placeholder for "generated"
            target=images,
            save_path=out_path,
            nrow=int(math.sqrt(n)),
        )
        print(f"Saved preview grid to {out_path}")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config_ckpt = ckpt["config"]
    model = create_model_from_config(config_ckpt).to(device)
    model.load_state_dict(ckpt["model"])
    ema = EMA(model, decay=config_ckpt["training"]["ema_decay"])
    ema.load_state_dict(ckpt["ema"])
    ema.apply_shadow()

    method = FlowMatching.from_config(model, config_ckpt, device)
    method.eval_mode()

    image_shape = (
        config_ckpt["data"]["channels"],
        config_ckpt["data"]["image_size"],
        config_ckpt["data"]["image_size"],
    )
    num_steps = config_ckpt.get("sampling", {}).get("num_steps", 50)

    print(f"Generating {n} samples (num_steps={num_steps})...")
    with torch.no_grad():
        generated = method.sample(
            batch_size=n,
            image_shape=image_shape,
            num_steps=num_steps,
            condition=condition,
        )

    ema.restore()

    out_path = args.output
    if out_path is None:
        out_dir = REPO_ROOT / "outputs" / "hw4_sample_grid_10"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "hw4_sample_grid_10.png")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_conditional_triplet_grid(
        condition=condition,
        generated=generated,
        target=images,
        save_path=out_path,
        nrow=int(math.sqrt(n)),
    )
    print(f"Saved grid to {out_path}")


if __name__ == "__main__":
    main()
