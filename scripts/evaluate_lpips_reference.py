"""
Evaluate LPIPS to reference source images under edge conditioning.

This measures source-image perceptual fidelity (not controllability):
LPIPS(generated, reference_image_for_same_condition)
Lower is better.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import lpips
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.data import canny_edges, create_dataloader_from_config, xdog_edges
from src.methods import DDPM, FlowMatching
from src.models import create_model_from_config
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])

    ema = EMA(model, decay=config["training"]["ema_decay"])
    ema.load_state_dict(checkpoint["ema"])
    return model, config, ema


def _collect_dataset_pairs(
    config: dict,
    num_samples: int,
    split: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    condition_config = dict(config)
    condition_config["data"] = dict(config["data"])
    condition_config["training"] = dict(config["training"])
    condition_config["data"]["conditional"] = True
    condition_config["training"]["batch_size"] = batch_size

    dataloader = create_dataloader_from_config(condition_config, split=split)

    ref_batches: List[torch.Tensor] = []
    cond_batches: List[torch.Tensor] = []
    remaining = num_samples
    while remaining > 0:
        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise RuntimeError("Expected conditional dataset batch to return (images, edges).")
            images, edges = batch
            take = min(remaining, images.shape[0])
            ref_batches.append(images[:take].to(device))
            cond_batches.append(edges[:take].to(device))
            remaining -= take
            if remaining <= 0:
                break

    return torch.cat(ref_batches, dim=0), torch.cat(cond_batches, dim=0)


def _collect_dir_conditions(
    config: dict,
    edge_source: str,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    image_dir = Path(edge_source)
    if not image_dir.exists():
        raise FileNotFoundError(f"Edge source path does not exist: {edge_source}")

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in: {edge_source}")

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    tensors: List[torch.Tensor] = []
    target_size = (config["data"]["image_size"], config["data"]["image_size"])
    edge_method = config["data"].get("edge_method", "xdog")
    for p in image_files:
        img = Image.open(p).convert("RGB").resize(target_size, Image.BILINEAR)
        if edge_method == "canny":
            sketch = canny_edges(img)
        else:
            sketch = xdog_edges(img)
        tensors.append(to_tensor(sketch))
        if len(tensors) >= num_samples:
            break
    if len(tensors) < num_samples:
        reps = math.ceil(num_samples / len(tensors))
        tensors = (tensors * reps)[:num_samples]
    return torch.stack(tensors, dim=0).to(device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS-to-reference under edge conditioning.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["ddpm", "flow_matching"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--edge_source", type=str, default="dataset")
    parser.add_argument("--edge_split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_log_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, config, ema = load_checkpoint(args.checkpoint, device)
    method = DDPM.from_config(model, config, device) if args.method == "ddpm" else FlowMatching.from_config(model, config, device)
    ema.apply_shadow()
    method.eval_mode()

    refs_all = None
    if args.edge_source == "dataset":
        refs_all, cond_all = _collect_dataset_pairs(
            config=config,
            num_samples=args.num_samples,
            split=args.edge_split,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        cond_all = _collect_dir_conditions(
            config=config,
            edge_source=args.edge_source,
            num_samples=args.num_samples,
            device=device,
        )
        raise ValueError("LPIPS-to-reference currently requires edge_source='dataset'.")

    image_shape = (
        config["data"]["channels"],
        config["data"]["image_size"],
        config["data"]["image_size"],
    )
    num_steps = args.num_steps or config["sampling"]["num_steps"]
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    lpips_scores: List[float] = []

    with torch.no_grad():
        pbar = tqdm(range(0, args.num_samples, args.batch_size), desc="LPIPS-ref eval")
        for start in pbar:
            end = min(start + args.batch_size, args.num_samples)
            cond_batch = cond_all[start:end]
            ref_batch = refs_all[start:end]
            bsz = cond_batch.shape[0]

            samples = method.sample(
                batch_size=bsz,
                image_shape=image_shape,
                num_steps=num_steps,
                condition=cond_batch,
                method=args.sampler,
                eta=args.eta,
            )
            vals = lpips_fn(samples, ref_batch).view(bsz)
            lpips_scores.extend(vals.cpu().tolist())

    t = torch.tensor(lpips_scores, dtype=torch.float32)
    lpips_mean = float(t.mean().item())
    lpips_std = float(t.std(unbiased=False).item())

    result: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "method": args.method,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_steps": num_steps,
        "sampler": args.sampler,
        "eta": args.eta,
        "edge_source": args.edge_source,
        "edge_split": args.edge_split if args.edge_source == "dataset" else None,
        "lpips_reference_mean": lpips_mean,
        "lpips_reference_std": lpips_std,
    }

    print(json.dumps(result, indent=2))

    if args.save_log_path:
        save_path = Path(args.save_log_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved LPIPS-reference results to {save_path}")

    ema.restore()


if __name__ == "__main__":
    main()

