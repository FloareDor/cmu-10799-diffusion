"""
Generate a paired triplet strip using the SAME edge conditions for two checkpoints.

Layout: rows = examples, columns = [edge | model_a | model_b]
"""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.data import create_dataloader_from_config, unnormalize
from src.methods import DDPM, FlowMatching
from src.models import create_model_from_config
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = create_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model"])
    ema = EMA(model, decay=config["training"]["ema_decay"])
    ema.load_state_dict(ckpt["ema"])
    return model, config, ema


def build_method(model, config, device, method_name: str):
    if method_name == "ddpm":
        return DDPM.from_config(model, config, device)
    if method_name == "flow_matching":
        return FlowMatching.from_config(model, config, device)
    raise ValueError(f"Unsupported method: {method_name}")


def collect_conditions(config: dict, num_pairs: int, split: str, device: torch.device) -> torch.Tensor:
    cond_cfg = deepcopy(config)
    cond_cfg["data"] = deepcopy(config["data"])
    cond_cfg["training"] = deepcopy(config["training"])
    cond_cfg["data"]["conditional"] = True
    cond_cfg["training"]["batch_size"] = max(8, num_pairs)

    loader = create_dataloader_from_config(cond_cfg, split=split)

    chunks = []
    remaining = num_pairs
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise RuntimeError("Expected conditional dataset to return (images, edges).")
        _, edges = batch
        take = min(edges.shape[0], remaining)
        chunks.append(edges[:take].to(device))
        remaining -= take
        if remaining == 0:
            break
    if remaining > 0:
        raise RuntimeError(f"Could not collect requested {num_pairs} conditions.")
    return torch.cat(chunks, dim=0)


def sample_with_conditions(
    method,
    image_shape,
    conditions: torch.Tensor,
    num_steps: int,
    sampler: str,
    eta: float,
    batch_size: int,
) -> torch.Tensor:
    out = []
    total = conditions.shape[0]
    idx = 0
    with torch.no_grad():
        while idx < total:
            c = conditions[idx:idx + batch_size]
            x = method.sample(
                batch_size=c.shape[0],
                image_shape=image_shape,
                num_steps=num_steps,
                condition=c,
                method=sampler,
                eta=eta,
            )
            out.append(x)
            idx += c.shape[0]
    return torch.cat(out, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate paired edge|A|B triplet strip.")
    parser.add_argument("--baseline_checkpoint", type=str, required=True)
    parser.add_argument("--hw4_checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, default="flow_matching", choices=["ddpm", "flow_matching"])
    parser.add_argument("--num_pairs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--dataset_split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition_source", type=str, default="baseline", choices=["baseline", "hw4"])
    parser.add_argument("--layout", type=str, default="triplet_rows", choices=["triplet_rows", "one_row"])
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load both checkpoints
    base_model, base_cfg, base_ema = load_checkpoint(args.baseline_checkpoint, device)
    hw_model, hw_cfg, hw_ema = load_checkpoint(args.hw4_checkpoint, device)

    base_method = build_method(base_model, base_cfg, device, args.method)
    hw_method = build_method(hw_model, hw_cfg, device, args.method)

    base_ema.apply_shadow()
    hw_ema.apply_shadow()
    base_method.eval_mode()
    hw_method.eval_mode()

    # Source fixed edge conditions from selected config.
    cond_cfg = base_cfg if args.condition_source == "baseline" else hw_cfg
    conditions = collect_conditions(cond_cfg, args.num_pairs, args.dataset_split, device)

    base_shape = (
        base_cfg["data"]["channels"],
        base_cfg["data"]["image_size"],
        base_cfg["data"]["image_size"],
    )
    hw_shape = (
        hw_cfg["data"]["channels"],
        hw_cfg["data"]["image_size"],
        hw_cfg["data"]["image_size"],
    )

    base_samples = sample_with_conditions(
        base_method, base_shape, conditions, args.num_steps, args.sampler, args.eta, args.batch_size
    )
    hw_samples = sample_with_conditions(
        hw_method, hw_shape, conditions, args.num_steps, args.sampler, args.eta, args.batch_size
    )

    # Build [edge | baseline | hw4] row per example.
    edge_vis = unnormalize(conditions.detach().cpu())
    base_vis = unnormalize(base_samples.detach().cpu())
    hw_vis = unnormalize(hw_samples.detach().cpu())

    triplets = [torch.cat([edge_vis[i], base_vis[i], hw_vis[i]], dim=2) for i in range(args.num_pairs)]  # C,H,3W
    if args.layout == "one_row":
        grid = torch.cat(triplets, dim=2)  # C, H, (num_pairs*3W)
    else:
        grid = torch.cat(triplets, dim=1)  # C, (num_pairs*H), 3W

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))

    print(
        f"Saved paired triplet strip: {out_path}\n"
        f"num_pairs={args.num_pairs}, steps={args.num_steps}, sampler={args.sampler}, "
        f"condition_source={args.condition_source}, layout={args.layout}"
    )


if __name__ == "__main__":
    main()
