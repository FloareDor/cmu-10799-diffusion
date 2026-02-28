"""
Evaluate conditional controllability via edge adherence.

Metric:
- EAS IoU: IoU between generated-image edges and conditioning edge map.
- Also reports precision, recall, and F1 for edge overlap.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.data import canny_edges, create_dataloader_from_config, unnormalize, xdog_edges
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


def _as_binary_edge_mask(
    x: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """x: [B, 3, H, W] in [-1, 1], returns [B, 1, H, W] bool."""
    x01 = unnormalize(x).clamp(0.0, 1.0)
    gray = x01.mean(dim=1, keepdim=True)
    return gray >= threshold


def _dilate_binary(mask: torch.Tensor, tolerance_px: int) -> torch.Tensor:
    if tolerance_px <= 0:
        return mask
    k = 2 * tolerance_px + 1
    out = F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=tolerance_px)
    return out > 0.0


def _extract_edges_from_generated(
    samples: torch.Tensor,
    edge_method: str,
    canny_sigma: float,
    canny_low: int,
    canny_high: int,
    xdog_sigma: float,
) -> torch.Tensor:
    """
    samples: [B, 3, H, W] in [-1, 1]
    returns: [B, 3, H, W] in [-1, 1] edge images
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    out_tensors: List[torch.Tensor] = []
    samples01 = unnormalize(samples).clamp(0.0, 1.0).cpu()

    for i in range(samples01.shape[0]):
        pil_img = to_pil(samples01[i])
        if edge_method == "canny":
            edge_pil = canny_edges(
                pil_img,
                sigma=canny_sigma,
                low=canny_low,
                high=canny_high,
            )
        elif edge_method == "xdog":
            edge_pil = xdog_edges(pil_img, sigma=xdog_sigma)
        else:
            raise ValueError(f"Unsupported eval edge method: {edge_method}")
        out_tensors.append(to_tensor(edge_pil))
    return torch.stack(out_tensors, dim=0).to(samples.device)


def _collect_dataset_conditions(
    config: dict,
    num_samples: int,
    split: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    condition_config = dict(config)
    condition_config["data"] = dict(config["data"])
    condition_config["training"] = dict(config["training"])
    condition_config["data"]["conditional"] = True
    condition_config["training"]["batch_size"] = batch_size

    dataloader = create_dataloader_from_config(condition_config, split=split)

    condition_batches: List[torch.Tensor] = []
    remaining = num_samples
    while remaining > 0:
        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise RuntimeError("Expected conditional dataset batch to return (images, edges).")
            edges = batch[1]
            take = min(remaining, edges.shape[0])
            condition_batches.append(edges[:take].to(device))
            remaining -= take
            if remaining <= 0:
                break

    return torch.cat(condition_batches, dim=0)


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
    parser = argparse.ArgumentParser(description="Evaluate controllability via edge adherence.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["ddpm", "flow_matching"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--edge_source", type=str, default="dataset")
    parser.add_argument("--edge_split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--eval_edge_method", type=str, default="canny", choices=["canny", "xdog"])
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--tolerance_px", type=int, default=1)
    parser.add_argument("--canny_sigma", type=float, default=1.2)
    parser.add_argument("--canny_low", type=int, default=80)
    parser.add_argument("--canny_high", type=int, default=200)
    parser.add_argument("--xdog_sigma", type=float, default=0.29)
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

    if args.edge_source == "dataset":
        condition_all = _collect_dataset_conditions(
            config=config,
            num_samples=args.num_samples,
            split=args.edge_split,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        condition_all = _collect_dir_conditions(
            config=config,
            edge_source=args.edge_source,
            num_samples=args.num_samples,
            device=device,
        )

    image_shape = (
        config["data"]["channels"],
        config["data"]["image_size"],
        config["data"]["image_size"],
    )
    num_steps = args.num_steps or config["sampling"]["num_steps"]

    iou_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []

    with torch.no_grad():
        pbar = tqdm(range(0, args.num_samples, args.batch_size), desc="Controllability eval")
        for start in pbar:
            end = min(start + args.batch_size, args.num_samples)
            condition_batch = condition_all[start:end]
            batch_size = condition_batch.shape[0]

            samples = method.sample(
                batch_size=batch_size,
                image_shape=image_shape,
                num_steps=num_steps,
                condition=condition_batch,
                method=args.sampler,
                eta=args.eta,
            )

            generated_edges = _extract_edges_from_generated(
                samples=samples,
                edge_method=args.eval_edge_method,
                canny_sigma=args.canny_sigma,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                xdog_sigma=args.xdog_sigma,
            )

            pred = _as_binary_edge_mask(generated_edges, threshold=args.mask_threshold)
            target = _as_binary_edge_mask(condition_batch, threshold=args.mask_threshold)

            pred_d = _dilate_binary(pred, args.tolerance_px)
            target_d = _dilate_binary(target, args.tolerance_px)

            tp_precision = (pred & target_d).flatten(1).sum(dim=1).float()
            tp_recall = (target & pred_d).flatten(1).sum(dim=1).float()
            pred_pos = pred.flatten(1).sum(dim=1).float()
            tgt_pos = target.flatten(1).sum(dim=1).float()
            inter_iou = (pred_d & target_d).flatten(1).sum(dim=1).float()
            union = (pred_d | target_d).flatten(1).sum(dim=1).float()

            eps = 1e-8
            precision = tp_precision / (pred_pos + eps)
            recall = tp_recall / (tgt_pos + eps)
            f1 = 2.0 * precision * recall / (precision + recall + eps)
            iou = inter_iou / (union + eps)

            iou_scores.extend(iou.cpu().tolist())
            precision_scores.extend(precision.cpu().tolist())
            recall_scores.extend(recall.cpu().tolist())
            f1_scores.extend(f1.cpu().tolist())

    def _mean_std(xs: List[float]) -> Tuple[float, float]:
        t = torch.tensor(xs, dtype=torch.float32)
        return float(t.mean().item()), float(t.std(unbiased=False).item())

    iou_mean, iou_std = _mean_std(iou_scores)
    precision_mean, precision_std = _mean_std(precision_scores)
    recall_mean, recall_std = _mean_std(recall_scores)
    f1_mean, f1_std = _mean_std(f1_scores)

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
        "eval_edge_method": args.eval_edge_method,
        "mask_threshold": args.mask_threshold,
        "tolerance_px": args.tolerance_px,
        "edge_adherence_iou_mean": iou_mean,
        "edge_adherence_iou_std": iou_std,
        "edge_precision_mean": precision_mean,
        "edge_precision_std": precision_std,
        "edge_recall_mean": recall_mean,
        "edge_recall_std": recall_std,
        "edge_f1_mean": f1_mean,
        "edge_f1_std": f1_std,
    }

    print(json.dumps(result, indent=2))

    if args.save_log_path:
        save_path = Path(args.save_log_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved controllability results to {save_path}")

    ema.restore()


if __name__ == "__main__":
    main()
