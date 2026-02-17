"""
Sampling Script for DDPM (Denoising Diffusion Probabilistic Models)

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.

Usage:
    # Sample from DDPM (saves individual images to ./samples/)
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64

    # With custom number of sampling steps
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_steps 500

    # Generate a grid image instead of individual images
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64 --grid

    # Save individual images to custom directory
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --output_dir my_samples

What you need to implement:
- Incorporate your sampling scheme to this pipeline
- Save generated samples as images for logging
"""

import os
import sys
import argparse
import math
from datetime import datetime
from pathlib import Path

import yaml
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from src.models import create_model_from_config
from src.data import save_image, unnormalize, create_dataloader_from_config, xdog_edges
from src.methods import DDPM, FlowMatching
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and return model, config, and EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint['model'])
    
    # Create EMA and load
    ema = EMA(model, decay=config['training']['ema_decay'])
    ema.load_state_dict(checkpoint['ema'])
    
    return model, config, ema


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 8,
) -> None:
    """
    TODO: save generated samples as images.

    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        nrow: Number of images per row in the grid.
    """
    # 1. Unnormalize samples from [-1, 1] to [0, 1]
    samples = unnormalize(samples)

    # 2. Save using the imported save_image utility
    save_image(samples, save_path, nrow=nrow)


def _repeat_to_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    repeats = (batch_size + tensor.shape[0] - 1) // tensor.shape[0]
    repeat_dims = [repeats] + [1] * (tensor.dim() - 1)
    return tensor.repeat(*repeat_dims)[:batch_size]


def _load_condition_from_source(
    config: dict,
    edge_source: str,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    if edge_source == "dataset":
        condition_config = dict(config)
        condition_config['data'] = dict(config['data'])
        condition_config['data']['conditional'] = True
        dataloader = create_dataloader_from_config(condition_config, split='train')
        batch = next(iter(dataloader))
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise RuntimeError("Expected conditional dataset batch to return (images, edges).")
        edges = batch[1]
        return _repeat_to_batch(edges, num_samples).to(device)

    image_dir = Path(edge_source)
    if not image_dir.exists():
        raise FileNotFoundError(f"Edge source path does not exist: {edge_source}")
    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
    )
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in: {edge_source}")
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensors = []
    target_size = (config['data']['image_size'], config['data']['image_size'])
    for p in image_files:
        img = Image.open(p).convert("RGB").resize(target_size, Image.BILINEAR)
        sketch = xdog_edges(img)
        tensors.append(to_tensor(sketch))
        if len(tensors) >= num_samples:
            break
    if len(tensors) < num_samples:
        stacked = torch.stack(tensors, dim=0)
        return _repeat_to_batch(stacked, num_samples).to(device)
    return torch.stack(tensors, dim=0).to(device)


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'flow_matching'],
                       help='Method used for training (ddpm or flow_matching)')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save individual images (default: samples)')
    parser.add_argument('--grid', action='store_true',
                       help='Save as grid image instead of individual images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid (only used with --grid, default: samples_<timestamp>.png)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--edge_source', type=str, default=None,
                       help='Condition source: "dataset" or directory path containing source images')
    
    # Sampling arguments
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: from config)')
    parser.add_argument('--sampler', type=str, default='ddpm',
                       choices=['ddpm', 'ddim'],
                       help='Sampler to use (only for ddpm method)')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='Eta parameter for DDIM (default: 0.0)')
    
    # Other options
    parser.add_argument('--no_ema', action='store_true',
                       help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)
    
    # Create method
    if args.method == 'ddpm':
        method = DDPM.from_config(model, config, device)
    elif args.method == 'flow_matching':
        method = FlowMatching.from_config(model, config, device)
    else:
        raise ValueError(f"Unknown method: {args.method}. Only 'ddpm' and 'flow_matching' are currently supported.")
    
    # Apply EMA weights
    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA)")
    
    method.eval_mode()
    
    # Image shape
    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    condition_all = None
    if args.edge_source is not None:
        condition_all = _load_condition_from_source(config, args.edge_source, args.num_samples, device)
        print(f"Using sketch conditioning from: {args.edge_source}")

    all_samples = []
    all_conditions = []
    remaining = args.num_samples
    sample_idx = 0
    generated_so_far = 0

    # Create output directory if saving individual images
    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)

            num_steps = args.num_steps or config['sampling']['num_steps']
            condition_batch = None
            if condition_all is not None:
                condition_batch = condition_all[generated_so_far:generated_so_far + batch_size]

            samples = method.sample(
                batch_size=batch_size,
                image_shape=image_shape,
                num_steps=num_steps,
                condition=condition_batch,
                method=args.sampler,
                eta=args.eta,
            )

            # Save individual images immediately or collect for grid
            if args.grid:
                all_samples.append(samples)
                if condition_batch is not None:
                    all_conditions.append(condition_batch)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    if condition_batch is not None:
                        panel = torch.cat([condition_batch[i:i+1], samples[i:i+1]], dim=3)
                        save_samples(panel, img_path, 1)
                    else:
                        save_samples(samples[i:i+1], img_path, 1)
                    sample_idx += 1

            remaining -= batch_size
            generated_so_far += batch_size
            pbar.update(batch_size)

        pbar.close()

    # Save samples
    if args.grid:
        # Concatenate all samples for grid
        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]
        if condition_all is not None:
            all_conditions = torch.cat(all_conditions, dim=0)[:args.num_samples]

        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"

        nrow = int(math.sqrt(all_samples.shape[0]))
        if condition_all is not None:
            panel = torch.cat([all_conditions, all_samples], dim=3)
            save_samples(panel, args.output, nrow=nrow)
        else:
            save_samples(all_samples, args.output, nrow=nrow)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    # Restore EMA if applied
    if not args.no_ema:
        ema.restore()


if __name__ == '__main__':
    main()
