"""
Dataset Exploration Script

Converted from 02_dataset_exploration.ipynb
Explores the custom CelebA subset used for training.
"""

import sys
import os

# Fix Unicode encoding issues on Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Add parent directory to path for src module
# Use __file__ if available (normal execution), otherwise use hard-coded paths
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
except NameError:
    # Fallback for exec() or interactive use
    parent_dir = r'E:\cmu-e\courses\diffusion\homeworks\cmu-10799-diffusion'
    script_dir = os.path.join(parent_dir, 'notebooks')
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Set style
plt.style.use('default')

from src.data import create_dataloader, unnormalize

def main():
    # Use absolute paths
    data_root = os.path.join(parent_dir, 'data', 'celeba-subset')
    output_path = os.path.join(script_dir, 'training_samples_grid.png')

    # Load directly from HuggingFace Hub
    dataloader = create_dataloader(
        root=data_root,
        split='train',
        image_size=64,
        batch_size=64,
        num_workers=0,
        augment=False,  # Disable augmentation for exploration
        from_hub=True,
        repo_name='electronickale/cmu-10799-celeba64-subset',
    )

    print(f"Dataset size: {len(dataloader.dataset):,} images")
    print(f"Batches: {len(dataloader):,}")

    # Get a batch of images
    batch = next(iter(dataloader))

    # Unnormalize images from [-1, 1] to [0, 1] for display
    images = unnormalize(batch)

    # Create a grid of images (8x8 = 64 images)
    grid = make_grid(images, nrow=8, padding=2, normalize=False)

    # Convert to numpy and display
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = np.clip(grid_np, 0, 1)  # Ensure values are in [0, 1]

    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title('Q1 (a): Grid of Training Samples', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved grid to {output_path}")

if __name__ == "__main__":
    main()
