
import os
import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image as PILImage
import numpy as np

from src.models import UNet, AudioEncoder
from src.data.voxceleb import VoxCelebDataset, unnormalize
from src.methods import FlowMatching
from src.utils import EMA
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
import torchvision

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_sample_grid(clean, masked, generated, path):
    # Create a grid: [Clean, Masked, Generated]
    # clean, masked, generated are (B, C, H, W) tensors
    
    n = min(clean.shape[0], 4) # Max 4 samples
    
    # Unnormalize
    clean = unnormalize(clean[:n]).cpu()
    masked = unnormalize(masked[:n]).cpu()
    generated = unnormalize(generated[:n]).cpu()
    
    grid = []
    for i in range(n):
        grid.append(clean[i])
        grid.append(masked[i])
        grid.append(generated[i])
        
    # Grid is list of 3*N tensors
    grid_tensor = torchvision.utils.make_grid(grid, nrow=3) 
    # nrow=3 means: Clean, Masked, Gen (row 1)
    #               Clean, Masked, Gen (row 2)...
    
    torchvision.utils.save_image(grid_tensor, path)
    return grid_tensor

def train(config_path):
    config = load_config(config_path)
    device = torch.device(config['infrastructure']['device'])
    
    # 1. Dataset
    print("Loading dataset...")
    dataset = VoxCelebDataset(
        root=config['data']['root'],
        image_size=config['data']['image_size']
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'],
        drop_last=True
    )
    
    # 2. Models
    print("Initializing models...")
    # UNet takes 6 channels (3 noisy + 3 masked)
    model = UNet(
        in_channels=3 + 3, # Noisy + Masked
        out_channels=3,    # Predict velocity for 3 channels
        base_channels=config['model']['base_channels'],
        channel_mult=config['model']['channel_mult'],
        num_res_blocks=config['model']['num_res_blocks'],
        attention_resolutions=config['model']['attention_resolutions'],
        audio_cond_dim=768 # Wav2Vec2 base dim
    ).to(device)
    
    audio_encoder = AudioEncoder().to(device)
    audio_encoder.eval() # Always frozen
    
    # 3. Method
    method = FlowMatching(model, device=device)
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scaler = GradScaler()
    
    # 5. Logging & Checkpoints
    os.makedirs("results/lipsync", exist_ok=True)
    os.makedirs("checkpoints/lipsync", exist_ok=True)
    
    # Loop
    print("Starting training...")
    method.train_mode()
    
    iter_loader = iter(dataloader)
    for step in range(config['training']['num_iterations']):
        try:
            batch_data = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch_data = next(iter_loader)
            
        # Unpack
        # dataset returns: image, {'image': masked, 'audio': audio}
        gt_images, cond = batch_data
        gt_images = gt_images.to(device)
        masked_images = cond['image'].to(device)
        audio_raw = cond['audio'].to(device)
        
        # Squeeze audio if needed (B, 1, L) -> (B, L)
        if audio_raw.dim() == 3 and audio_raw.shape[1] == 1:
            audio_raw = audio_raw.squeeze(1)
            
        with torch.no_grad():
            audio_emb = audio_encoder(audio_raw)
            
        # Conditioning for Flow Matching
        # passed as kwargs to model(x, t, **kwargs)
        # UNet expects 'image' and 'audio'
        conditioning = {
            'image': masked_images,
            'audio': audio_emb
        }
        
        optimizer.zero_grad()
        
        with autocast():
            loss, metrics = method.compute_loss(gt_images, condition=conditioning)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            
        if step % config['training']['sample_every'] == 0:
            print(f"Sampling at step {step}...")
            # Pick first sample from batch as reference
            sample_cond = {
                'image': masked_images[:4],
                'audio': audio_emb[:4]
            }
            
            samples = method.sample(
                batch_size=4,
                image_shape=(3, 64, 64),
                condition=sample_cond,
                num_steps=config['sampling']['num_steps']
            )
            
            save_path = f"results/lipsync/sample_{step:05d}.png"
            save_sample_grid(gt_images[:4], masked_images[:4], samples, save_path)
            print(f"Saved sample to {save_path}")
            
            method.train_mode()

        # Save checkpoint periodically
        if (step + 1) % config['training']['save_every'] == 0 or step == config['training']['num_iterations'] - 1:
            ckpt_path = f"checkpoints/lipsync/lipsync_step{step+1}.pt"
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lipsync.yaml')
    args = parser.parse_args()
    
    train(args.config)
