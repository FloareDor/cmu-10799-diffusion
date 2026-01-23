"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self. beta_start = beta_start
        self. beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.register_buffer('betas', betas)
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        # TODO: Implement your own init

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(self, x_0: torch.Tensor, t:torch.Tensor, noise: Optional[torch.Tensor] = None): # TODO: Add your own arguments here
        # TODO: Implement the forward (noise adding) process of DDPM
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alpa_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        # x_t = signal_strength_at_t*x_0 + noise_strength_at_t*noise
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        batch_size = x_0.shape[0]
        # fill in the blanks
        t = torch.randint(
            low=0,
            high=self.num_timesteps, 
            size=(batch_size,), 
            device=self.device # generally good practice to create tensors directly on the GPU
        )
        noise = torch.randn_like(x_0)

        x_t, _ = self.forward_process(x_0, t, noise)
        noise_pred = self.model(x_t, t)

        loss = F.mse_loss(noise_pred, noise)

        return loss, {'mse': loss.item()}

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments

        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        # prev_image = 1/signal_strength_at_t (current_image - noise_strength_at_t/accumulated_noise * predicted_noise)
        epsilon = self.model(x_t, t)

        beta_t = self.betas[t][:, None, None, None]
        alpha_t = self.alphas[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        mean = 1/torch.sqrt(alpha_t) * (x_t - beta_t/sqrt_one_minus_alpha_cumprod_t * epsilon)

        # noise = step_wise_noise_strength * random_noise
        sigma_t = torch.sqrt(beta_t)
        z = torch.randn_like(x_t)
        noise = sigma_t * z

        nonzero_mask = (t>0).float()[:, None, None, None]
        x_prev = mean + nonzero_mask * noise # dont add noise to the first step cuz that is the final image output lol
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.model.eval()
        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        for timestep in range(self.num_timesteps-1, -1, -1):
            t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            x_t = self.reverse_process(x_t, t)
        return x_t

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )
