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
        prediction_type: str = "epsilon",
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_type = prediction_type
        
        # Validate prediction_type
        valid_types = ["epsilon", "x0", "v"]
        if prediction_type not in valid_types:
            raise ValueError(f"prediction_type must be one of {valid_types}, got {prediction_type}")

        # Compute all values as local variables first
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        
        # Register ALL of them as buffers
        # Use persistent=False if you don't want to save them in the checkpoint 
        # (since they can be recalculated), but standard practice is often just to register them.
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)

        # Move all buffers to the correct device
        self.to(device)

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    
    # =========================================================================
    # Prediction type conversion helpers
    # =========================================================================
    
    def _predict_x0_from_epsilon(self, x_t: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """Convert epsilon prediction to x0 prediction."""
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        return (x_t - sqrt_one_minus_alpha * epsilon) / sqrt_alpha
    
    def _predict_epsilon_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Convert x0 prediction to epsilon prediction."""
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        return (x_t - sqrt_alpha * x0) / sqrt_one_minus_alpha
    
    def _predict_epsilon_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Convert v-prediction to epsilon prediction."""
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        return sqrt_alpha * v + sqrt_one_minus_alpha * x_t
    
    def _predict_v_from_epsilon(self, x_t: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """Convert epsilon prediction to v-prediction."""
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        x0 = self._predict_x0_from_epsilon(x_t, t, epsilon)
        return sqrt_alpha * epsilon - sqrt_one_minus_alpha * x0
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(self, x_0: torch.Tensor, t:torch.Tensor, noise: Optional[torch.Tensor] = None): # TODO: Add your own arguments here
        # TODO: Implement the forward (noise adding) process of DDPM
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        # x_t = signal_strength_at_t*x_0 + noise_strength_at_t*noise
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Implement DDPM loss function with support for different prediction types.

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        batch_size = x_0.shape[0]
        t = torch.randint(
            low=0,
            high=self.num_timesteps, 
            size=(batch_size,), 
            device=self.device
        )
        noise = torch.randn_like(x_0)

        x_t, _ = self.forward_process(x_0, t, noise)
        model_output = self.model(x_t, t)

        # Compute target based on prediction_type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x_0
        elif self.prediction_type == "v":
            # v = sqrt(alpha_bar) * epsilon - sqrt(1 - alpha_bar) * x0
            sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None, None]
            sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        loss = F.mse_loss(model_output, target)

        return loss, {'loss': loss.item(), 'mse': loss.item()}

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Implement one step of the DDPM reverse process with support for different prediction types.

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments

        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        # Get model output and convert to epsilon
        model_output = self.model(x_t, t)
        
        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "x0":
            epsilon = self._predict_epsilon_from_x0(x_t, t, model_output)
        elif self.prediction_type == "v":
            epsilon = self._predict_epsilon_from_v(x_t, t, model_output)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        beta_t = self.betas[t][:, None, None, None]
        alpha_t = self.alphas[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]

        mean = 1/torch.sqrt(alpha_t) * (x_t - beta_t/sqrt_one_minus_alpha_cumprod_t * epsilon)

        # noise = step_wise_noise_strength * random_noise
        sigma_t = torch.sqrt(beta_t)
        z = torch.randn_like(x_t)
        noise = sigma_t * z

        nonzero_mask = (t>0).float()[:, None, None, None]
        x_prev = mean + nonzero_mask * noise
        return x_prev

    @torch.no_grad()
    def ddim_reverse_process(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        t_prev: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Implement one step of the DDIM reverse process.
        
        Args:
            x_t: Noisy samples at time t
            t: Current timestep index
            t_prev: Previous timestep index (can be t-1 or earlier)
            eta: Coefficient for noise (0.0 for deterministic DDIM)
        """
        model_output = self.model(x_t, t)
        
        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "x0":
            epsilon = self._predict_epsilon_from_x0(x_t, t, model_output)
        elif self.prediction_type == "v":
            epsilon = self._predict_epsilon_from_v(x_t, t, model_output)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        alpha_bar_t = self.alpha_cumprod[t][:, None, None, None]
        alpha_bar_t_prev = self.alpha_cumprod[t_prev][:, None, None, None] if t_prev >= 0 else torch.ones_like(alpha_bar_t)

        # 1. Estimate x0
        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon) / torch.sqrt(alpha_bar_t)
        
        # 2. Compute sigma_t (noise level)
        # sigma_t = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_prev)
        sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
        
        # 3. Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * epsilon
        
        # 4. Random noise
        noise = sigma_t * torch.randn_like(x_t) if eta > 0 else 0
        
        x_prev = torch.sqrt(alpha_bar_t_prev) * x0_hat + dir_xt + noise
        
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        method: Literal["ddpm", "ddim"] = "ddpm",
        eta: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Implement DDPM/DDIM sampling loop.

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of sampling steps.
            method: Sampling method ('ddpm' or 'ddim')
            eta: Noise coefficient for DDIM (default 0.0)
            **kwargs: Additional method-specific arguments
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.model.eval()
        
        # Determine timesteps to use
        if num_steps is None or num_steps == self.num_timesteps:
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
        else:
            # Use uniform stride to skip steps
            # Example: 1000 steps, num_steps=100 => steps 999, 989, ..., 9
            indices = torch.linspace(self.num_timesteps - 1, 0, num_steps).long().tolist()
            timesteps = indices
        
        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        
        for i, timestep in enumerate(timesteps):
            t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            
            if method == "ddpm":
                x_t = self.reverse_process(x_t, t)
            else:
                # Get previous timestep in the sequence
                prev_timestep = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t_prev = torch.full((batch_size,), prev_timestep, device=self.device, dtype=torch.long)
                x_t = self.ddim_reverse_process(x_t, t, t_prev, eta=eta)
                
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
        state["prediction_type"] = self.prediction_type
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
            prediction_type=ddpm_config.get("prediction_type", "epsilon"),
        )
