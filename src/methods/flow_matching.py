"""
Flow Matching for Generative Modeling
"""

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000,
    ):
        super().__init__(model, device)
        self.num_timesteps = num_timesteps
        self.to(device)

    def compute_loss(
        self,
        x_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the flow matching loss.
        
        Args:
            x_1: Clean data samples (batch_size, channels, height, width)
            **kwargs: Additional arguments
            
        Returns:
            loss: Scalar MSE loss
            metrics: Dictionary with loss value
        """
        batch_size = x_1.shape[0]
        
        # 1. Sample t ~ Uniform(0, 1)
        t = torch.rand((batch_size,), device=self.device)
        
        # 2. Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # 3. Compute x_t = (1 - t) * x_0 + t * x_1
        # Need to reshape t for broadcasting: (batch_size, 1, 1, 1)
        t_reshaped = t[:, None, None, None]
        x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1
        
        # 4. Target velocity is x_1 - x_0
        target = x_1 - x_0
        
        # 5. Model prediction
        # Scale t from [0, 1] to [0, num_timesteps] for the model's time embedding
        # Note: Some implementations use t * 1000, others use specific embeddings.
        # We'll stick to scaling to match the UNet's expected input range.
        t_input = t * (self.num_timesteps - 1)
        model_input = torch.cat([x_t, condition], dim=1) if condition is not None else x_t
        v_theta = self.model(model_input, t_input)
        
        loss = F.mse_loss(v_theta, target)
        
        return loss, {'loss': loss.item(), 'mse': loss.item()}

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 50,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using Euler integration.
        
        Args:
            batch_size: Number of samples to generate
            image_shape: (channels, height, width)
            num_steps: Number of Euler steps
            
        Returns:
            samples: Generated images at t=1
        """
        self.model.eval()
        
        # Start from pure noise at t=0
        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        if condition is not None:
            condition = condition.to(self.device)
            if condition.shape[0] != batch_size:
                raise ValueError(
                    f"Condition batch size ({condition.shape[0]}) must match batch_size ({batch_size})."
                )
        
        dt = 1.0 / num_steps
        
        # Iterate from t=0 to t=1
        for i in range(num_steps):
            # Current time t
            t_val = i / num_steps
            t = torch.full((batch_size,), t_val, device=self.device)
            
            # Scale t for model
            t_input = t * (self.num_timesteps - 1)
            
            # Predict velocity
            model_input = torch.cat([x_t, condition], dim=1) if condition is not None else x_t
            v_theta = self.model(model_input, t_input)
            
            # Euler step: x_{t+dt} = x_t + v_theta * dt
            x_t = x_t + v_theta * dt
            
        return x_t

    def to(self, device: torch.device) -> "FlowMatching":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        fm_config = config.get("flow_matching", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=fm_config.get("num_timesteps", 1000),
        )
