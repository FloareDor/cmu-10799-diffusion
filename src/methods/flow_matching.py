"""Flow Matching for Generative Modeling."""

from typing import Dict, Tuple, Optional

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
        # HW4: training-time conditioning controls
        enable_timestep_edge_weighting: bool = False,
        edge_weight_power: float = 1.0,
        condition_dropout_prob: float = 0.0,
        edge_cycle_loss_weight: float = 0.0,
        contrastive_loss_weight: float = 0.0,
        contrastive_margin: float = 0.02,
        # HW4: sampling-time CFG controls
        cfg_guidance_scale: float = 1.0,
        cfg_zero_steps: int = 0,
    ):
        super().__init__(model, device)
        self.num_timesteps = num_timesteps
        self.enable_timestep_edge_weighting = enable_timestep_edge_weighting
        self.edge_weight_power = edge_weight_power
        self.condition_dropout_prob = condition_dropout_prob
        self.edge_cycle_loss_weight = edge_cycle_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_margin = contrastive_margin
        self.cfg_guidance_scale = cfg_guidance_scale
        self.cfg_zero_steps = cfg_zero_steps

        if not (0.0 <= self.condition_dropout_prob < 1.0):
            raise ValueError("condition_dropout_prob must be in [0, 1).")
        if self.edge_weight_power <= 0:
            raise ValueError("edge_weight_power must be > 0.")
        if self.edge_cycle_loss_weight < 0:
            raise ValueError("edge_cycle_loss_weight must be >= 0.")
        if self.contrastive_loss_weight < 0:
            raise ValueError("contrastive_loss_weight must be >= 0.")
        if self.cfg_guidance_scale < 0:
            raise ValueError("cfg_guidance_scale must be >= 0.")
        if self.cfg_zero_steps < 0:
            raise ValueError("cfg_zero_steps must be >= 0.")

        # Fixed Sobel kernels for differentiable edge consistency loss.
        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.to(device)

    def _apply_timestep_condition_weight(
        self,
        condition: Optional[torch.Tensor],
        t: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if condition is None:
            return None
        if not self.enable_timestep_edge_weighting:
            return condition
        # Strong early, weaker late: weight(t) = (1 - t)^alpha.
        w = torch.pow((1.0 - t).clamp(0.0, 1.0), self.edge_weight_power)
        return condition * w[:, None, None, None]

    def _drop_condition_for_cfg_training(
        self,
        condition: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if condition is None or self.condition_dropout_prob <= 0.0:
            return condition
        drop_mask = (
            torch.rand((condition.shape[0], 1, 1, 1), device=condition.device)
            < self.condition_dropout_prob
        ).float()
        return condition * (1.0 - drop_mask)

    def _sobel_edge_map(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in [-1, 1], shape (B, C, H, W).
        gray = ((x + 1.0) * 0.5).mean(dim=1, keepdim=True)  # [0, 1]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        denom = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return mag / denom

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        t_input: torch.Tensor,
        condition: Optional[torch.Tensor],
    ) -> torch.Tensor:
        model_input = torch.cat([x_t, condition], dim=1) if condition is not None else x_t
        return self.model(model_input, t_input)

    def compute_loss(
        self,
        x_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute FM loss with optional HW4 terms."""
        batch_size = x_1.shape[0]

        # 1) Sample t ~ U(0, 1), 2) sample noise x_0, 3) interpolate x_t.
        t = torch.rand((batch_size,), device=self.device)
        x_0 = torch.randn_like(x_1)
        t_reshaped = t[:, None, None, None]
        x_t = (1.0 - t_reshaped) * x_0 + t_reshaped * x_1

        # 4) Target velocity
        target = x_1 - x_0
        t_input = t * (self.num_timesteps - 1)

        condition_used = self._drop_condition_for_cfg_training(condition)
        condition_used = self._apply_timestep_condition_weight(condition_used, t)

        # 5) Main prediction and MSE
        v_theta = self._predict_velocity(x_t, t_input, condition_used)
        mse = F.mse_loss(v_theta, target)
        total_loss = mse

        metrics: Dict[str, torch.Tensor] = {
            "loss": total_loss.detach(),
            "mse": mse.detach(),
        }

        # Optional: edge cycle consistency loss.
        if self.edge_cycle_loss_weight > 0.0 and condition is not None:
            # From x_t = (1-t)x_0 + t x_1 and v = x_1 - x_0:
            # x_1_hat = x_t + (1-t) * v_theta
            x_1_hat = x_t + (1.0 - t_reshaped) * v_theta
            pred_edges = self._sobel_edge_map(x_1_hat)
            cond_edges = ((condition + 1.0) * 0.5).mean(dim=1, keepdim=True)
            edge_cycle = F.l1_loss(pred_edges, cond_edges)
            total_loss = total_loss + self.edge_cycle_loss_weight * edge_cycle
            metrics["edge_cycle"] = edge_cycle.detach()

        # Optional: simple contrastive condition separation term.
        if self.contrastive_loss_weight > 0.0 and condition is not None and batch_size > 1:
            # Deterministic non-identity permutation.
            perm = (torch.arange(batch_size, device=self.device) + 1) % batch_size
            mismatched = condition[perm]
            mismatched = self._apply_timestep_condition_weight(mismatched, t)
            v_mismatch = self._predict_velocity(x_t, t_input, mismatched)

            pos_err = F.mse_loss(v_theta, target, reduction="none").mean(dim=(1, 2, 3))
            neg_err = F.mse_loss(v_mismatch, target, reduction="none").mean(dim=(1, 2, 3))
            contrastive = F.relu(self.contrastive_margin + pos_err - neg_err).mean()
            total_loss = total_loss + self.contrastive_loss_weight * contrastive
            metrics["contrastive"] = contrastive.detach()

        metrics["loss"] = total_loss.detach()
        return total_loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 50,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        cfg_zero_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using Euler integration with optional CFG."""
        self.model.eval()

        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        if condition is not None:
            condition = condition.to(self.device)
            if condition.shape[0] != batch_size:
                raise ValueError(
                    f"Condition batch size ({condition.shape[0]}) must match batch_size ({batch_size})."
                )

        g_scale = self.cfg_guidance_scale if guidance_scale is None else guidance_scale
        zero_steps = self.cfg_zero_steps if cfg_zero_steps is None else cfg_zero_steps
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((batch_size,), t_val, device=self.device)
            t_input = t * (self.num_timesteps - 1)

            if condition is None:
                v_theta = self._predict_velocity(x_t, t_input, None)
            else:
                cond_t = self._apply_timestep_condition_weight(condition, t)

                # If guidance is effectively off, run single conditioned pass.
                if g_scale <= 1.0 and zero_steps <= 0:
                    v_theta = self._predict_velocity(x_t, t_input, cond_t)
                else:
                    uncond = torch.zeros_like(condition)
                    uncond_t = self._apply_timestep_condition_weight(uncond, t)
                    v_uncond = self._predict_velocity(x_t, t_input, uncond_t)
                    v_cond = self._predict_velocity(x_t, t_input, cond_t)

                    # CFG-Zero style: disable guidance boost for first N steps.
                    step_scale = 1.0 if i < zero_steps else g_scale
                    v_theta = v_uncond + step_scale * (v_cond - v_uncond)

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
            enable_timestep_edge_weighting=fm_config.get("enable_timestep_edge_weighting", False),
            edge_weight_power=float(fm_config.get("edge_weight_power", 1.0)),
            condition_dropout_prob=float(fm_config.get("condition_dropout_prob", 0.0)),
            edge_cycle_loss_weight=float(fm_config.get("edge_cycle_loss_weight", 0.0)),
            contrastive_loss_weight=float(fm_config.get("contrastive_loss_weight", 0.0)),
            contrastive_margin=float(fm_config.get("contrastive_margin", 0.02)),
            cfg_guidance_scale=float(fm_config.get("cfg_guidance_scale", 1.0)),
            cfg_zero_steps=int(fm_config.get("cfg_zero_steps", 0)),
        )
