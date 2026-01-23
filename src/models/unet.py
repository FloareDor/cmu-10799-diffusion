"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # Input and Time

        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(time_embed_dim)
        # 3 channels -> base_channels = 128 channels
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = base_channels # 128

        # We need to remember how many channels were at each step 
        # so we can combine them later in the Up-Path
        self.input_block_chans = [base_channels]

        ds = 1 # current downsampling factor (1=32*32)

        # Assume 64x64 input for resolution-based attention
        current_resolution = 64
        
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                layers = []
                resblock = ResBlock(
                    in_channels=ch, 
                    out_channels=out_ch, 
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                layers.append(resblock)

                ch = out_ch
                # Check actual resolution (not ds factor) for attention
                if current_resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels=out_ch, num_heads=num_heads))

                # Add the combined block to our main list
                self.down_blocks.append(nn.ModuleList(layers))
                self.input_block_chans.append(ch)
            
            # If this is NOT the last level, we need to downsample (shrink image)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))
                self.input_block_chans.append(ch)
                ds *= 2
                current_resolution //= 2
        
        # =====================================================================
        # Middle block
        # =====================================================================
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim, dropout, use_scale_shift_norm),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, time_embed_dim, dropout, use_scale_shift_norm),
        ])
        
        # =====================================================================
        # Upsampling path (decoder)
        # =====================================================================
        self.up_blocks = nn.ModuleList()
        
        # Copy channel list for iteration (we pop from this during construction)
        up_skip_channels = list(self.input_block_chans)
        
        for level in reversed(range(len(channel_mult))):
            mult = channel_mult[level]
            out_ch = base_channels * mult
            
            # Calculate number of blocks for this level:
            # Each level needs num_res_blocks + 1 to consume:
            # - num_res_blocks resblock skips from the same encoder level
            # - 1 additional skip (downsample from previous encoder level, or head for level 0)
            num_blocks = num_res_blocks + 1
            
            for i in range(num_blocks):
                skip_ch = up_skip_channels.pop()
                
                layers = []
                layers.append(
                    ResBlock(
                        in_channels=ch + skip_ch,  # Concatenate with skip connection
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = out_ch
                
                # Add attention at matching resolutions
                if current_resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                
                # Add upsample at the last block of each level (except level 0)
                if level != 0 and i == num_blocks - 1:
                    layers.append(Upsample(ch))
                    current_resolution *= 2
                
                self.up_blocks.append(nn.ModuleList(layers))
        
        # =====================================================================
        # Output
        # =====================================================================
        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Compute time embedding
        t_emb = self.time_embed(t)
        
        # Initial convolution
        h = self.head(x)
        
        # =====================================================================
        # Downsampling path - collect skip connections
        # =====================================================================
        skips = [h]
        for block in self.down_blocks:
            if isinstance(block, Downsample):
                h = block(h)
            else:
                # It's a ModuleList of [ResBlock, maybe AttentionBlock]
                for layer in block:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            skips.append(h)
        
        # =====================================================================
        # Middle block
        # =====================================================================
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # =====================================================================
        # Upsampling path - use skip connections
        # =====================================================================
        for block in self.up_blocks:
            # Concatenate with skip connection
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            # Process through layers in this block
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, Upsample):
                    h = layer(h)
                else:
                    h = layer(h)
        
        # =====================================================================
        # Output
        # =====================================================================
        return self.out(h)


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
