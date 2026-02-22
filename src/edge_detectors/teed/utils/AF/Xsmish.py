"""
Smish as nn.Module wrapper.
Based on: Wang et al., "Smish: A Novel Activation Function for Deep Learning Methods."
"""

import torch
from torch import nn

from .Fsmish import smish


class Smish(nn.Module):
    """
    Applies the smish function element-wise.
    smish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + sigmoid(x)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return smish(input)
