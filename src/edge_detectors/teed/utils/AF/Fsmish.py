"""
Script based on:
Wang, Xueliang, Honge Ren, and Achuan Wang.
"Smish: A Novel Activation Function for Deep Learning Methods."
Electronics 11.4 (2022): 540.
"""

import torch


@torch.jit.script
def smish(input):
    """
    Applies the smish function element-wise:
    smish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + sigmoid(x)))
    """
    return input * torch.tanh(torch.log(1 + torch.sigmoid(input)))
