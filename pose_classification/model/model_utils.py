# -*- coding: utf-8 -*-
# author: Trung Pham
# description: Utility functions used to build models

import torch
import torch.nn as nn 
import math 

def silu_activation(x):
    """
    Implement SiLU activation 
    Formula: SiLU(x) = x * sigmoid(x)

    Parameters
    ----------
    x: torch.Tensor
        Input value

    Returns
    ----------
    torch.Tensor
        Output of SiLU activation function
    """
    return x * torch.sigmoid(x)


def leaky_activation(x):
    """
    Implement Leaky-ReLU activation function
    Adjust alpha parameter to govern slope for values lower than threshold in ReLU function

    Parameters
    ----------
    x: torch.Tensor
        Input value

    Returns
    ----------
    torch.Tensor
        Output of Leaky ReLU activation function
    """
    activation = nn.LeakyReLU(negative_slope=0.1)
    return activation(x)

def autopad(k, p=None, d=1):  
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p