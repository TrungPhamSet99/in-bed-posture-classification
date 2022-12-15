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

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def count_params(model):
    """Count parameters of model (total and just trainable)

    Parameters
    ----------
    model : nn.Mopdule
        Pytorch model as nn.Module
    """
    total_params = sum(param.numel() for param in model.parameters())
    trainable_pararms = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return total_params, trainable_pararms
