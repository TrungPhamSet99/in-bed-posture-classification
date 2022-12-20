# -*- coding: utf-8 -*-
# author: Trung Pham
# description: Utility functions used to build models

import torch
import torch.nn as nn 
import torch.nn.functional as F
import math 
import numpy as np 
from utils.general import pose_to_embedding_v2

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
    """Init weight for model

    Parameters
    ----------
    model : nn.Module as Pytorch model
        Input model
    """
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

def combine_pose_embedding_and_autoencoder_output(pose_embedding, autoencoder_output):
    """Combine pose embedding and autoencoder output into one tensor

    Parameters
    ----------
    pose_embedding : torch.Tensor or numpy
        Pose embedding from HRNet
    autoencoder_output : torch.Tensor or numpy
        Image feature extracted from autoencoder
    """
    if isinstance(pose_embedding, np.ndarray):
        pose_embedding = torch.from_numpy(pose_embedding)
    elif not isinstance(pose_embedding, torch.Tensor):
        raise TypeError("Only support torch.Tensor or np.ndarray for 'pose_embedding' argument")
    
    if isinstance(autoencoder_output, np.ndarray):
        autoencoder_output = torch.from_numpy(autoencoder_output)
    elif not isinstance(autoencoder_output, torch.Tensor):
        raise TypeError("Only support torch.Tensor or np.ndarray for 'autoencoder_output' argument")
    
    assert pose_embedding.shape == torch.Size([2,22]), "Invalid shape for pose embedding"
    assert len(autoencoder_output.shape) == 3, "Output of autoencoder must be 3D tensor"
    autoencoder_output_size = autoencoder_output.shape[2]
    # Process pose embedding vector to get suitable shape 
    pad_size = (int((autoencoder_output_size-pose_embedding.shape[1])//2),
                int((autoencoder_output_size-pose_embedding.shape[1])//2 + 1))
    pose_embedding = F.pad(pose_embedding, pad_size, "constant", 0)
    pose_embedding = pose_embedding.repeat(int(autoencoder_output_size//pose_embedding.shape[0]),1)
    pose_embedding = torch.vstack((pose_embedding, pose_embedding[1,:])) # torch.Size([29,29])
    pose_embedding = torch.unsqueeze(pose_embedding, 0)
    pose_embedding = pose_embedding.repeat(int(autoencoder_output.shape[0]//4), 1, 1)
    # Concate pose embedding and return combined tensor
    return torch.cat((autoencoder_output, pose_embedding), dim=0)

if __name__ == "__main__":
    autoencoder_output = torch.rand(32,59,59)
    raw_pose = np.random.rand(2,14)
    pose_embedding = pose_to_embedding_v2(raw_pose)
    print(pose_embedding.shape)

    combined_tensor = combine_pose_embedding_and_autoencoder_output(pose_embedding, autoencoder_output)
    print(combined_tensor.shape)
