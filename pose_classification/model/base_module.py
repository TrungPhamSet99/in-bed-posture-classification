# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define base module used to build models 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import model.model_utils as utils
from utils.general import load_config


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=None, act="leaky", shortcut=False):
        """Constructor for ConvBlock

        Parameters
        ----------
        input_channels : int
            Number channels of inputs
        output_channels : int
            Number channels of outputs
        kernel_size : int or tuple
            Kernel size
        stride : int
            Stride value
        padding : int, optional
            Padding value, by default None
        act : str, optional
            Activation function, by default "leaky"
        shortcut : bool, optional
            Option to use shortcut connection, by default False
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size,
                              stride, utils.autopad(kernel_size, padding), bias=False)
        self.bn = BatchNorm2D(output_channels) 
        self.act = eval(f"utils.{act}_activation")
        self.shortcut = shortcut
    
    def forward(self, inputs, **kwargs):
        """Forward implementation for ConvBlock
        Inputs -> Convolution 2D -> BatchNorm -> Activation function (Leaky ReLU)
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if not self.shortcut:
            return self.act(self.bn(self.conv(inputs)))
        else:
            return inputs + self.act(self.bn(self.conv(inputs)))

class TransposeConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=None, act="leaky", shortcut=False):
        """Constructor for TransposeConvBlock

        Parameters
        ----------
        input_channels : int
            Number channels of input
        output_channels : int
            Number channels of output
        kernel_size : int
            Kernel size
        stride : int
            Stride value
        padding : int, optional
            Padding value, by default None
        act : str, optional
            Activation function, by default "leaky"
        shortcut : bool, optional
            Activation function, by default False
        """
        super(TransposeConvBlock, self).__init__()
        assert input_channels >= output_channels, "Output channels should be less than input channels in TransposeConv2D, Check your config again" 
        self.conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size,
                                       stride, utils.autopad(kernel_size, padding), bias=False)
        self.bn = BatchNorm2D(output_channels)
        self.act = eval(f"utils.{act}_activation")
        self.shortcut = shortcut

    def forward(self, inputs, **kwargs):
        """Forward implementation for TransposeConvBlock

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if not self.shortcut:
            return self.act(self.bn(self.conv(inputs)))
        else:
            return inputs + self.act(self.bn(self.conv(inputs)))

class BatchNorm2D(nn.Module):
    def __init__(self, num_features):
        """Wrapper for BatchNorm layer

        Parameters
        ----------
        num_features : int
            Size of feature
        """
        super(BatchNorm2D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, momentum=0.97, eps=1e-3)

    def forward(self, inputs, **kwargs):
        """Forward implementation

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return self.bn(inputs)

class UpSample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        """Wrapper for Upsample module

        Parameters
        ----------
        scale_factor : int
            Factor for upsampling
        mode : str, optional
            Mode to upsample, by default "nearest"
        """
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        
    def forward(self, inputs, **kwargs):
        """Forward implementation

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return self.upsample(inputs)

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride):
        """Wrapper for Max Pooling

        Parameters
        ----------
        kernel_size : int 
            Size for kernel in pooling
        stride : int
            Stride valuen in pooling
        """
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride) 
    
    def forward(self, inputs, **kwargs):
        return self.pool(inputs)

class ReShape(nn.Module):
    def __init__(self, mode = "flatten", target_shape = None):
        """Transform the shape of tensor

        Parameters
        ----------
        mode : str, optional
            There are 2 modes "flatten" or "reconstruct", by default "flatten"
        target_shape : tuple or list, optional
            Target shape for mode "reconstruct" , by default None
        """
        super(ReShape, self).__init__()
        mode = mode.lower()
        assert mode in ["flatten", "reconstruct"], f"Do not support mode {mode} for ReShape module, Check your config again"
        self.mode = mode
        if self.mode == "reconstruct":
            assert len(target_shape) == 3, f"Target shape for 'reconstruct' mode should be 3 dimension" 
            self.target_shape = target_shape
    
    def forward(self, inputs, **kwargs):
        """Forward implementation
            If use flatten mode, just flatten tensor except 1st dimension (batch size)
            If use reconstruct mode, reshape to target shape and keep 1st dimension (batch size)
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if self.mode == "flatten":
            # Flatten all dimensions except batch (N,C,W,H) -> (N,C*W*H)
            return torch.flatten(inputs, 1)
        else:
            return torch.reshape(inputs, (-1, *self.target_shape))

class ZeroPad(nn.Module):
    def __init__(self, size):
        """Wrapper for zero padding module

        Parameters
        ----------
        size : int
            Size of padding region
        """
        super(ZeroPad, self).__init__()
        self.pad = nn.ZeroPad2d(size)

    def forward(self, inputs, **kwargs):
        return self.pad(inputs)

class ConvBottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, compress_ratio, shortcut=True):
        """Constructor for bottlecneck module in autoencoder

        Parameters
        ----------
        input_channels : int
            Number channels of input tensor
        output_channels : int
            Number channels of output tensor
        compress_ratio : float 
            Ratio to calculate bottleneck size 
        shortcut : bool, optional
            Option to use shortcut connection, by default True
        """
        super(ConvBottleneck, self).__init__()
        assert input_channels == output_channels, "Input channels should be equal output channels in bottleneck module"
        assert compress_ratio < 1, "Compress ratio must < 1"
        botlleneck_size = int(input_channels * compress_ratio)
        self.conv1 = ConvBlock(input_channels, botlleneck_size, 1, 1)
        self.conv2 = ConvBlock(botlleneck_size, output_channels, 3, 1)
        self.shortcut = shortcut

    def forward(self, inputs, training=False, **kwargs):
        """Forward implementation

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if training:
            if self.shortcut:
                return inputs + self.conv2(self.conv1(inputs))
            else:
                return self.conv2(self.conv1(inputs))
        else:
            return self.conv1(inputs)
