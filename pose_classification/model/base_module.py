# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define base module used to build models 
import re
import collections
import math
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import utils.general as utils
# from utils.general import load_config


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])


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
        # self.act = eval(f"utils.{act}_activation")
        self.act = nn.ReLU()
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
        if self.shortcut:
            return inputs + self.conv2(self.conv1(inputs))
        else:
            return self.conv2(self.conv1(inputs))
    
    def predict(self, inputs):
        return self.conv1(inputs)


class LinearBlock(nn.Module):
    def __init__(self, input_channels, output_channels, act_func="relu"):
        super(LinearBlock, self).__init__() 
        self.linear = nn.Linear(input_channels, output_channels)
        self.act_func = act_func

    def forward(self, inputs, **kwargs):
        if self.act_func == "relu":
            return F.relu(self.linear(inputs))
        elif self.act_func == "softmax":
            return F.softmax(self.linear(inputs), dim=0)
        else:
            raise NotImplementedError(f"Do not support {self.act_func} function")


class DropoutBlock(nn.Module):
    def __init__(self, dropout_rate=.5):
        super(DropoutBlock, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, inputs, **kwargs):
        return self.dropout(inputs)



class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.
        Args:
            block (namedtuple): A BlockArgs type argument.
        Returns:
            block_string: A String form of BlockArgs.
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.
        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class TransposedConv2dStaticSamePadding(nn.ConvTranspose2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #         op: output padding
    #     Output after ConvTranspose2d:
    #         (i-1)*s + (k-1)*d + op + 1

    def __init__(self, in_channels, out_channels, kernel_size, image_size, stride=1, output_padding=0, groups=1, bias=True, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, output_padding, groups, bias, dilation)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.output_padding = output_padding
        # NOTE: image_size here represents the desired output image_size
        oh, ow = (image_size, image_size) if isinstance(image_size, int) else image_size
        self._oh, self._ow = oh, ow
        sh, sw = self.stride
        ih, iw = math.ceil(oh / sh), math.ceil(ow / sw) # using same calculation in Conv2dStaticSamePadding
        self._ih, self._iw = ih, iw
        kh, kw = self.weight.size()[-2:]
        # actual height/width after TransposedConv2d
        actual_oh = (ih - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + self.output_padding + 1
        actual_ow = (iw - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + self.output_padding + 1
        crop_h = actual_oh - oh
        crop_w = actual_ow - ow
        assert crop_h >= 0 and crop_w >= 0
        self._crop_h = crop_h
        self._crop_w = crop_w
        self._actual_oh = actual_oh
        self._actual_ow = actual_ow

    def forward(self, x):
        # assert x.size()[-2:] == (self._ih, self._iw)
        x = F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  self.output_padding, self.groups, self.dilation)
        # assert x.size()[-2:] == (self._actual_oh,  self._actual_ow)
        crop_h, crop_w = self._crop_h, self._crop_w
        if crop_h > 0 or crop_w > 0:
            x = x[:, :, crop_h // 2 : - (crop_h - crop_h // 2), crop_w // 2 : - (crop_w - crop_w // 2)]
        # assert x.size()[-2:] == (self._oh, self._ow)
        return x