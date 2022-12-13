# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define base model as abstract model for DNN model used in pose classification module
from abc import ABCMeta, abstractmethod
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import model.base_module as base_module
from tqdm import tqdm
from utils.general import colorstr


class BasePoseClassifier(nn.Module):
    # Abstract model 
    def __init__(self, config, model_name):
        """Constructor for BasePoseClassifier

        Parameters
        ----------
        config : dict
            Config parameters to build model as dict
        model_name : str
            Model name
        """
        super(BasePoseClassifier, self).__init__()
        self.config = config
        for key in config.keys():
            if "linear" in key:
                setattr(self, key, nn.Linear(config[key][0], config[key][1]))
            elif "conv1d" in key:
                index = key.split("_")[-1]
                _key = f"conv_block_{index}"
                if len(config[key]) <= 3:
                    conv_layer = nn.Conv1d(
                        config[key][0], config[key][1], config[key][2])
                else:
                    conv_layer = nn.Conv1d(config[key][0], config[key][1], config[key][2],
                                           padding=config[key][3])
                if model_name == "PoseClassifierV2_1":
                    module = [conv_layer, nn.ReLU(
                    ), nn.BatchNorm1d(config[key][1])]
                elif model_name == "PoseClassifierV2_2":
                    module = [conv_layer, nn.BatchNorm1d(
                        config[key][1]), nn.ReLU()]
                else:
                    pass

                if index == 3 or index == 5:
                    module.append(nn.MaxPool1d(2, 2))

                sub_net = nn.Sequential(*module)
                setattr(self, _key, sub_net)
            else:
                pass
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=0)
        self.dropout1 = nn.Dropout(.5)
        self.dropout2 = nn.Dropout(.3)
        self.dropout3 = nn.Dropout(.2)

    @abstractmethod
    def forward(self, xb, device):
        pass


class EnDeCoder(nn.Module):
    """Class represent for EnCoder and Decoder as Sequential in AutoEncoder model version 1
    """
    def __init__(self, config):
        super(EnDeCoder, self).__init__()
        self.module_list = self.parse_model(config)

    @staticmethod
    def parse_model(config):
        """Parse module from config

        Parameters
        ----------
        config : dict
            Parameter config as dict

        Returns
        -------
        list
            list of nn.Module
        """
        module_list = []
        for idx in tqdm(range(len(config)), desc=colorstr("Parsing module list from config")):
            element = config[idx]
            module_name, params = element[0], element[1:]
            module = eval(f"base_module.{module_name}")
            module_list.append(module(*params))
        return module_list
    
    def forward(self, inputs, **kwargs):
        """Forward implementation as Sequential model

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        for module in self.module_list:
            inputs = module(inputs)
        return inputs 


class BottleNeckAE(nn.Module):
    """Class represent bottleneck module in AutoEncoder model
    """
    def __init__(self, config, training=False):
        """Constructor for BottleNeckAE class

        Parameters
        ----------
        config : dict
            Parameters configs for BottleNeckAE
        training : bool, optional
            True if model work in training mode, by default False
        """
        super(BottleNeckAE, self).__init__()
        self.module_list = []
        self.training = training
        for element in config:
            module = eval(f"nn.{element[0]}")
            self.module_list.append(module(*element[1:]))
    
    def forward(self, inputs, **kwargs):
        """Forward implementation
            If model is in training mode, take output from the last module in bottleneck
            Else just take output from the first module in bottleneck
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        outputs = []
        for module in self.module_list:
            inputs = module(inputs)
            outputs.append(inputs)
        if self.training:
            return outputs[-1]
        else:
            return outputs[0]
