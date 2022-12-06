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