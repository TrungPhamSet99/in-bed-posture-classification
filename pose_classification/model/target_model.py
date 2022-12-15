# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define target models with different version for experiments
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from model.base_model import BasePoseClassifier, EnDeCoder, BottleNeckAE
from utils.general import load_config

class PoseClassifierV1(BasePoseClassifier):
    def __init__(self, config, model_name):
        """Constructor for PoseClassifierV1

        Parameters
        ----------
        config : dict
            Config parameter to build model v1 as dict
        model_name : str
            Model name
        """
        super(PoseClassifierV1, self).__init__(config, model_name)

    def forward(self, xb, device):
        """Forward path for PoseClassifierV1 model

        Parameters
        ----------
        xb : np.ndarray or torch.Tensor
            Input
        device : str
            device to run forward (cpu or gpu)

        Returns
        -------
        torch.Tensor
            Output from model as torch.Tensor after softmax activation
        """
        xb = xb.float()
        xb.to(device)

        # for i in range(len(list(self.config.keys()))):
        #     layer = eval(f"self.linear{i+1}")
        #     xb = nn.Dropout(.5)
        #     xb = F.relu(layer(xb))
        # output = self.softmax(xb)

        out = F.relu(self.linear1(xb))
        out = self.dropout1(out)
        out = F.relu(self.linear2(out))
        out = self.dropout2(out)
        out = F.relu(self.linear3(out))
        out = self.dropout3(out)
        out = self.linear4(out)
        output = F.softmax(out, dim=0)
        return output


class PoseClassifierV2_1(BasePoseClassifier):
    def __init__(self, config, model_name):
        """Constructor for PoseClassifierV2_1

        Parameters
        ----------
        config : dict
            Config parameter to build model v1 as dict
        model_name : str
            Model name
        """
        super(PoseClassifierV2_1, self).__init__(config, model_name)
        for key in config.keys():
            if "linear" in key:
                index = key.split("_")[-1]
                _key = f"linear_block_{index}"
                module = nn.Sequential(nn.Dropout(.5),
                                       nn.Linear(
                                           config[key][0], config[key][1]),
                                       nn.ReLU())
                setattr(self, _key, module)

    def forward(self, xb, device):
        """Forward path for PoseClassifierV2_1 model

        Parameters
        ----------
        xb : np.ndarray or torch.Tensor
            Input
        device : str
            device to run forward (cpu or gpu)

        Returns
        -------
        torch.Tensor
            Output from model as torch.Tensor after softmax activation
        """
        xb = xb.float()
        xb.to(device)
        conv_block_list = [block for block in list(
            self.config.keys()) if "conv" in block]
        linear_block_list = [block for block in list(
            self.config.keys()) if "linear" in block]

        for i in range(len(conv_block_list)):
            conv_block = eval(f"self.conv_block_{i+1}")
            xb = conv_block(xb)

        xb = xb.reshape(xb.size(0), -1)
        for i in range(len(linear_block_list)):
            linear_block = eval(f"self.linear_block_{i+1}")
            xb = linear_block(xb)
        output = self.softmax(xb)
        return output


class PoseClassifierV2_2(BasePoseClassifier):
    def __init__(self, config, model_name):
        """Constructor for PoseClassifierV2_2

        Parameters
        ----------
        config : dict
            Config parameter to build model v1 as dict
        model_name : str
            Model name
        """
        super(PoseClassifierV2_2, self).__init__(config, model_name)
        for key in config.keys():
            if "linear" in key:
                index = key.split("_")[-1]
                _key = f"linear_block_{index}"
                module = nn.Sequential(nn.Dropout(.5),
                                       nn.Linear(
                                           config[key][0], config[key][1]),
                                       nn.ReLU())
                setattr(self, _key, module)

    def forward(self, xb, device):
        """Forward path for PoseClassifierV2_2 model

        Parameters
        ----------
        xb : np.ndarray or torch.Tensor
            Input
        device : str
            device to run forward (cpu or gpu)

        Returns
        -------
        torch.Tensor
            Output from model as torch.Tensor after softmax activation
        """
        xb = xb.float()
        xb.to(device)
        conv_block_list = [block for block in list(
            self.config.keys()) if "conv" in block]
        linear_block_list = [block for block in list(
            self.config.keys()) if "linear" in block]

        for i in range(len(conv_block_list)):
            conv_block = eval(f"self.conv_block_{i+1}")
            xb = conv_block(xb)

        xb = xb.reshape(xb.size(0), -1)
        for i in range(len(linear_block_list)):
            linear_block = eval(f"self.linear_block_{i+1}")
            xb = linear_block(xb)
        output = self.softmax(xb)
        return output


class End2EndPoseClassifer(nn.Module):
    def __init__(self, raw_model, supine_model, lying_left_model, lying_right_model):
        """Constructor for End2EndPoseClassifier

        Parameters
        ----------
        raw_model : nn.Module
            Raw model 
        supine_model : nn.Module
            Supine model
        lying_left_model : nn.Module
            Lying left model
        lying_right_model : nn.Module
            Lying right model
        """
        super(End2EndPoseClassifer, self).__init__()
        self.raw_model = raw_model
        self.supine_model = supine_model
        self.lying_right_model = lying_right_model
        self.lying_left_model = lying_left_model

    def forward(self, xb, device):
        """Forward path for End2EndPoseClassifier model

        Parameters
        ----------
        xb : np.ndarray or torch.Tensor
            Input
        device : str
            device to run forward (cpu or gpu)

        Returns
        -------
        torch.Tensor
            Output from model as torch.Tensor after softmax activation
        """
        xb = xb.float()
        xb.to(device)
        raw_pred = self.raw_model(xb, device)
        final_output = torch.zeros(raw_pred.shape[0], raw_pred.shape[1])
        raw_pred = torch.argmax(raw_pred, axis=1)

        # Get indexes for each class
        lying_left_indexes = torch.where(raw_pred == 0)[0]
        supine_indexes = torch.where(raw_pred == 1)[0]
        lying_right_indexes = torch.where(raw_pred == 2)[0]

        # Get sub batch for each class
        # Use submodel to predict sub-batch
        if lying_left_indexes.size()[0] != 0:
            raw_lying_left_samples = torch.index_select(
                input=xb, dim=0, index=lying_left_indexes)
            lying_left_outputs = self.lying_left_model(
                raw_lying_left_samples, device)
            for i in range(lying_left_indexes.size()[0]):
                index = lying_left_indexes[i]
                final_output[index, :] = lying_left_outputs[i]

        if supine_indexes.size()[0] != 0:
            raw_supine_samples = torch.index_select(
                input=xb, dim=0, index=supine_indexes)
            supine_outputs = self.supine_model(raw_supine_samples, device)
            for i in range(supine_indexes.size()[0]):
                index = supine_indexes[i]
                final_output[index, :] = supine_outputs[i]

        if lying_right_indexes.size()[0] != 0:
            raw_lying_right_samples = torch.index_select(
                input=xb, dim=0, index=lying_right_indexes)
            lying_right_outputs = self.lying_right_model(
                raw_lying_right_samples, device)
            for i in range(lying_right_indexes.size()[0]):
                index = lying_right_indexes[i]
                final_output[index, :] = lying_right_outputs[i]

        # Swap supine and lying left prediction
        raw_pred = raw_pred.index_fill(
            dim=0, index=lying_left_indexes, value=1)
        raw_pred = raw_pred.index_fill(
            dim=0, index=supine_indexes, value=0).numpy()

        # Get label from softmax output
        final_output = torch.argmax(final_output, dim=1).numpy()

        # Convert 3 classes label to 9 classes label and return output
        return (raw_pred*3) + final_output + 1

class AutoEncoderV1(nn.Module):
    def __init__(self, config, training=True):
        super(AutoEncoderV1, self).__init__()
        self.config = load_config(config)
        self.encoder = EnDeCoder(self.config["Encoder"])
        self.decoder = EnDeCoder(self.config["Decoder"])
        self.bottleneck = BottleNeckAE(self.config["Bottleneck"], training)

    def forward(self, inputs, device, **kwargs):
        inputs = self.encoder(inputs)
        if self.training:
            return self.decoder(self.bottleneck(inputs))
        else:
            return self.bottleneck(inputs)
    

def model_gateway(model_name, model_config, training=True):
    """Gateway to get model instance from config

    Parameters
    ----------
    model_name : str
        Model name
    model_config : dict
        Config parameter to build model as dict

    Returns
    -------
    nn.Module   
        A model instance as nn.Module

    Raises
    ------
    ValueError
        Raise ValueError if model_name is invalid
    """
    try:
        model = eval(model_name)
    except:
        raise ValueError(f"Do not support {model_name} in this version, please check your config again")
    if "AutoEncoder" in model_name:
        return model(model_config, training)
    else:
        return model(model_config, model_name)


