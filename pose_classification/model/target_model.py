# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define target models with different version for experiments
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import model.base_module as base_module
from model.base_model import BasePoseClassifier, EnDeCoder, BottleNeckAE
from utils.general import load_config, colorstr
from model.efficientnet import EfficientNetAutoEncoder, EfficientNet
from model.swin_utils import swin_transformer_gateway
from model.vgg19 import VGG19
from tqdm import tqdm

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

    def forward(self, xb):
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
        # xb = xb.float()
        # xb.to(device)
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
        # output = self.softmax(xb)
        # return output
        return xb


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


class PoseClassifierV3(nn.Module):
    def __init__(self, config, model_name):
        super(PoseClassifierV3, self).__init__()
        self.config = config
        self.model_name = model_name
        # self.feature_extractor = EfficientNet.from_pretrained("efficientnet-b2")
        # self.feature_extractor, self.swin_config = swin_transformer_gateway(pretrained=True)
        
        # print(self.feature_extractor)
        self.embedding = nn.Embedding(160, 8)
        # self.linear1 = nn.Linear(768, 512)
        # self.linear2 = nn.Linear(392, 256)
        self.linear3 = nn.Linear(96, 3)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    @staticmethod
    def parse_model(config):
        module_list = nn.ModuleList()
        for idx in tqdm(range(len(config)), desc=colorstr("Parsing module list from config")):
            element = config[idx]
            module_name, params = element[0], element[1:]
            module = eval(f"base_module.{module_name}")
            module_list.append(module(*params))
        return nn.Sequential(*module_list)


    def forward(self, inputs):
        pose_embedding = inputs[0].int()
        image = inputs[1].float()
        pose_embedding = pose_embedding.to(torch.device("cuda:0"))
        # pose_embedding = torch.abs(pose_embedding)

        # image = image.to(torch.device("cuda:0"))
        # image_feature = self.feature_extractor.extract_features(image)
        # pose_embedding = pose_embedding.flatten(start_dim=1)
        pose_embedding = self.embedding(pose_embedding).flatten(start_dim=1)
        pose_embedding = self.relu(pose_embedding)
        # image_feature = self.linear1(image_feature)

        pose_embedding = self.linear3(pose_embedding)

        return pose_embedding

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



def model_gateway(model_name, model_config):
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
    if model_name != "EfficientNetAutoEncoder":
        try:
            model = eval(model_name)
        except:
            raise ValueError(f"Do not support {model_name} in this version, please check your config again")
        if model_name != "PoseClassifierV3":
            return model(model_config, model_name)
        else:
            _model = model(model_config, model_name)
            try:
                return _model, _model.swin_config
            except:
                return _model
    else:
        return EfficientNetAutoEncoder.from_pretrained(model_config["version"])


if __name__ == "__main__":
    config_path = "../cfg/model_config/efficient_ae_config.yaml"
    config = load_config(config_path)
    model = model_gateway("EfficientNetAutoEncoder", config)
    