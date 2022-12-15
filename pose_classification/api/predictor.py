# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define predictor for pose classification
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score

from utils.general import load_config, accuracy, plot_confusion_matrix, pose_to_embedding_v2
from model.target_model import model_gateway


CLASSES = ["lying_left_0", "lying_left_1", "lying_left_2"]


class PosePredictor():
    # Predictor for Pose classifier module
    def __init__(self, config):
        """constructor for PosePredictor

        Parameters
        ----------
        config : dict
            config for PosePredictor as dict
        """
        self.conf = load_config(config)
        self.model_config = load_config(self.conf['model_config'])
        self.model = model_gateway(self.conf['model_name'], self.model_config)
        self.weights = self.conf['weight']
        self.device = None

    def init_predictor(self):
        """Initialize predictor before execute
        - Load weight 
        - Init device
        - Create test dataset if neccessary
        """
        self._init_torch_tensor()
        self.model.load_state_dict(torch.load(self.weights))
        self.model.eval()

    def _init_torch_tensor(self):
        """Init torch tensor for device
        """
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
            # Move model to GPU 
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
    
    @staticmethod
    def _validate_input_data(input_data):
        assert isinstance(input_data, (np.ndarray, torch.Tensor)), f"Only support np.ndarray or torch.Tensor but got {type(input_data)} as input"
    
    def _preprocess(self, input_data):
        """Preprocess input data before execute prediction

        Parameters
        ----------
        input_data : np.ndarray or torch.Tensor
            Input keypoints
        """
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        return pose_to_embedding_v2(input_data).unsqueeze(0)
    
    def _postprocess(self, input_data):
        """Postprocess DNN output for final prediction

        Parameters
        ----------
        input_data : torch.Tensor
            DNN model output
        """
        return torch.argmax(input_data).numpy()

    def execute(self, input_data):
        """Execute prediction on input data

        Parameters
        ----------
        input_data : _type_
            _description_
        """
        self._validate_input_data(input_data)
        input_data = self._preprocess(input_data)
        dnn_output = self.model(input_data, self.device)
        return self._postprocess(dnn_output)
