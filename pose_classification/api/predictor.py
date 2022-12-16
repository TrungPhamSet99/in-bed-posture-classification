# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define predictor for pose classification
import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Compose, Normalize
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from utils.general import load_config, accuracy, plot_confusion_matrix, pose_to_embedding_v2
from model.target_model import model_gateway


class PosePredictor():
    # Predictor for Pose classifier module
    AE_RGB_MEAN = [0.18988903, 0.18988903, 0.18988903]
    AE_RGB_STD = [0.09772425, 0.09772425, 0.09772425]
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
        self.module_name = self.conf["module_name"]

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
            self.device = torch.device(self.conf["device"])
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
        if self.module_name == "classifier":
            # Get embedding vector as torch.Tensor from numpy pose
            return pose_to_embedding_v2(input_data).unsqueeze(0)
        elif self.module_name == "autoencoder":
            input_data = cv2.resize(input_data, (120,120)) 
            input_data = Compose([ToTensor(), Normalize(mean=self.AE_RGB_MEAN, std=self.AE_RGB_STD)])(input_data) # Convert (W,H,C) -> (C,W,H), normalize and convert to torch.FloatTensor
            return input_data.unsqueeze(0)
        else:
            raise ValueError(f"Do not support {self.module_name} as module name")
    
    def _postprocess(self, input_data):
        """Postprocess DNN output for final prediction

        Parameters
        ----------
        input_data : torch.Tensor
            DNN model output
        """
        if self.module_name == "classifier":
            return torch.argmax(input_data).cpu().detach().numpy()
        else:
            return input_data.cpu().detach().numpy()

    def execute(self, input_data):
        """Execute prediction on input data

        Parameters
        ----------
        input_data : np.ndarray
            Input data for prediction
        """
        # Validate and preprocess input data
        self._validate_input_data(input_data)
        input_data = self._preprocess(input_data)
        # Move input to device
        input_data = input_data.to(self.device)
        # Run inference using DNN model
        with torch.no_grad():
            if self.module_name == "classifier":
                dnn_output = self.model(input_data)
            else:
                dnn_output = self.model.predict(input_data)
        # Post process and return output 
        return self._postprocess(dnn_output)
