# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define single evaluator for pose classification
import os
import json 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score

from utils.general import load_config, accuracy, plot_confusion_matrix, pose_to_embedding_v2, colorstr
from data.normal_dataset import NormalPoseDataset
from model.target_model import model_gateway

CLASSES = ["lying_left_0", "lying_left_1", "lying_left_2"]


class SingeModuleEvaluator:
    def __init__(self, config):
        """Constructor for SingleModuleEvaluator

        Parameters
        ----------
        config : str or dict
            Config as dict or path to JSON config file

        Raises
        ------
        TypeError
            Raise TypeError when config type is not str or dict
        """
        if isinstance(config, str):
            self.config = load_config(config)
        elif isinstance(config, dict):
            self.config = config 
        else:
            raise TypeError("Only support str or dict type for `config` parameters")
        
        self.model_config = load_config(self.config['model_config'])
        self.weight = self.config['weight']
        self.device = None 
        self.model = model_gateway(self.config["model_name"], self.model_config)
        self.test_dataset = NormalPoseDataset(self.config['data_dir'],
                                              self.config['test_list'])
    
    def _init_torch_tensor(self):
        """Init torch tensor for device
        """
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device('cpu')
    
    def initialize(self):
        """Init evaluator 
        - Init device
        - Load weight
        - Create dataset
        """
        print(colorstr(f"Number data samples in test set: {len(self.test_dataset)}"))
        self._init_torch_tensor()
        self.model.load_state_dict(torch.load(self.weight))
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=32,
                                                      shuffle=False, num_workers=4,
                                                      pin_memory=False)
        self.model.eval()

    def run_evaluate(self):
        """Execute evaluation on test set

        Returns
        -------
        tuple
            (labels, prediction)
        """
        label = []
        prediction = []
        for batch in self.testloader:
            poses, labels = batch
            output = self.model(poses, self.device)
            output = torch.argmax(output, axis=1).numpy()
            labels = labels.numpy()
            label += labels.tolist()
            prediction += output.tolist()
        self.create_report(label, prediction)
        return label, prediction

    def create_report(self, label, prediction):
        """Create and print classification report

        Parameters
        ----------
        label : np.ndarray or list
            label values
        prediction : np.ndarray or list
            prediction values
        """
        precision, recall, fscore, support = score(label, prediction)
        report = classification_report(label, prediction)
        print(label)
        print(prediction)
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        print(report)