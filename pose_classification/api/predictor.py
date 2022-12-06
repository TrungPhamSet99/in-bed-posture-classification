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
# from data.data import NormalPoseDataset


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
        self.model_config = self.conf['model']
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
        # if self.conf["model_name"] != "pose_classifier_v1" and not self.conf["end2end"]:
        #     self.test_dataset = NormalPoseDataset(self.conf['data_dir'],
        #                                           self.conf['test_list'])
        #     self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=32,
        #                                                   shuffle=False, num_workers=4,
        #                                                   pin_memory=False)
        # elif self.conf["model_name"] != "pose_classifier_v1" and self.conf["end2end"]:
        #     pass
        self.model.eval()

    def _init_torch_tensor(self):
        """Init torch tensor for device
        """
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
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

    # def predict_numpy_ver1(self, input_sample):
    #     c = input_sample.split("/")[-2]
    #     print(input_sample)
    #     if self.conf["model_name"] == "pose_classifier_v1":
    #         embedding_method = eval("pose_to_embedding_v1")
    #         pose = embedding_method(np.load(input_sample))
    #     else:
    #         embedding_method = eval("pose_to_embedding_v2")
    #         pose = embedding_method(np.load(input_sample))
    #         pose = pose.unsqueeze(0)
    #     # pose = embedding_method(np.load(input))
    #     print(pose)
    #     output = self.model(pose, self.device)
    #     print(output)
    #     print("----------------------------------------")
    #     return torch.argmax(output).numpy(), CLASSES.index(c)

    # def predict_numpy_ver2(self, input_sample):
    #     file_name = input_sample.split("/")[-1]
    #     splittor = file_name.split("_")
    #     if self.conf["model_name"] == "pose_classifier_v1":
    #         embedding_method = eval("pose_to_embedding_v1")
    #         pose = embedding_method(np.load(input_sample))
    #     else:
    #         embedding_method = eval("pose_to_embedding_v2")
    #         pose = embedding_method(np.load(input_sample))
    #         pose = pose.unsqueeze(0)
    #     if len(splittor) == 3:
    #         c = "{0}_{1}".format(splittor[0], splittor[1])
    #     else:
    #         c = "{0}".format(splittor[0])
    #     #pose = embedding_method(np.load(input))
    #     output = self.model(pose, self.device)
    #     return torch.argmax(output).numpy(), CLASSES.index(c)

    # def execute(self):
    #     """Execute evaluation on test set

    #     Returns
    #     -------
    #     tuple
    #         (labels, prediction)
    #     """
    #     label = []
    #     prediction = []
    #     for batch in self.testloader:
    #         poses, labels = batch
    #         output = self.model(poses, self.device)
    #         output = torch.argmax(output, axis=1).numpy()
    #         labels = labels.numpy()
    #         label += labels.tolist()
    #         prediction += output.tolist()

    #     return label, prediction




def main():
    parser = parse_argument()
    args = parser.parse_args()

    predictor = PosePredictor("cfg/predict_config.json")
    predictor.init_predictor()
    label, prediction = predictor.execute()
    precision, recall, fscore, support = score(label, prediction)
    report = classification_report(label, prediction)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print(report)
    plot_confusion_matrix(label, prediction, CLASSES, normalize=False, title="Non-normalized confusion matrix (supine)",
                          savepath="Non_normalized_confusion_matrix.png")
    plot_confusion_matrix(label, prediction, CLASSES, normalize=True, title="Normalized confusion matrix (supine)",
                          savepath="Normalized_confusion_matrix.png")


if __name__ == "__main__":
    main()
