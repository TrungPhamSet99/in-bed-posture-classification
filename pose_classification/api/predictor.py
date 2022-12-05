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

from utils import load_config, accuracy, pose_to_embedding_v1, plot_confusion_matrix, pose_to_embedding_v2
from model import model_gateway
from data import NormalPoseDataset


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
        if self.conf["model_name"] != "pose_classifier_v1" and not self.conf["end2end"]:
            self.test_dataset = NormalPoseDataset(self.conf['data_dir'],
                                                  self.conf['test_list'])
            self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=32,
                                                          shuffle=False, num_workers=4,
                                                          pin_memory=False)
        elif self.conf["model_name"] != "pose_classifier_v1" and self.conf["end2end"]:
            pass
        self.model.eval()

    def _init_torch_tensor(self):
        """Init torch tensor for device
        """
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device('cpu')

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

    def execute(self):
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

        return label, prediction


def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser(
        "Run script to train pose classification model")
    parser.add_argument('--config-path', type=str,
                        help='Path to training config file', default="./cfg/predict_config.json")
    return parser


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
