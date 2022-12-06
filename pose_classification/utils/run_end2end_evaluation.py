# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define end2end evaluator for pose classification module
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torchvision.transforms import ToTensor

from utils.general import load_config, accuracy, plot_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from data.end2end_dataset import End2EndDataset
from model.target_model import model_gateway, End2EndPoseClassifer
from utils.logger import Logger

warnings.filterwarnings('ignore')
CLASSES = ["supine_0", "supine_1", "supine_2", "lying_left_0", "lying_left_1", "lying_left_2",
           "lying_right_0", "lying_right_1", "lying_righ_2"]

class End2EndEvaluator:
    def __init__(self, config_path):
        """Constructor for End2EndEvaluator

        Parameters
        ----------
        config_path : str
            path to JSON config file
        """
        self.config = load_config(config_path)
        # model_config = self.config["model"]
        for k in self.config.keys():
            if "_model" not in k:
                continue
            setattr(self, k, model_gateway(
                self.config[k]["model_name"], load_config(self.config[k]["model_config"])))
        self.logger = Logger(self.config["log"]).logger
        self.end2end_evaluator = End2EndPoseClassifer(self.raw_model, self.supine_model,
                                                      self.lying_left_model, self.lying_right_model)
        self.test_dataset = End2EndDataset(self.config["data"]["data_dir"],
                                           self.config["data"]["mapping_file"],
                                           self.config["data"]["mode"])
        self.testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=32,
                                                      shuffle=False, num_workers=4,
                                                      pin_memory=False)

    def initialize(self):
        """Initialize for evaluator
        - Load weight for sub models
        - Init device (cpu or gpu)
        """
        for k in self.config:
            if "_model" not in k:
                continue
            sub_model = eval(f"self.{k}")
            sub_model.load_state_dict(torch.load(self.config[k]["weight"]))
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device('cpu')

    def run(self):
        """Run end2end evaluator on test set

        Returns
        -------
        tuple
            label, prediction
        """
        label = []
        prediction = []
        for i, batch in enumerate(self.testloader):
            print(f"Run evaluation {i}/{len(self.test_dataset)//32}", end="\r")
            poses, labels = batch
            output = self.end2end_evaluator(poses, self.device)
            if isinstance(output, torch.Tensor):
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
                        help='Path to training config file', default="./cfg/eval/end2end_config.json")
    return parser


def main():
    parser = parse_argument()
    args = parser.parse_args()

    end2end = End2EndEvaluator(args.config_path)
    end2end.initialize()
    labels, prediction = end2end.run()

    precision, recall, fscore, support = score(labels, prediction)
    report = classification_report(labels, prediction)
    logger = end2end.logger
    logger.info('\nprecision: {}'.format(precision))
    logger.info('\nrecall: {}'.format(recall))
    logger.info('\nfscore: {}'.format(fscore))
    logger.info('\nsupport: {}\n'.format(support))
    logger.info(report)

    plot_confusion_matrix(labels, prediction, CLASSES, normalize=False, title="Non-normalized confusion matrix (all)",
                          savepath="Non_normalized_confusion_matrix.png")
    plot_confusion_matrix(labels, prediction, CLASSES, normalize=True, title="Normalized confusion matrix (all)",
                          savepath="Normalized_confusion_matrix.png")


if __name__ == "__main__":
    main()
