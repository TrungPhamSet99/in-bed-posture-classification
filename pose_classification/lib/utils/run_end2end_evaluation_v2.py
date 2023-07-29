import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Compose
from utils.general import load_config, accuracy, plot_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from data.normal_dataset import NormalPoseDataset 
from model.target_model import model_gateway, End2EndPoseClassifer
from utils.logger import Logger

CLASSES = ["1", "2", "3"]

class Evaluator:
    def __init__(self, config_path):
        """Constructor for End2EndEvaluation
        """
        self.conf = load_config(config_path)
        self.model, self.swin_config = model_gateway(self.conf["model_name"], self.conf["model_config"])
        if self.conf["weight"]:
            # Load weight for evaluation
            self.model.load_state_dict(torch.load(self.conf["weight"]))
        self.data_config = self.conf["data"]
        self.dataset = eval(self.data_config["dataset_name"])
       
        
    def initialize(self):
        """Initialize for evaluator
        - Load weight for sub models
        - Init device (cpu or gpu)
        """
        # Build evaluation dataset
        self.test_dataset = self.dataset(self.data_config['data_dir'],
                                    self.data_config['test_list'],
                                    self.data_config['mapping_file_test'],
                                    self.data_config['image_dir'],
                                    transform = Compose([eval(self.data_config["test_transform"])()]))
        self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                      batch_size=self.data_config['batch_size'],
                                                      shuffle=self.data_config['shuffle'],
                                                      num_workers=self.data_config['num_workers'],
                                                      pin_memory=self.data_config['pin_memory']
                                                      )
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.model = self.model.to(self.device)
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
        # self.model.eval()
        # with torch.no_grad():
        for i, batch in enumerate(self.testloader):
            print(f"Run evaluation {i}/{len(self.test_dataset)//32}", end="\r")
            inputs, labels = batch
            output = self.model(inputs)
            if isinstance(output, torch.Tensor):
                output = torch.argmax(output, axis=1).cpu().numpy()
            labels = labels.numpy()
            label += labels.tolist()
            prediction += output.tolist()

        return label, prediction


def main():
    config = "./cfg/eval/end2end_config_v2.json"
    evaluator = Evaluator(config)
    evaluator.initialize()
    labels, prediction = evaluator.run()

    precision, recall, fscore, support = score(labels, prediction)
    report = classification_report(labels, prediction)
    print('\nprecision: {}'.format(precision))
    print('\nrecall: {}'.format(recall))
    print('\nfscore: {}'.format(fscore))
    print('\nsupport: {}\n'.format(support))
    print(report)

    plot_confusion_matrix(labels, prediction, CLASSES, normalize=False, title="Non-normalized confusion matrix (all)",
                          savepath="Non_normalized_confusion_matrix.png")
    plot_confusion_matrix(labels, prediction, CLASSES, normalize=True, title="Normalized confusion matrix (all)",
                          savepath="Normalized_confusion_matrix.png")
if __name__ == "__main__":
    main()