# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define trainer for pose classification
import os
import json
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torch.utils.data.dataloader

from model.target_model import model_gateway
from data.normal_dataset import NormalPoseDataset
from utils.general import load_config, accuracy, colorstr
from utils.focal_loss import FocalLoss


class PoseTrainer():
    def __init__(self, config_path):
        """
        Init trainer class

        Parameters
        ----------
        config_path : dict
            Config for trainer as dictionary
        """
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        self.model_config = load_config(self.config['model_config'])
        self.optim_config = self.config['optimizer']
        self.training_config = self.config['training']

        self.model = model_gateway(self.config["model_name"], self.model_config)
        self.train_dataset = NormalPoseDataset(self.data_config['data_dir'],
                                               self.data_config['train_list'],
                                               augment_config_path=self.data_config['augmentation_config_path'])
        self.test_dataset = NormalPoseDataset(self.data_config['data_dir'],
                                              self.data_config['test_list'])
        self.device = self.config["device"]
        self.loss_calculate = FocalLoss()
        self.trainloader = None
        self.testloader = None
        self.optimizer = None
        self.scheduler = None

    def initialize(self):
        """
            Init some hyper parameter such as optimizer, scheduler and create data loader for trainer
        Raises
        ------
        ValueError
            Raise ValueError if config for optimizer is not SGD
        """
        print(colorstr("optimizer hyperparameter:\n") + ', '.join(f'{k}: {v}'for k, v in self.config['optimizer'].items()))
        print(colorstr(f"Number sample for training set: {len(self.train_dataset)}"))
        print(colorstr(f"Number sample for test set: {len(self.test_dataset)}"))
        self.model.apply(self.init_weights)
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.data_config['batch_size'],
                                                       shuffle=self.data_config['shuffle'],
                                                       num_workers=self.data_config['num_workers'],
                                                       pin_memory=self.data_config['pin_memory']
                                                       )
        self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                      batch_size=self.data_config['batch_size'],
                                                      shuffle=self.data_config['shuffle'],
                                                      num_workers=self.data_config['num_workers'],
                                                      pin_memory=self.data_config['pin_memory']
                                                      )
        if self.optim_config['name'] == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.optim_config['lr'],
                                             momentum=self.optim_config['momentum'],
                                             weight_decay=self.optim_config['weight_decay'])
        else:
            raise ValueError("Only support SGD optimizer")

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.optim_config['lr_step'], self.optim_config['lr_factor'])

        if not os.path.exists(self.training_config['output_dir']):
            os.makedirs(self.training_config['output_dir'], exist_ok=True)
        print("Successfully init trainer")
        
    
    @staticmethod
    def init_weights(m):
        if type(m) in [nn.Module, nn.Linear, nn.Conv1d]:
            nn.init.xavier_uniform(m.weight)

    def run_train(self):
        """
        Main function to run train

        Returns
        -------
        list
            Loss report
        """
        print("Start to train pose classification model")
        loss_report = list()
        best_lost = np.inf
        best_acc = -np.inf
        for epoch in range(self.training_config['epoch']):
            print("-----------------{} epoch-----------------".format(epoch))
            if not epoch % self.training_config['saving_interval']:
                model_name = "{}_epoch.pth".format(epoch)
                # self.save_model(model_name)
            for batch in self.trainloader:
                loss = self.train_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            result = self.evaluate(self.testloader)
            self.epoch_end(epoch, result)
            loss_report.append(result)
            if result['val_loss'] < best_lost:
                best_lost = result["val_loss"]
                self.save_model("best_loss_model.pth")
            if result['val_acc'] > best_acc:
                best_acc = result["val_acc"]
                self.save_model("best_acc_model.pth")

        self.save_model("final.pth")
        with open(os.path.join(self.training_config['output_dir'], self.training_config["loss_report_path"]), "w") as f:
            json.dump(loss_report, f)
        return loss_report

    def train_step(self, batch):
        """Run train for a step corresponding to a batch

        Parameters
        ----------
        batch : tuple
            A data bacth

        Returns
        -------
        float
            Loss value
        """
        inputs, labels = batch
        # print(inputs.shape)
        out = self.model(inputs, self.device)
        # print(f"Train output: {out}")
        # print(f"Train label: {labels}")
        # loss = F.cross_entropy(out, labels)
        loss = self.loss_calculate(out, labels)
        return loss

    def validation_step(self, batch):
        """Validate model for each step (batch)

        Parameters
        ----------
        batch : tuple
            A data batch for validation

        Returns
        -------
        dict
            {val loss, val acc}
        """
        inputs, labels = batch
        out = self.model(inputs, self.device)
        # loss = F.cross_entropy(out, labels)
        loss = self.loss_calculate(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        """Calculate everage loss and acc on validation dataset

        Parameters
        ----------
        outputs : list
            list of dict with elements are epoch outputs

        Returns
        -------
        dict
            {avg val loss, avg val acc}
        """
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def evaluate(self, dataloader):
        """Evaluate model on a dataset

        Parameters
        ----------
        dataloader : torch dataloader
            A dataloader in torch

        Returns
        -------
        dict
            {loss on dataset, acc on dataset}
        """
        outputs = [self.validation_step(batch) for batch in dataloader]
        return self.validation_epoch_end(outputs)

    def epoch_end(self, epoch, result):
        """Print result at the end of epoch

        Parameters
        ----------
        epoch : int
            epoch index
        result : dict
            a dict contain result (acc and loss)
        """
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['val_loss'], result['val_acc']))

    def save_model(self, model_name):
        """Save model as .pth file

        Parameters
        ----------
        model_name : str
            model name file
        """
        print('\n*INFO: Saving model...*\n')
        save_path = os.path.join(
            self.training_config['output_dir'], model_name)
        torch.save(self.model.state_dict(), save_path)

