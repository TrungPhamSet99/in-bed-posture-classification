# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define trainer for pose classification
import os
import json
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms import ToTensor, Compose
import torch.utils.data.dataloader
import itertools
from torch.utils.tensorboard import SummaryWriter
from model.target_model import model_gateway
from model.base_module import ConvBlock, TransposeConvBlock
from model.model_utils import count_params
from data.normal_dataset import NormalPoseDataset
from data.autoencoder_dataset import AutoEncoderDataset
from utils.general import load_config, accuracy, colorstr
from utils.focal_loss import FocalLoss


class Trainer:
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
        self.metric_config = self.config["metric"]
        # Get model from config
        self.model = model_gateway(self.config["model_name"], self.model_config)
        total_params, trainable_params = count_params(self.model)
        print(colorstr("Total params: "), total_params)
        print(colorstr("Trainable params: "), trainable_params)

        dataset = eval(self.data_config["dataset_name"])
        self.train_dataset = dataset(self.data_config['data_dir'],
                                     self.data_config['train_list'],
                                     augment_config_path=self.data_config['augmentation_config_path'],
                                     transform = Compose([eval(self.data_config["train_transform"])()]))
        self.test_dataset = dataset(self.data_config['data_dir'],
                                    self.data_config['test_list'],
                                    augment_config_path=self.data_config['augmentation_config_path'],
                                    transform = Compose([eval(self.data_config["test_transform"])()]))
        self.loss_calculate = eval(self.config["loss"])()
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
        # Init tensorboard writer
        # if os.path.exists(os.path.join(self.training_config["output_dir"], "run")):
        #     os.rmdir(os.path.join(self.training_config["output_dir"], "run"))
        self.writer = SummaryWriter(os.path.join(self.training_config["output_dir"], "run"))
        self._init_torch_tensor()
        print("Successfully init trainer")
    
    def _init_torch_tensor(self):
        """Init torch tensor for device
        """
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device(self.config["device"])
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
    
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
        print("Start to train model")
        loss_report = list()
        best_lost = np.inf
        best_acc = -np.inf
        for epoch in range(self.training_config['epoch']):
            if not epoch % self.training_config['saving_interval']:
                model_name = "{}_epoch.pth".format(epoch)
            for i, batch in enumerate(tqdm(self.trainloader, desc=f"Epoch [{epoch}]")):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._train_step(batch)
                loss = torch.mean(loss)
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('Loss/Train/Iteration', loss, epoch*(len(self.train_dataset)//self.data_config["batch_size"]) + i)
                
            result = self._evaluate(self.testloader, epoch)
            self._epoch_end(epoch, result)
            loss_report.append(result)
            if result['val_loss'] < best_lost:
                best_lost = result["val_loss"]
                self._save_model("best_loss_model.pth")
            if "acc" in self.metric_config:
                if result['val_acc'] > best_acc:
                    best_acc = result["val_acc"]
                    self._save_model("best_acc_model.pth")
        self.writer.flush()
        self.writer.close()
        self._save_model("final.pth")
        with open(os.path.join(self.training_config['output_dir'], self.training_config["loss_report_path"]), "w") as f:
            json.dump(loss_report, f)
        return loss_report

    def _train_step(self, batch):
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
        print(inputs)
        if torch.cuda.is_available():
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        out = self.model(inputs)
        loss = self.loss_calculate(out, labels)
        return loss

    def _validation_step(self, batch, epoch, idx):
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
        print(inputs)
        if torch.cuda.is_available():
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        out = self.model(inputs)
        loss = self.loss_calculate(out, labels)
        self.writer.add_scalar("Loss/Val/Iteration", loss, epoch*(len(self.test_dataset)//self.data_config["batch_size"]) + idx)
        if "acc" in self.metric_config:
            acc = accuracy(out, labels)
            self.writer.add_scalar("Acc/Val/Iteration", acc, epoch*(len(self.test_dataset)//self.data_config['batch_size']) + idx)
            return {'val_loss': loss, 'val_acc': acc}
        else:
            return {'val_loss': loss}

    def _validation_epoch_end(self, outputs, epoch):
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
        self.writer.add_scalar("Loss/Val/Epoch", epoch_loss, epoch)
        if "acc" in self.metric_config:
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()
            self.writer.add_scalar("Acc/Val/Epoch", epoch_acc, epoch)
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        else:
            return {'val_loss': epoch_loss.item()}

    def _evaluate(self, dataloader, epoch):
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
        self.model.eval()
        with torch.no_grad():
            outputs = [self._validation_step(batch, epoch, idx) for idx, batch in enumerate(dataloader)]
        self.model.train()
        return self._validation_epoch_end(outputs, epoch)

    def _epoch_end(self, epoch, result):
        """Print result at the end of epoch

        Parameters
        ----------
        epoch : int
            epoch index
        result : dict
            a dict contain result (acc and loss)
        """
        if "acc" in self.metric_config:
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['val_loss'], result['val_acc']))
        else:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))

    def _save_model(self, model_name):
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
