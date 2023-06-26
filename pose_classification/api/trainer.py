# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define trainer for pose classification
import os
import json
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
from tqdm import tqdm
from torchvision.transforms import ToTensor, Compose
import torch.utils.data.dataloader
import itertools
from torch.utils.tensorboard import SummaryWriter
from model.target_model import model_gateway
from model.base_module import ConvBlock, TransposeConvBlock
from data.normal_dataset import NormalPoseDataset
from utils.general import load_config, accuracy, colorstr, count_params, adjust_learning_rate
from utils.focal_loss import FocalLoss
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lr_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


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
        self.training_config = self.config['training']
        self.metric_config = self.config["metric"]
        self.backbone = self.config['backbone']

        if self.backbone == "SwinTransformer":
            self.swin = True 
            self.model, self.swin_config = model_gateway(self.config["model_name"], self.model_config)
            self.optim_config = self.swin_config
            self.scheduler_config = self.swin_config
        else:  
            self.swin = False
            self.model = model_gateway(self.config["model_name"], self.model_config)
            self.optim_config = self.config['optimizer']
            self.scheduler_config = {"lr_stpe": self.optim_config['lr_step'],
                                     "lr_factor": self.optim_config['lr_factor']}
        # Get model from config
        
        total_params, trainable_params = count_params(self.model)
        
        dataset = eval(self.data_config["dataset_name"])
        # Build dataloader
        self.train_dataset = dataset(self.data_config['data_dir'],
                                     self.data_config['train_list'],
                                     self.data_config['mapping_file_train'],
                                     self.data_config['image_dir'],
                                     augment_config_path=self.data_config['augmentation_config_path'],
                                     transform = Compose([eval(self.data_config["train_transform"])()]))
        self.test_dataset = dataset(self.data_config['data_dir'],
                                    self.data_config['test_list'],
                                    self.data_config['mapping_file_test'],
                                    self.data_config['image_dir'],
                                    transform = Compose([eval(self.data_config["test_transform"])()]))
        self.loss_calculate = eval(self.config["loss"])(0.05)
        self.loss_scaler = NativeScalerWithGradNormCount()
        self.trainloader = None
        self.testloader = None
        self.optimizer = None
        self.scheduler = None

        print(colorstr("Total params: "), total_params)
        print(colorstr("Trainable params: "), trainable_params)



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
        # print(self.model)
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
        
        if self.backbone == "SwinTransformer":
            self.optimizer = build_optimizer(self.optim_config, self.model, is_pretrain=True, transformer=self.swin)
            self.scheduler = build_lr_scheduler(self.scheduler_config, self.optimizer,
                                                len(self.trainloader) // self.swin_config.TRAIN.ACCUMULATION_STEPS,
                                                self.swin)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.optim_config['lr'],
                                             momentum=self.optim_config['momentum'],
                                             weight_decay=self.optim_config['weight_decay'])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.optim_config['lr_step'], 
                                                                  self.optim_config['lr_factor'])                           

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
            adjust_learning_rate(self.optimizer, epoch, self.optim_config['lr'])
            if not epoch % self.training_config['saving_interval']:
                model_name = "{}_epoch.pth".format(epoch)
            if self.backbone == "SwinTransformer":
                self.train_one_epoch(epoch)
            else:
                for i, batch in enumerate(tqdm(self.trainloader, desc=f"Epoch [{epoch}]")):
                    self.optimizer.zero_grad()
                    loss = self._train_step(batch)
                    loss = torch.mean(loss)
                    loss.backward()
                    self.optimizer.step()
                    self.writer.add_scalar('Loss/Train/Iteration', loss, epoch*(len(self.train_dataset)//self.data_config["batch_size"]) + i)
                print("Train loss: ", loss)

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
            # self.optimizer.zero_grad()
            # self.scheduler.step_update((epoch * 326 + idx) // config.TRAIN.ACCUMULATION_STEPS)

        self.writer.flush()
        self.writer.close()
        self._save_model("final.pth")
        with open(os.path.join(self.training_config['output_dir'], self.training_config["loss_report_path"]), "w") as f:
            json.dump(loss_report, f)
        return loss_report

    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        num_steps = len(self.trainloader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()
        start = time.time()
        end = time.time()
        for idx, (samples, targets) in enumerate(tqdm(self.trainloader, desc=f"Epoch [{epoch}]")):
            if torch.cuda.is_available():
                for _input in samples:
                    _input = _input.float()
                    _input = _input.to(self.device)
                targets = targets.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.swin_config.AMP_ENABLE):
                outputs = self.model(samples)
            loss = self.loss_calculate(outputs, targets)
            loss = loss / self.swin_config.TRAIN.ACCUMULATION_STEPS

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=self.swin_config.TRAIN.CLIP_GRAD,
                                    parameters=self.model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % self.swin_config.TRAIN.ACCUMULATION_STEPS == 0)
            if (idx + 1) % self.swin_config.TRAIN.ACCUMULATION_STEPS == 0:
                self.optimizer.zero_grad()
                self.scheduler.step_update((epoch * num_steps + idx) // self.swin_config.TRAIN.ACCUMULATION_STEPS)

            loss_scale_value = self.loss_scaler.state_dict()["scale"]
            loss_meter.update(loss.item(), targets.size(0))
            if grad_norm is not None:  # loss_scaler return None if not update
                norm_meter.update(grad_norm)
            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx == num_steps:
                lr = self.optimizer.param_groups[0]['lr']
                wd = self.optimizer.param_groups[0]['weight_decay']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                print(
                    f'Train: [{epoch}/{self.swin_config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
        epoch_time = time.time() - start
        print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

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
        self.model.train()
        inputs, labels = batch
        # print(inputs.shape)
        if torch.cuda.is_available():
            for _input in inputs:
                _input = _input.float()
                _input = _input.to(self.device)
            labels = labels.to(self.device)
        out = self.model(inputs)
        if self.swin:
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            grad_norm = self.loss_scaler(loss, optimizer, clip_grad=self.optim_config.TRAIN.CLIP_GRAD,
                                         parameters=self.model.parameters(), create_graph=is_second_order,
                                         update_grad=(idx + 1) % self.optim_config.TRAIN.ACCUMULATION_STEPS == 0)
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
        criterion = nn.CrossEntropyLoss()
        inputs, labels = batch
        if torch.cuda.is_available():
            for _input in inputs:
                _input = _input.float()
                _input = _input.to(self.device)
            labels = labels.to(self.device)
        out = self.model(inputs)
        loss = criterion(out, labels)
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
