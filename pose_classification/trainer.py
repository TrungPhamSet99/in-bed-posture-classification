from model import PoseClassifierV1, model_gateway
from data import NormalPoseDataset
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.transforms import ToTensor
import torch.utils.data.dataloader
from utils import load_config, accuracy
from focal_loss import FocalLoss
import os
import json 
import numpy as np
import argparse 

def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser("Run script to train pose classification model")
    parser.add_argument('--config-path', type=str, 
                        help='Path to training config file', default="./cfg/train_config.json")
    return parser

def init_weights(m):
    if type(m) in [nn.Module, nn.Linear, nn.Conv1d]:
        nn.init.xavier_uniform(m.weight)


class PoseTrainer():
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        self.model_config = self.config['model']
        self.optim_config = self.config['optimizer']
        self.training_config = self.config['training']

        self.model = model_gateway(self.config)
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
        self.model.apply(init_weights)
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

    def run_train(self):
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
        inputs, labels = batch
        # print(inputs.shape)
        out = self.model(inputs, self.device)
        # print(f"Train output: {out}")
        # print(f"Train label: {labels}")
        # loss = F.cross_entropy(out, labels)
        loss = self.loss_calculate(out, labels)
        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        out = self.model(inputs, self.device)
        # loss = F.cross_entropy(out, labels)
        loss = self.loss_calculate(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}


    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def evaluate(self, dataloader):
        """Evaluate the model's performance on the validation set"""
        outputs = [self.validation_step(batch) for batch in dataloader]
        return self.validation_epoch_end(outputs)

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    def save_model(self, model_name):
        print('\n*INFO: Saving model...*\n')
        save_path = os.path.join(self.training_config['output_dir'], model_name)
        torch.save(self.model.state_dict(), save_path) 

def main():
    parser = parse_argument()
    args = parser.parse_args()

    trainer = PoseTrainer(args.config_path)
    trainer.initialize()
    loss_report=trainer.run_train()
    # with open("loss_report.json", "w") as f:
    #     json.dump({"loss_report": loss_report}, f)

if __name__ == "__main__":
    main()