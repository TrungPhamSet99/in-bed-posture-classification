import torch
import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from baseline.config import dataset_config, train_config
from data.normal_dataset import NormalPoseDataset
from torchvision.transforms import ToTensor, Compose

import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from model.efficientnet import EfficientNet
from utils.focal_loss import FocalLoss
from model.base_module import LinearBlock
from utils.general import adjust_learning_rate
from utils.general import load_config, accuracy, plot_confusion_matrix, colorstr, accuracy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from model.target_model import PoseClassifierV2_1
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

# CLASSES = ["1", "2", "3"]

class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained("efficientnet-b0")
        self.dropout1 = nn.Dropout(0.4)
        self.linear = LinearBlock(1280, 9)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        # print(inputs.shape)
        inputs = inputs.float()
        feature = self.feature_extractor.extract_features(inputs)
        return self.linear(feature)
    

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.linear1 = nn.Linear(56, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 3)
    
    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs = F.relu(self.linear1(inputs))
        inputs = F.relu(self.linear2(inputs))
        return self.linear3(inputs)
    
    
def get_baseline_model(model_name: str):
    if model_name == "efficientnet":
        return EfficientNetB0()
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        num_classes = 3
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name == "vgg16":
        vgg16 = models.vgg16(pretrained=False)
        num_classes = 9
        in_features = vgg16.classifier[-1].in_features
        vgg16.classifier[-1] = nn.Linear(in_features, num_classes)
        return vgg16
    else:
        raise ValueError(f"Do not support model {model_name}")
    

def get_leg_pose_classification():
    cfg = json.load(open("../../cfg/model_config/pose_classifier_v2.1.json"))
    model = PoseClassifierV2_1(cfg, "PoseClassifierV2_1")
    return model


def build_dataloader(data_cfg):
    # Build train/test dataloader
    train_dataset = NormalPoseDataset(data_cfg['data_dir'],
                                      data_cfg['train_list'],
                                      data_cfg['mapping_file_train'],
                                      data_cfg['image_dir'],
                                      classes=CLASSES,
                                      augment_config_path=data_cfg['augmentation_config_path'],
                                      transform = Compose([eval(data_cfg["test_transform"])()]))
    test_dataset = NormalPoseDataset(data_cfg['data_dir'],
                                     data_cfg['test_list'],
                                     data_cfg['mapping_file_test'],
                                     data_cfg['image_dir'],
                                     classes=CLASSES,
                                     transform = Compose([eval(data_cfg["test_transform"])()]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=data_cfg['batch_size'],
                                                   shuffle=data_cfg['shuffle'],
                                                   num_workers=data_cfg['num_workers'],
                                                   pin_memory=data_cfg['pin_memory'])
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=data_cfg['batch_size'],
                                                   shuffle=data_cfg['shuffle'],
                                                   num_workers=data_cfg['num_workers'],
                                                   pin_memory=data_cfg['pin_memory'])
    return train_dataloader, test_dataloader

def train(model, train_dataloader, test_dataloader, num_epochs=50, use_pose=False):
    loss_report = []
    best_loss = np.inf
    best_acc = -np.inf
    # define loss and optimizer
    train_criterion = LabelSmoothingCrossEntropy(0.05)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], 0.1)   
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # train
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        adjust_learning_rate(optimizer, epoch, 0.001)
        for i, batch in enumerate(tqdm(train_dataloader)):
            image, pose, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            if use_pose:
                loss = train_step(model, pose, label, train_criterion)
            else:
                loss = train_step(model, image, label, train_criterion)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Train Loss: {loss}')
        results = validate(model, test_dataloader, train_criterion, use_pose)
        epoch_end(epoch, results)
        loss_report.append({"Train_loss": loss, "Val_loss": results["val_loss"]})
        if results['val_loss'] < best_loss:
            best_loss = results['val_loss']
            save_model(model, train_config["output_dir"], "best_loss_model.pth")
        if results['val_acc'] > best_acc:
            best_acc = results['val_acc']
            save_model(model, train_config["output_dir"], "best_acc_model.pth")

    with open(os.path.join(train_config['output_dir'], "loss_report.json"), "w") as f:
            json.dump(loss_report, f)
    print(colorstr("Final test"))
    # Final test
    model.eval()
    correct = 0
    total = 0
    prediction = []
    labels = []
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            image, pose, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            if use_pose:
                output = model(pose)
            else:
                output = model(image)
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()

            label = label.cpu().numpy()
            labels += label.tolist()
            output = torch.argmax(output, axis=1)
            prediction += output.cpu().numpy().tolist()
        
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    precision, recall, fscore, support = score(labels, prediction)
    report = classification_report(labels, prediction)
    print('\nprecision: {}'.format(precision))
    print('\nrecall: {}'.format(recall))
    print('\nfscore: {}'.format(fscore))
    print('\nsupport: {}\n'.format(support))
    print(report)

    plot_confusion_matrix(labels, prediction, ["1", "2", "3"], normalize=False, title="Non-normalized confusion matrix (all)",
                        savepath=f"non_normalize.png")
    plot_confusion_matrix(labels, prediction, ["1", "2", "3"], normalize=True, title="Normalized confusion matrix (all)",
                        savepath=f"normalize.png")
    print("------------------------------------------------------")


def save_model(model, output_dir, filename):
    print(f"Info: Save checkpoint named {filename}")
    save_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), save_path)

def train_step(model, input, label, criterion):
    model.train()
    # print(input[:, :, 19])
    out = model(input)
    loss = criterion(out, label)
    return loss 

def validate(model, test_loader, criterion, use_pose):
    model.eval()
    with torch.no_grad():
        outputs = [validate_step(model, batch, criterion, use_pose) for batch in test_loader]
    
    return validate_epoch_end(outputs)

def validate_step(model, batch, criterion, use_pose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image, pose, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    if use_pose:
        out = model(pose)
    else:
        out = model(image)
    loss = criterion(out, label)
    acc = accuracy(out, label)
    return {"val_loss": loss, "val_acc": acc}

def validate_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end(epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['val_loss'], result['val_acc']))

if __name__ == "__main__":
    # Create output directory
    if not os.path.isdir(train_config['output_dir']):
        os.makedirs(train_config['output_dir'])
    # Prepare model
    baseline_model = get_baseline_model("resnet18")
    leg_pose_model = get_leg_pose_classification()
    # Prepare dataset
    train_dataloader, test_dataloader = build_dataloader(dataset_config)
    # train(LinearClassifier(), train_dataloader, test_dataloader, num_epochs=100, use_pose=True)
    train(baseline_model, train_dataloader, test_dataloader, num_epochs=100)


