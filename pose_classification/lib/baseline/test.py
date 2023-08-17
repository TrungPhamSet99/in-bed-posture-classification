import os 
import json
import argparse
import torch
from torchvision.transforms import ToTensor, Compose
from data.normal_dataset import NormalPoseDataset
from baseline.train import get_baseline_model, build_dataloader
from baseline.config import dataset_config, test_config
from utils.general import load_config, accuracy, plot_confusion_matrix, colorstr, accuracy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score

CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

def prepare_model(test_config):
    model = get_baseline_model("resnet18")
    weight_path = test_config['weight_path']
    model.load_state_dict(torch.load(weight_path))
    return model

def build_dataloader(data_cfg):
    test_dataset = NormalPoseDataset(data_cfg['data_dir'],
                                     data_cfg['test_list'],
                                     data_cfg['mapping_file_test'],
                                     data_cfg['image_dir'],
                                     classes=CLASSES,
                                     transform = Compose([eval(data_cfg["test_transform"])()]),
                                     load_from_gt=False)
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=data_cfg['batch_size'],
                                                  shuffle=data_cfg['shuffle'],
                                                  num_workers=data_cfg['num_workers'],
                                                  pin_memory=data_cfg['pin_memory'])
    return test_dataloader

def test(model, test_dataloader, use_pose=False, device="cuda"):
    model.to(device)
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

if __name__ == "__main__":
    model = prepare_model(test_config)
    dataloader = build_dataloader(dataset_config)
    test(model, dataloader)