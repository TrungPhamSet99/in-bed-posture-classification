# -*- coding: utf-8 -*-
# author: OSS (refer to github) and Trung Pham (EDABK lab - HUST)
# description: Script define utility functions used in pose classification module
import torch
import os
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
import pathlib
import torch.nn.functional as F
import torch.nn as nn 
from sklearn.metrics import confusion_matrix

SLP_DICT = {"Right Ankle": 0, "Right Knee": 1, "Right Hip": 2, "Left Hip": 3, "Left Knee": 4, "Left Ankle": 5, "Right Wrist": 6,
            "Right Elbow": 7, "Right Shoulder": 8, "Left Shoulder": 9, "Left Elbow": 10, "Left Wrist": 11, "Thorax": 12, "Head Top": 13}


def silu_activation(x):
    """
    Implement SiLU activation 
    Formula: SiLU(x) = x * sigmoid(x)

    Parameters
    ----------
    x: torch.Tensor
        Input value

    Returns
    ----------
    torch.Tensor
        Output of SiLU activation function
    """
    return x * torch.sigmoid(x)


def leaky_activation(x):
    """
    Implement Leaky-ReLU activation function
    Adjust alpha parameter to govern slope for values lower than threshold in ReLU function

    Parameters
    ----------
    x: torch.Tensor
        Input value

    Returns
    ----------
    torch.Tensor
        Output of Leaky ReLU activation function
    """
    activation = nn.LeakyReLU(negative_slope=0.1)
    return activation(x)

def to_onehot(inputs, num_classes):
    return np.squeeze(np.eye(num_classes)[inputs.reshape(-1)])

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def autopad(k, p=None, d=1):  
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def initialize_weights(model):
    """Init weight for model

    Parameters
    ----------
    model : nn.Module as Pytorch model
        Input model
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def count_params(model):
    """Count parameters of model (total and just trainable)

    Parameters
    ----------
    model : nn.Mopdule
        Pytorch model as nn.Module
    """
    total_params = sum(param.numel() for param in model.parameters())
    trainable_pararms = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return total_params, trainable_pararms

def combine_pose_embedding_and_autoencoder_output(pose_embedding, autoencoder_output):
    """Combine pose embedding and autoencoder output into one tensor

    Parameters
    ----------
    pose_embedding : torch.Tensor or numpy
        Pose embedding from HRNet
    autoencoder_output : torch.Tensor or numpy
        Image feature extracted from autoencoder
    """
    if isinstance(pose_embedding, np.ndarray):
        pose_embedding = torch.from_numpy(pose_embedding)
    elif not isinstance(pose_embedding, torch.Tensor):
        raise TypeError("Only support torch.Tensor or np.ndarray for 'pose_embedding' argument")
    
    if isinstance(autoencoder_output, np.ndarray):
        autoencoder_output = torch.from_numpy(autoencoder_output)
    elif not isinstance(autoencoder_output, torch.Tensor):
        raise TypeError("Only support torch.Tensor or np.ndarray for 'autoencoder_output' argument")
    
    assert pose_embedding.shape == torch.Size([2,22]), "Invalid shape for pose embedding"
    assert len(autoencoder_output.shape) == 3, "Output of autoencoder must be 3D tensor"
    autoencoder_output_size = autoencoder_output.shape[2]
    # Process pose embedding vector to get suitable shape 
    pad_size = (int((autoencoder_output_size-pose_embedding.shape[1])//2),
                int((autoencoder_output_size-pose_embedding.shape[1])//2 + 1))
    pose_embedding = F.pad(pose_embedding, pad_size, "constant", 0)
    pose_embedding = pose_embedding.repeat(int(autoencoder_output_size//pose_embedding.shape[0]),1)
    pose_embedding = torch.vstack((pose_embedding, pose_embedding[1,:])) # torch.Size([29,29])
    pose_embedding = torch.unsqueeze(pose_embedding, 0)
    pose_embedding = pose_embedding.repeat(int(autoencoder_output.shape[0]//4), 1, 1)
    # Concate pose embedding and return combined tensor
    return torch.cat((autoencoder_output, pose_embedding), dim=0)


def get_distance(lmk_from, lmk_to):
    # Return L2 distance
    return lmk_to - lmk_from


def get_distance_by_name(pose, name_from, name_to):
    lmk_from = pose[SLP_DICT[name_from], :]
    lmk_to = pose[SLP_DICT[name_to], :]
    return get_distance(lmk_from, lmk_to)


def get_center_points(pose, left_point, right_point):
    left = pose[SLP_DICT[left_point], :]
    right = pose[SLP_DICT[right_point], :]
    return left*0.5 + right*0.5


def get_pose_size(pose, torso_size_multiplier=2.5):
    """Get size of pose by measure distance from neck to middle of hip and multiple by 2.5

    Parameters
    ----------
    pose : np.ndarray
        pose as keypoints
    torso_size_multiplier : float, optional
        coefficient to multiple, by default 2.5

    Returns
    -------
    np.ndarray
        Pose size for normalization
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().detach().numpy()

    hips_center = get_center_points(pose, "Left Hip", "Right Hip")
    shoulder_center = get_center_points(
        pose, "Left Shoulder", "Right Shoulder")
    torso_size = np.linalg.norm(shoulder_center - hips_center)

    pose_center_new = get_center_points(pose, "Left Hip", "Right Hip")
    d = np.take(pose - pose_center_new, 0, axis=0)
    max_dis = np.amax(np.linalg.norm(d, axis=0))
    # length of body = torso_size * torso_size_multiplier
    pose_size = np.maximum(torso_size*torso_size_multiplier, max_dis)

    return pose_size


def normalize_pose(pose):
    """Normalize pose using torso size

    Parameters
    ----------
    pose : np.ndarray
        Input keypoints

    Returns
    -------
    np.ndarray
        Normalized pose
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().detach().numpy()
    pose_center = get_center_points(pose, "Left Hip", "Right Hip")
    pose = pose - pose_center
    pose_size = get_pose_size(pose)

    pose /= pose_size
    return pose


def build_embedding_from_distance(pose):
    # distance_list = []
    # for start_point in SLP_DICT:
    #     index = list(SLP_DICT.keys()).index(start_point)
    #     for i in range(index+1, len(SLP_DICT)):
    #         end_point = list(SLP_DICT.keys())[i]
    #         distance_list.append(get_distance_by_name(pose, start_point, end_point))
    # return np.asarray(distance_list)
    distance_embedding = np.array([
        get_distance_by_name(pose, "Left Shoulder", "Right Shoulder"),
        get_distance_by_name(pose, "Left Elbow", "Right Elbow"),
        get_distance_by_name(pose, "Left Wrist", "Right Wrist"),
        get_distance_by_name(pose, "Left Hip", "Right Hip"),
        get_distance_by_name(pose, "Left Knee", "Right Knee"),
        get_distance_by_name(pose, "Left Ankle", "Right Ankle"),
        get_distance_by_name(pose, "Thorax", "Left Wrist"),
        get_distance_by_name(pose, "Thorax", "Right Wrist")
    ])
    return distance_embedding



def pose_to_embedding_v1(pose):
    """Get embedding vector from raw pose from HRNet (version 1)

    Parameters
    ----------
    pose : np.ndarray or torch.Tensor
        raw pose from HRNet

    Returns
    -------
    torch.Tensor
        Embedding vector
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().detach().numpy()
    reshaped_inputs = np.reshape(pose, (14, 2))
    norm_inputs = normalize_pose(reshaped_inputs)
    return torch.from_numpy(norm_inputs.flatten())


def pose_to_embedding_v2(pose):
    """Get embedding vector from raw pose from HRNet (version 2)

    Parameters
    ----------
    pose : np.ndarray or torch.Tensor
        raw pose from HRNet

    Returns
    -------
    torch.Tensor
        Embedding vector
    """
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().detach().numpy()
    reshaped_input = np.reshape(pose, (14, 2))
    norm_input = normalize_pose(reshaped_input)
    distance_embedding = build_embedding_from_distance(norm_input)

    # return torch.from_numpy(np.concatenate((norm_input, distance_embedding), axis=0).flatten())
    embedding = torch.from_numpy(np.transpose(np.concatenate((norm_input, distance_embedding), axis=0))).flatten()
    # embedding = torch.from_numpy(np.transpose(norm_input)).flatten()
    return embedding

def load_config(path):
    """Load JSON or yaml config from path

    Parameters
    ----------
    path : str
        path to JSON or yaml config

    Returns
    -------
    dict or list
        JSON object
    """
    if isinstance(path, dict):
        return path
    elif pathlib.Path(path).suffix == ".json":
        return json.load(open(path, "r"))
    elif pathlib.Path(path).suffix == ".yaml":
        return yaml.safe_load(open(path, "r"))
    
    else:
        raise ValueError(f"Do not support config as {pathlib.Path(path).suffix} format")



def accuracy(outputs, labels):
    """Calculate accuracy based on prediction and labels

    Parameters
    ----------
    outputs : torch.Tensor or np.ndarray
        prediction values
    labels : torch.Tensor or np.ndarray
        label values

    Returns
    -------
    torch.Tensor
        accuracy value
    """
    _, preds = torch.max(outputs, dim=1)
    # _, labels = torch.max(labels, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          savepath="confusion_matrix.png"):
    """Plot and save confusion matrix for evaluation phase

    Parameters
    ----------
    y_true : np.ndarray
        labels 
    y_pred : np.ndarray
        prediction
    classes : list or np.ndarray
        list of class names
    normalize : bool, optional
        normalized acc to [0,1] range, by default False
    title : str, optional
        title on image, by default None
    cmap : tuple or list, optional
        color map, by default plt.cm.Blues
    savepath : str, optional
        path to save confusion matrix as image, by default "confusion_matrix.png"

    Returns
    -------
    _type_
        _description_
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Computing confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

        # Visualizing
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotating the tick labels and setting their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Looping over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    print("Saving confusion matrix...")
    plt.savefig(savepath)
    return ax


def visualize_keypoint(image, keypoint):
    pass

def colorstr(*input_):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input_ if len(input_) > 1 else ('blue', 'bold', input_[0])
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + '{}'.format(string) + colors['end']

