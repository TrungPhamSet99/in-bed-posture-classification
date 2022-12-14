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
from sklearn.metrics import confusion_matrix

SLP_DICT = {"Right Ankle": 0, "Right Knee": 1, "Right Hip": 2, "Left Hip": 3, "Left Knee": 4, "Left Ankle": 5, "Right Wrist": 6,
            "Right Elbow": 7, "Right Shoulder": 8, "Left Shoulder": 9, "Left Elbow": 10, "Left Wrist": 11, "Thorax": 12, "Head Top": 13}


def get_distance(lmk_from, lmk_to):
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
    return torch.from_numpy(np.transpose(np.concatenate((norm_input, distance_embedding), axis=0)))


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

