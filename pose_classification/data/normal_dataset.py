# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define dataset used for pose classification inherited from torch.utils.data.Dataset
import cv2
import copy
import random
import os
import json
import numpy as np
import torch

from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from utils.general import pose_to_embedding_v2
from data.augmentation import Augmentor

class NormalPoseDataset(Dataset):
    RGB_MEAN = [0.18988903, 0.18988903, 0.18988903]
    RGB_STD = [0.09772425, 0.09772425, 0.09772425]
    NORMALIZE = Compose([Normalize(mean=RGB_MEAN, std=RGB_STD)])
    def __init__(self, data_dir, list_path, mapping_file, image_dir, augment_config_path=None, transform=None):
        """Constructor for NormalPoseDataset

        Parameters
        ----------
        data_dir : str
            Root directory for data
        list_path : str
            Path to text file contain list of data samples
        augment_config_path : str, optional
            Path to data augmentation config file, by default None
        transform : callable object, optional
            Optional transform to be applied, by default None
        """
        self.data_root = data_dir
        self.data_list_path = list_path
        self.transform = transform
        self.mapping = json.load(open(mapping_file, "r"))
        self.image_keys = list(self.mapping.keys())
        self.image_dir = image_dir
        if "train" in self.data_list_path:
            self.mode = "train"
        else:
            self.mode = "test"
        if isinstance(list_path, str):
            self.data_paths = open(self.data_list_path).readlines()
            self.data_paths = [ele.rstrip() for ele in self.data_paths]
        elif isinstance(list_path, list):
            self.data_paths = list_path

        # if "supine" in self.data_list_path:
        #     self.classes = ["1", "2", "3"]
        # elif "lying_left" in self.data_list_path:
        #     self.classes = ["4", "5", "6"]
        # elif "lying_right" in self.data_list_path:
        #     self.classes = ["7", "8", "9"]
        # else:
        #     self.classes = ["lying_left", "supine", "lying_right"]
        self.classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        if augment_config_path is not None:
            self.augmentor = Augmentor(augment_config_path)
        else:
            self.augmentor = None
        self.image_keys, self.dis_of_samples = self.filter_image_keys()
        print(self.dis_of_samples)

        
    def __len__(self):
        """Get length of dataset

        Returns
        -------
        int
            length of dataset
        """
        return len(self.image_keys)

    def filter_image_keys(self):
        output = []
        dis_of_samples = 9*[0]
        for sample in self.image_keys:
            c = self.mapping[sample].split("/")[-2]
            if c in self.classes:
                index = self.classes.index(c)
                dis_of_samples[index] += 1
                output.append(sample)
        return output, dis_of_samples

    def __getitem__(self, idx):
        """Get data items by index

        Parameters
        ----------
        idx : int
            index

        Returns
        -------
        _type_
            _description_
        """
        image_file = self.image_keys[idx]    
        path = self.mapping[image_file]
        c = path.split("/")[-2]
        pose_fp = os.path.join(self.data_root, path)
        image_fp = os.path.join(self.image_dir, image_file)
        if self.mode == "train":
            image_fp = image_fp.replace("train_", "slp_train/")
        else:
            image_fp = image_fp.replace("test_", "slp_val/")
        pose = np.load(pose_fp)
        image = cv2.imread(image_fp)
        if self.augmentor is not None:
            image, pose = self.augmentor(image, pose, image_file)
        image = cv2.resize(image, (180, 180))
        pose = pose_to_embedding_v2(pose)
        if self.transform is not None:
            image = self.transform(image)
        image = self.NORMALIZE(image)


        return (pose, image), self.classes.index(c)

def batch_mean_and_std(loader):
    cnt = 0
    fst_moment = torch.empty(40)
    snd_moment = torch.empty(40)
    for tensor, _ in loader:
        print(torch.max(tensor))
        print(torch.min(tensor))
        print("--------------------------------")
        b, c, h, w = tensor.shape
        nb_elements = b * h * w
        sum_ = torch.sum(tensor, dim=[0,2,3])
        sum_of_square = torch.sum(tensor ** 2, dim=[0,2,3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt+nb_elements)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_elements)
    
        cnt += nb_elements
    
    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean, std


if __name__ == "__main__":
    # Implement function to calculate mean and std devitation of dataset
    mean_path = "./cfg/mean.npy"
    std_path = "./cfg/std.npy"
    mean = np.load(open(mean_path, "rb")).tolist()
    std = np.load(open(std_path, "rb")).tolist()
    print(len(mean), len(std))
    data_dir = "/data2/samba/public/TrungPQ/22B/in-bed-posture-classification/pose_classification/combined_feature"
    list_path = "/data2/samba/public/TrungPQ/22B/in-bed-posture-classification/pose_classification/combined_feature/lying_left_train_list.txt"
    augment_config_path = None
    transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])
    dataset = NormalPoseDataset(data_dir, list_path, augment_config_path, transform)
    loader = DataLoader(dataset, batch_size=2, num_workers=1)
    mean, std = batch_mean_and_std(loader)
    