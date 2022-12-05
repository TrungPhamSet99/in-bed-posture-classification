# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define dataset used for pose classification inherited from torch.utils.data.Dataset
import copy
import logging
import random
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import pose_to_embedding_v1, pose_to_embedding_v2
from augmentation import PoseAugmentor


class NormalPoseDataset(Dataset):
    def __init__(self, data_dir, list_path, augment_config_path=None, transform=None):
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
        if isinstance(list_path, str):
            self.data_paths = open(self.data_list_path).readlines()
            self.data_paths = [ele.rstrip() for ele in self.data_paths]
        elif isinstance(list_path, list):
            self.data_paths = list_path
        if "supine" in self.data_list_path:
            self.classes = ["1", "2", "3"]
        elif "lying_left" in self.data_list_path:
            self.classes = ["4", "5", "6"]
        elif "lying_right" in self.data_list_path:
            self.classes = ["7", "8", "9"]
        else:
            self.classes = ["lying_left", "supine", "lying_right"]
        if augment_config_path is not None:
            self.augmentor = PoseAugmentor(augment_config_path)
        else:
            self.augmentor = None

    def __len__(self):
        """Get length of dataset

        Returns
        -------
        int
            length of dataset
        """
        return len(self.data_paths)

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
        path = self.data_paths[idx]
        c = path.split("/")[-2]
        fp = os.path.join(self.data_root, path)
        pose = np.load(fp)
        if self.augmentor is not None:
            pose = self.augmentor.augment(pose)
        embedding = pose_to_embedding_v2(pose)
        return embedding, self.classes.index(c)


class End2EndDataset(Dataset):
    def __init__(self, data_dir, mapping_file_path, mode="all", transform=None):
        """Constructor for End2EndDataset

        Parameters
        ----------
        data_dir : str or list
            root path to data directory
        mapping_file_path : str
            path to mapping file as JSON file
        mode : str, optional
            mode to build dataset, mode must be in ["all", "uncover", "cover1", "cover2"], by default "all"
        transform : callable object, optional
            Optional transform to be applied, by default None
        """
        # mode will be in ["all", "uncover", "cover1", "cover2"]
        # data_dir can be path or list
        if isinstance(data_dir, str):
            self.all_samples = [os.path.join(
                data_dir, file) for file in os.listdir(data_dir)]
        elif isinstance(data_dir, list):
            self.all_samples = data_dir
        self.mapping_file = mapping_file_path

        mapping_info = json.load(open(self.mapping_file))
        self.sample_list = []
        self.label_list = []
        for file in self.all_samples:
            for sample in mapping_info:
                if file.split("/")[-1] == sample.split("/")[-1]:
                    condition = mapping_info[sample]["condition"]
                    class_index = mapping_info[sample]["class"]
                    if mode == "all":
                        self.sample_list.append(file)
                        self.label_list.append(class_index)
                    elif condition == mode:
                        self.sample_list.append(file)
                        self.label_list.append(class_index)

    def __len__(self):
        """Get length of dataset

        Returns
        -------
        int
            length of dataset
        """
        return len(self.sample_list)

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
        file_name = self.sample_list[idx]
        label = self.label_list[idx]
        pose_embedding = pose_to_embedding_v2(np.load(file_name))
        return pose_embedding, int(label)


if __name__ == "__main__":
    config_file = "config.json"
    config = json.load(open(config_file, "r"))
    config = config['data']
    Dataset = PoseDataset(config['data_dir'],
                          config['train_list'], config['classes'])
