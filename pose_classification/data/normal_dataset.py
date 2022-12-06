# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define dataset used for pose classification inherited from torch.utils.data.Dataset
import copy
import random
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.general import pose_to_embedding_v2
from data.augmentation import PoseAugmentor

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
