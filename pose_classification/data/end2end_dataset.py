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