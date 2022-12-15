# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define dataset used to train AutoEncoder
import copy
import random
import os
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset


class AutoEncoderDataset(Dataset):
    def __init__(self, data_dir, data_list=None, augment_config_path=None ,transform=None):
        if isinstance(data_dir, str) and os.path.isdir(data_dir):
            self.data_dir = data_dir 
        else:
            raise ValueError(f"Invalid path to data dir {data_dir}")

        if data_list:
            if isinstance(data_list, str) and os.path.isfile(data_list):
                with open(data_list, "r") as f:
                    self.data_list = [os.path.join(self.data_dir, path.strip()) for path in f.readlines()]
            elif isinstance(data_list, list):
                self.data_list = data_list
            else:
                raise TypeError("`data_list` parameter is invalid, check it again")
        else:
            self.data_list = None
        self.transform = transform

        if not self.data_list:
            self.data_list = [os.path.join(self.data_dir, path) for path in os.listdir(str(self.data_dir))]
        self.data_samples = []
        for idx in tqdm(range(len(self.data_list)), desc="Building dataset"):
            fp = self.data_list[idx]
            image = cv2.imread(fp)
            image = cv2.resize(image, (120,120))
            # image = np.transpose(image, (2,0,1)) # Convert (W,H,C) -> (C,W,H)
            # Use transform.ToTensor() to make transpose and convert image to torch.Tensor
            # Reference: https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
            self.data_samples.append(image)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, sample


if __name__ == "__main__":
    data_dir = "/data/users/trungpq/22B/in-bed-posture-classification/data/coco/images/"
    save_path = "/data/users/trungpq/22B/in-bed-posture-classification/data/coco/images/test_list.txt"