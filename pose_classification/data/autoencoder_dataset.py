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
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from pathlib import Path

class AutoEncoderDataset(Dataset):
    RGB_MEAN = [0.18988903, 0.18988903, 0.18988903]
    RGB_STD = [0.09772425, 0.09772425, 0.09772425]
    NORMALIZE = Compose([Normalize(mean=RGB_MEAN, std=RGB_STD)])
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
            # Use transform.ToTensor() to make transpose, convert image to torch.Tensor and normalize to range (0,1)
            # Consider use torchvision.transform.Normalize to normalize batch using mean and standard value of dataset
            # Reference: https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
            self.data_samples.append(image)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        if self.transform:
            sample = self.transform(sample)
        self.NORMALIZE(sample)
        return sample, sample


if __name__ == "__main__":
    data_dir = "/data/users/trungpq/22B/in-bed-posture-classification/data/coco/images/"
    data_list = "/data/users/trungpq/22B/in-bed-posture-classification/data/coco/images/train_list.txt"

    imageFilesDir = Path("/data/users/trungpq/22B/in-bed-posture-classification/data/coco/images/")
    files = list(imageFilesDir.rglob('*.png'))

    mean = np.array([0.,0.,0.])
    stdTemp = np.array([0.,0.,0.])
    std = np.array([0.,0.,0.])

    numSamples = len(files)

    for i in range(numSamples):
        print(i)
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        
        for j in range(3):
            mean[j] += np.mean(im[:,:,j])

    mean = (mean/numSamples)

    for i in range(numSamples):
        print(i)
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        for j in range(3):
            stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

    std = np.sqrt(stdTemp/numSamples)

    print(mean)
    print(std)