# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define dataset used for pose classification inherited from torch.utils.data.Dataset
import cv2
import copy
import math
import random
import os
import json
import numpy as np
import torch
import warnings

from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from utils.general import pose_to_embedding_v2, to_onehot, leg_pose_to_embedding
from data.augmentation import Augmentor

warnings.filterwarnings('ignore')

class NormalPoseDataset(Dataset):
    RGB_MEAN = [0.18988903, 0.18988903, 0.18988903]
    RGB_STD = [0.09772425, 0.09772425, 0.09772425]
    NORMALIZE = Compose([Normalize(mean=RGB_MEAN, std=RGB_STD)])
    def __init__(self, data_dir: str, list_path: str, mapping_file: str, image_dir: str, 
                 classes = None, augment_config_path=None, transform=None, load_from_gt=True):
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
        self.update_classes = json.load(open("/data/users/trungpq/23A/in-bed-posture-classification/pose_classification/scripts/data/phase2/single_module_data/update_classes.json"))
        self.hrnet_output = json.load(open("/data/users/trungpq/23A/in-bed-posture-classification/pose_classification/scripts/data/phase2/json_files/hrnet_test_output.json"))
        self.data_root = data_dir
        self.data_list_path = list_path
        self.transform = transform
        self.mapping = json.load(open(mapping_file, "r"))
        self.image_keys = list(self.mapping.keys())
        self.image_dir = image_dir
        self.load_from_gt = load_from_gt
        if "train" in self.data_list_path:
            self.mode = "train"
        else:
            self.mode = "test"
        if isinstance(list_path, str):
            self.data_paths = open(self.data_list_path).readlines()
            self.data_paths = [ele.rstrip() for ele in self.data_paths]
        elif isinstance(list_path, list):
            self.data_paths = list_path

        self.classes = classes
        if augment_config_path is not None:
            self.augmentor = Augmentor(augment_config_path)
        else:
            self.augmentor = None
        self.image_keys, self.dis_of_samples = self.filter_image_keys()
        print(self.dis_of_samples)

    @staticmethod
    def calculate_angle(first_line, second_line):
        def slop(line):
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            return (y2-y1)/(x2-x1)
        
        def angle(first_slope, second_slope):
            return math.degrees(math.atan((second_slope-first_slope)/(1+(second_slope*first_slope))))

        first_slope = slop(first_line)
        second_slope = slop(second_line)

        return abs(angle(first_slope, second_slope))

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
            c = self.mapping[sample][0].split("/")[-2]
            if c in self.classes:
                condition = self.mapping[sample][1]
                if condition in ["cover2"]:
                    index = self.classes.index(c)
                    dis_of_samples[index] += 1
                    output.append(sample)
            else: 
                continue
        return output, dis_of_samples

    def scale_pose(self, pose, original_shape, target_shape):
        x_scale = target_shape[0] / original_shape[0]
        y_scale = target_shape[1] / original_shape[1]
        return np.array([pose[0,:]*x_scale, pose[1,:]*y_scale])
    
    def load_pose_from_hrnet_output(self, image_name):
        image_id = image_name.lstrip("0").replace(".png", "")
        if image_id == "":
            image_id = "0"
        for element in self.hrnet_output:
            if element['image_id'] == int(image_id):
                pose = element['keypoints']
                break 
        pose = [element for element in pose if element > 2]
        x_list = []
        y_list = []
        for i, element in enumerate(pose):
            if i in list(range(0,27,2)):
                x_list.append(element)
            else:
                y_list.append(element)
        pose = np.asarray([[x_list, y_list]])
        return pose[0, :, :]
    
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
        path = self.mapping[image_file][0]
        c = path.split("/")[-2]
        pose_fp = os.path.join(self.data_root, path)
        image_fp = os.path.join(self.image_dir, image_file)
        # print(image_fp, pose_fp)

        if self.mode == "train":
            image_fp = image_fp.replace("train_", "slp_train/")
        else:
            image_fp = image_fp.replace("test_", "slp_val/")
        if self.load_from_gt:
            pose = np.load(pose_fp)
        else:
            pose = self.load_pose_from_hrnet_output(image_file.replace("test_", ""))
        angle_left, angle_right = self.calculate_angles(pose)
        image = cv2.imread(image_fp)
        image = self.crop_leg_from_image(image, pose)
        if self.augmentor is not None:
            image, pose, _= self.augmentor(image, pose, image_file)
            cv2.imwrite(f"../../vis/{image_file}", image)
        
        image = cv2.resize(image, (180, 180))
        # pose = self.scale_pose(pose, (120,160), (160,160))
        pose = leg_pose_to_embedding(pose, angle_left, angle_right)
        if self.transform is not None:
            image = self.transform(image)
        image = self.NORMALIZE(image)

        if image_file in self.update_classes:
            label = self.classes.index(str(self.update_classes[image_file]))
        else:
            label = self.classes.index(c)

        if label in [0,3,6]: 
            label = 0
        elif label in [1,4,7]:
            label = 1
        else:
            label = 2

        return image, pose, label

    @staticmethod
    def crop_leg_from_image(image, pose):
        sub_pose = pose[:, :6]
        x = sub_pose[0,:]
        y = sub_pose[1,:]
        x_max, x_min = int(np.max(x)), int(np.min(x))
        y_max, y_min = int(np.max(y)), int(np.min(y))
        cropped_image = image[y_min-20:y_max+20, x_min-20:x_max+20, :]
        return cropped_image
    
    def calculate_angle(self, first_line, second_line):
        def slop(line):
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            if x2 - x1 == 0:
                return 1e9
            else:
                return (y2-y1)/(x2-x1)
        
        def angle(first_slope, second_slope):
            return math.degrees(math.atan((second_slope-first_slope)/(1+(second_slope*first_slope))))

        first_slope = slop(first_line)
        second_slope = slop(second_line)

        return abs(angle(first_slope, second_slope))

    def calculate_angles(self, pose):
        sub_pose = pose[:,:6]
        right_side = [sub_pose[0][0], sub_pose[1][0],
                      sub_pose[0][1], sub_pose[1][1],
                      sub_pose[0][2], sub_pose[1][2]]
        left_side = [sub_pose[0][3], sub_pose[1][3],
                     sub_pose[0][4], sub_pose[1][4],
                     sub_pose[0][5], sub_pose[1][5]]
        left_angle = self.calculate_angle((left_side[0], left_side[1], left_side[2], left_side[3]),
                                        (left_side[2], left_side[3], left_side[4], left_side[5]))
        
        right_angle = self.calculate_angle((right_side[0], right_side[1], right_side[2], right_side[3]),
                                        (right_side[2], right_side[3], right_side[4], right_side[5]))
        
        return left_angle, right_angle
    
def batch_mean_and_std(loader):
    cnt = 0

    fst_moment = torch.empty(40)
    snd_moment = torch.empty(40)
    for tensor, _ in loader:
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
    