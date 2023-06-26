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

from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from utils.general import pose_to_embedding_v2, to_onehot
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
        self.update_classes = json.load(open("/data/users/trungpq/23A/in-bed-posture-classification/pose_classification/scripts_to_prepare_data_phase2/single_module_data/update_classes.json"))
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

        self.classes = ["7", "8", "9"]
        if augment_config_path is not None:
            self.augmentor = Augmentor(augment_config_path)
        else:
            self.augmentor = None
        self.image_keys, self.dis_of_samples = self.filter_image_keys()

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
                if condition in ["uncover", "cover1", "cover2"]:
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

        if self.mode == "train":
            image_fp = image_fp.replace("train_", "slp_train/")
        else:
            image_fp = image_fp.replace("test_", "slp_val/")
        pose = np.load(pose_fp)
        image = cv2.imread(image_fp)

        # cv2.imwrite(f"./vis/{self.classes.index(c)}_{image_file}", image)

        if self.augmentor is not None:
            image, pose, vis_image = self.augmentor(image, pose, image_file)
        # image = cv2.resize(image, (224, 224))
        # pose = self.scale_pose(pose, (120,160), (160,160))
        # pose = pose_to_embedding_v2(pose)
        # if self.transform is not None:
        #     image = self.transform(image)
        # image = self.NORMALIZE(image)

        
        if image_file in self.update_classes:
            label = self.classes.index(str(self.update_classes[image_file]))
        else:
            label = self.classes.index(c)

        # Calculate geometric parameters related to angle and distance
        neck_coord = pose[:,12]
        sub_pose = pose[:,:6]
        center_of_hip = (sub_pose[:,2] + sub_pose[:,5]) / 2
        base_distance = np.linalg.norm(neck_coord - center_of_hip) # Distance from neck to center of hip
        avg_length_of_legs = (np.linalg.norm(sub_pose[:,2] - sub_pose[:,0]) + np.linalg.norm(sub_pose[:,2] - sub_pose[:,0])) / 2 
        knee_distance = np.linalg.norm(sub_pose[:,1] - sub_pose[:,4]) # Distance between 2 knees
        feet_distance = np.linalg.norm(sub_pose[:,0] - sub_pose[:,5]) # Distance between 2 feet

        knee_y = (sub_pose[1][1], sub_pose[1][4])
        feet_y = (sub_pose[1][0], sub_pose[1][5])
        hip_y = (sub_pose[1][2], sub_pose[1][3])
        

        knee_and_hip_diff_y = ((knee_y[0] - hip_y[0]) + (knee_y[0] - hip_y[1])) / 2
        

        knee_x = (sub_pose[0][1], sub_pose[0][4])
        feet_x = (sub_pose[0][0], sub_pose[0][5])
        knee_and_feet_diff_x = ((knee_x[0] - feet_x[0]) + (knee_x[0] - feet_x[1])) / 2

        knee_and_feet_diff_y = ((knee_y[0] - feet_y[0]) + (knee_y[0] - feet_y[1])) / 2

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
        

        angle_diff = abs(left_angle - right_angle)
        if math.isnan(right_angle):
            right_angle = 0
        if math.isnan(left_angle):
            left_angle = 0
        # return (sub_pose, image), self.classes.index(c)
        if self.classes == ["1","2","3"]:
            angle_thresh = [35, 25]
            distance_thresh = [5, 42]
            if left_angle <= angle_thresh[0] and right_angle <= angle_thresh[0]:
                if abs(sub_pose[1][2] - sub_pose[1][3]) < distance_thresh[0] and \
                   abs(sub_pose[1][0] - sub_pose[1][5]) < distance_thresh[0] and \
                   avg_length_of_legs > distance_thresh[1]:
                    pred = 0
                else: 
                    if knee_distance - feet_distance < distance_thresh[0] and avg_length_of_legs > distance_thresh[1]:
                        pred = 1
                    else: 
                        pred = 2
            else:
                if (left_angle >= angle_thresh[1] and right_angle <= angle_thresh[1]) or \
                   (left_angle <= angle_thresh[1] and right_angle >= angle_thresh[1]):
                    if feet_distance < 10:
                        pred = 2
                    else:
                        pred = 1
                elif (left_angle >= angle_thresh[1]) and (right_angle >= angle_thresh[1]):
                    pred = 2

        elif self.classes == ["4","5","6"] or self.classes == ["7","8","9"]:
            angle_thresh = [35, 25]
            if knee_and_hip_diff_y < 15:
                # Pred can be 1 or 2
                if knee_distance < 25 and (left_angle > 35 and right_angle > 35):
                    pred = 2
                else:
                    pred = 1
            else:
                # Pred can be 0, 1 or 2
                if knee_distance < 15 and feet_distance < 20:
                    # Pred can be 0 or 2
                    if (left_angle + right_angle) / 2 > 60:
                        if (left_angle + right_angle) / 2 > 70 or abs(knee_and_feet_diff_x) > 18:
                            pred = 2
                        else:
                            pred = 0
                    elif abs(knee_and_feet_diff_y) < 10:
                        pred = 2
                    else: 
                        pred = 0
                else:
                    if (knee_and_hip_diff_y > 25 and knee_distance < 15 and feet_distance < 20) or \
                       (left_angle < 45 and right_angle < 45 and angle_diff < 35) or \
                       (left_angle < 60 and right_angle < 60 and abs(feet_y[0] - feet_y[1]) < 10):
                        pred = 0
                    elif (left_angle > 60 and right_angle > 60 and angle_diff < 15) or \
                         (knee_distance < 20 and abs(feet_y[0] - feet_y[1]) < 10) or \
                         ((left_angle + right_angle) / 2 > 60 and knee_distance < 5):
                        pred = 2
                    else:
                        pred = 1  
                    if pred != label and label==1:
                        print(image_file, left_angle, right_angle, angle_diff, knee_distance, feet_distance, knee_y, feet_y, pred, label)
                        cv2.imwrite(f"./vis/{label}_{image_file}", image)
        
        return pred, label
        # return (left_angle, right_angle, knee_distance, feet_distance, knee_and_hip_diff_y), self.classes.index(c)


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
    