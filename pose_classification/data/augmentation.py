# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define data augmentation methods used for training
#              Need investigate to offline augmentation instead of online augmentation
import imgaug  
import imgaug.augmenters as iaa
import os
import cv2
import numpy as np
import json
import random
from utils.general import load_config


random.seed(10)

class Augmentor(object):
    def __init__(self, config_path):
        """Constructor for PoseAugmentor

        Parameters
        ----------
        config_path : str
            Path to JSON config file

        Raises
        ------
        ValueError
            Raise ValueError if path is invalid
        """
        if not os.path.exists(config_path):
            raise ValueError(f"No such config path {config_path}")
        self.config = load_config(config_path)

    def augment_keypoint(self, keypoint, magnitude):
        """Augment keypoint

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        magnitude_range : tuple or list
            Magnitude range for randomly select

        Returns
        -------
        np.ndarray
            Keypoints after moving up
        """
        magnitude_x, magnitude_y = magnitude[0], magnitude[1]
        return np.array([keypoint[0, :] + magnitude_x, keypoint[1, :] + magnitude_y])

    def translate_image(self, image, magnitude):
        height, width = image.shape[:2]
        translate_matrix = np.float32([[1,0,magnitude[0]], [0,1,magnitude[1]]])
        translated_image = cv2.warpAffine(image, translate_matrix, (height, width))
        return translated_image


    def non_geometric_transform(self, image):
        configs = self.config['non_geometric_transformations']
        method_list = list()
        for key in configs.keys():
            if key == 'prob':
                continue
            if random.choices([0, 1], [1 - configs[key]['prob'], configs[key]['prob']])[0]:
                sub_methods = configs[key]['sub_methods']
                prob_list = [method['prob'] for method in sub_methods]
                if sum(prob_list) != 1:
                    raise ValueError(
                        'Sum of probabilities should be 1 instead of {0}'.format(sum(prob_list)))
                index = random.choices(
                    list(range(len(sub_methods))), prob_list)[0]
                method = eval('iaa.' + sub_methods[index]['name'])
                method_list.append(
                    method(tuple(sub_methods[index]['magnitude'])))
        seq = iaa.Sequential(method_list)
        return seq(image=image)

    def __call__(self, image, keypoint, name, visualize=True):
        if random.random() < self.config["prob"]:
            magnitude_x = random.randint(self.config["magnitude_x"][0], self.config["magnitude_x"][1])
            magnitude_y = random.randint(self.config["magnitude_y"][0], self.config["magnitude_y"][1])
            image = self.translate_image(image, (magnitude_x, magnitude_y))
            keypoint = self.augment_keypoint(keypoint, (magnitude_x, magnitude_y))
        if random.random() < self.config["non_geometric_transformations"]["prob"]:
            image = self.non_geometric_transform(image)

        if visualize:
            keypoint_list = keypoint.tolist()
            save_path = os.path.join("../vis/", name)
            for i in range(14):
                x = int(keypoint_list[0][i])
                y = int(keypoint_list[1][i])
                vis_image = cv2.circle(image, (x,y), 2, (255,0,0), 1)
                cv2.imwrite(save_path, vis_image)
        return image, keypoint




if __name__ == "__main__":
    # Import sample data to test Pose Augmentor
    config_path = "/data2/samba/public/TrungPQ/22B/in-bed-posture-classification/pose_classification/cfg/augmentation.json"
    image_dir = "/data2/samba/public/TrungPQ/22B/in-bed-posture-classification/data/coco/images/"
    keypoint_dir = "/data2/samba/public/TrungPQ/22B/pose_data/POSE_SLP2022"
    mapping_file = "/data2/samba/public/TrungPQ/22B/in-bed-posture-classification/pose_classification/scripts_to_prepare_data_phase2/single_module_data/single_module_mapping_file_train.json"
    
    vis_dir = "../vis/"
    os.makedirs(vis_dir, exist_ok=True)

    mapping = load_config(mapping_file)
    augmentor = Augmentor(config_path)
    image_keys = list(mapping.keys())

    for i in range(len(image_keys)):
        image_file = image_keys[i]
        keypoint_file = mapping[image_file]
        image_fp = os.path.join(image_dir, image_file).replace("train_", "slp_train/")
        keypoint_fp = os.path.join(keypoint_dir, keypoint_file)
        img = cv2.imread(image_fp)
        keypoint = np.load(keypoint_fp)
        output_image, output_keypoint = augmentor(img, keypoint, image_file)
