# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define data augmentation methods used for training
#              Need investigate to offline augmentation instead of online augmentation
import numpy as np
import json
import random
from utils.general import load_config
import os

random.seed(10)

class PoseAugmentor:
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

    def up(self, keypoint, magnitude_range):
        """Move keypoints up

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
        magnitude = random.randint(magnitude_range[0], magnitude_range[1])
        return np.array([keypoint[0, :], keypoint[1, :] - magnitude])

    def left(self, keypoint, magnitude_range):
        """Move keypoints left

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        magnitude_range : tuple or list
            Magnitude range for randomly select
        Returns
        -------
        np.ndarray
            Keypoint after moving left
        """
        magnitude = random.randint(magnitude_range[0], magnitude_range[1])
        return np.array([keypoint[0, :] - magnitude, keypoint[1, :]])

    def down(self, keypoint, magnitude_range):
        """Move keypoints down

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        magnitude_range : tuple or list
            Magnitude range for randomly select
        Returns
        -------
        np.ndarray
            Keypoint after moving down
        """
        magnitude = random.randint(magnitude_range[0], magnitude_range[1])
        return np.array([keypoint[0, :], keypoint[1, :] + magnitude])

    def right(self, keypoint, magnitude_range):
        """Move keypoints right

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        magnitude_range : tuple or list
            Magnitude range for randomly select
        Returns
        -------
        np.ndarray
            Keypoint after moving right
        """
        magnitude = random.randint(magnitude_range[0], magnitude_range[1])
        return np.array([keypoint[0, :] + magnitude, keypoint[1, :]])

    def up_right(self, keypoint, magnitude_range):
        """Move keypoints up-right

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        magnitude_range : tuple or list
            Magnitude range for randomly select
        Returns
        -------
        np.ndarray
            Keypoint after moving up-right
        """
        magnitude = random.randint(magnitude_range[0], magnitude_range[1])
        return np.array([keypoint[0, :] + magnitude, keypoint[1, :] - magnitude])

    def up_left(self, keypoint, magnitude_range):
        """Move keypoints up-left

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        magnitude_range : tuple or list
            Magnitude range for randomly select
        Returns
        -------
        np.ndarray
            Keypoint after moving up-left
        """
        magnitude = random.randint(magnitude_range[0], magnitude_range[1])
        return keypoint - magnitude

    def down_right(self, keypoint, magnitude_range):
        """Move keypoints down-right

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        magnitude_range : tuple or list
            Magnitude range for randomly select
        Returns
        -------
        np.ndarray
            Keypoint after moving down-right
        """
        magnitude = random.randint(magnitude_range[0], magnitude_range[1])
        return keypoint + magnitude

    def down_left(self, keypoint, magnitude_range):
        """Move keypoints down-left

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        magnitude_range : tuple or list
            Magnitude range for randomly select
        Returns
        -------
        np.ndarray
            Keypoint after moving down-left
        """
        magnitude = random.randint(magnitude_range[0], magnitude_range[1])
        return np.array([keypoint[0, :] - magnitude, keypoint[1, :] + magnitude])

    def augment(self, keypoint):
        """Interface method used for keypoint augmentation

        Parameters
        ----------
        keypoint : np.ndarray
            Keypoints
        Returns
        -------
        np.ndarray
            Augmented keypoints
        """
        if random.random() < self.config["prob"]:
            group1_method_list = list(
                self.config["methods"]["group1"]["sub_methods_magnitude"].keys())
            group2_method_list = list(
                self.config["methods"]["group2"]["sub_methods_magnitude"].keys())
            # Select random a method from groups 1
            method1 = random.choices(
                group1_method_list, self.config["methods"]["group1"]["prob"])[0]
            method2 = random.choices(
                group2_method_list, self.config["methods"]["group1"]["prob"])[0]
            # Get magnitude range to perform augmentation
            method1_magnitude_range = self.config["methods"]["group1"]["sub_methods_magnitude"][method1]
            method2_magnitude_range = self.config["methods"]["group2"]["sub_methods_magnitude"][method2]
            # Perform augmentation
            _method1 = eval(f"self.{method1}")
            _method2 = eval(f"self.{method2}")
            keypoint = _method1(keypoint, method1_magnitude_range)
            keypoint = _method2(keypoint, method2_magnitude_range)
            return keypoint
        else:
            return keypoint


if __name__ == "__main__":
    # Import sample data to test Pose Augmentor
    config_path = "/data/users/trungpq/22A/hrnet_pose_estimate/deep-high-resolution-net.pytorch/pose_classification/cfg/augmentation.json"
    sample_keypoint_path = "/data/users/trungpq/22A/pose_data/POSE_SLP2022/1/00041_14.npy"
    sample_keypoint = np.load(sample_keypoint_path)
    print(f"Sample keypoint: \n {sample_keypoint}")

    # Create a instance of PoseAugmentor
    augmenter = PoseAugmentor(config_path)
    augmented_keypoint = augmenter.augment(sample_keypoint)

    print(f"Augmented keypoints: \n {augmented_keypoint}")
