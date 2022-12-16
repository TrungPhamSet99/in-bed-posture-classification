# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script to run demo prediction for pose classification
import os
import json 
import argparse
import numpy as np 
import warnings
import cv2
from api.predictor import PosePredictor
warnings.filterwarnings('ignore')

def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser(
        "Run script to predict pose classification model")
    parser.add_argument('--config-path', type=str,
                        help='Path to training config file', default="./cfg/predict/autoencoder_config.json")
    return parser

def main():
    parser = parse_argument()
    args = parser.parse_args()
    sample_input_path = json.load(open(args.config_path, "r"))["sample_input_path"]

    pose_predictor = PosePredictor(args.config_path)
    pose_predictor.init_predictor()
    if ".npy" in sample_input_path:
        input_sample = np.load(sample_input_path)
    elif ".png" in sample_input_path:
        input_sample = cv2.imread(sample_input_path)
    else:
        raise ValueError(f"Only support numpy and PNG file as input, check your config again")

    output = pose_predictor.execute(input_sample)
    print(output.shape)

if __name__ == "__main__":
    main()