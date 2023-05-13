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
    parser.add_argument('--data-dir', type=str,
                        help='Path to training config file', default="/data2/samba/public/TrungPQ/22B/in-bed-posture-classification/data/coco/images/slp_val/")
    parser.add_argument("--output-dir", type=str,
                        help='Path to training config file', default="./autoencoder_feature/test")
    return parser

def main():
    parser = parse_argument()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pose_predictor = PosePredictor(args.config_path)
    pose_predictor.init_predictor()
    
    for file in os.listdir(args.data_dir):
        fp = os.path.join(args.data_dir, file)
        input_sample = cv2.imread(fp)
        output = pose_predictor.execute(input_sample)
        save_path = os.path.join(args.output_dir, file)
        output = np.squeeze(output)
        print(output.shape)
        np.save(save_path, output)


if __name__ == "__main__":
    main()