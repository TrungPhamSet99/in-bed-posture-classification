# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script to run demo prediction for pose classification
import os
import json 
import argparse
import numpy as np 
import warnings

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
                        help='Path to training config file', default="./cfg/predict/predict_config.json")
    parser.add_argument('--sample_input_path', type=str,
                        help='Path to sample input as numpy file', default="/data/users/trungpq/22A/pose_data/POSE_SLP2022/6/00061_26.npy")
    return parser

def main():
    parser = parse_argument()
    args = parser.parse_args()
    
    pose_predictor = PosePredictor(args.config_path)
    pose_predictor.init_predictor()

    input_sample = np.load(args.sample_input_path)
    output = pose_predictor.execute(input_sample)
    print(output)

if __name__ == "__main__":
    main()