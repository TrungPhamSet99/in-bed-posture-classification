# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script to run demo training for pose classification
import os 
import json 
import argparse
import sys 
import warnings

from api.trainer import Trainer
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
        "Run script to train pose classification model")
    parser.add_argument('--config-path', type=str,
                        help='Path to training config file', default="./cfg/train/train_config.json")
    return parser


def main():
    parser = parse_argument()
    args = parser.parse_args()

    trainer = Trainer(args.config_path)
    trainer.initialize()
    loss_report = trainer.run_train()

if __name__ == "__main__":
    main()