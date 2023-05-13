# author: Trungpq
# date: 24/09/2022 rainny day
# description: Create a JSON file to contain all information about dataset especially class (9 instead of 3) for each images

import json
import os
import argparse
import sys


def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--nine-class-label', type=str, 
                        help='Path to 9 class label file', default="./9_classes_label.json")
    parser.add_argument('--train-json-path', type=str,
                        help='Path to train json path', default="./train_SLP_path.json")
    parser.add_argument('--val-json-path', type=str, 
                        help='Path to val json path', default="./val_SLP_path.json")
    parser.add_argument('--save-path', type=str, 
                        help='Path to save output file', default="./everything.json")
    return parser


def get_class_index(path, label_info):
    path = path.replace("../../../", "")
    for _path in label_info:
        if path in _path:
            return label_info[_path]


def main():
    parser = parse_argument()
    args = parser.parse_args()
    nine_class_label_path = args.nine_class_label
    train_HRNet_path = args.train_json_path
    val_HRNet_path = args.val_json_path
    save_path = args.save_path
    
    # Check whether input paths are valid
    if not os.path.exists(nine_class_label_path):
        print(f"No such input path {nine_class_label_path}")
    if not os.path.exists(train_HRNet_path):
        print(f"No such input path {train_HRNet_path}")
    if not os.path.exists(val_HRNet_path):
        print(f"No such input path {val_HRNet_path}")

    # Read data from JSON file
    label_info = json.load(open(nine_class_label_path))
    train_HRNet_info = json.load(open(train_HRNet_path))
    val_HRNet_info = json.load(open(val_HRNet_path))

    # print(f"Number of samples in train set for HRNet: {len(train_HRNet_info)}")
    # print(f"Number of samples in test set for HRNet: {len(val_HRNet_info)}")

    final_info = {}

    for sample in train_HRNet_info:
        if "simLab" in sample:
            continue
        info = {}
        # info["original_path"] = sample 
        info["condition"] = sample.split("/")[-2]
        info["class"] = get_class_index(sample, label_info)
        info["HRNet_path"] = train_HRNet_info[sample]
        info["is_train"] = 1
        final_info[sample] = info

    for sample in val_HRNet_info:
        if "simLab" in sample:
            continue
        info = {}
        # info["original_path"] = sample 
        info["condition"] = sample.split("/")[-2]
        info["class"] = get_class_index(sample, label_info)
        info["HRNet_path"] = val_HRNet_info[sample]
        info["is_train"] = 0
        final_info[sample] = info

    with open(save_path, "w") as f:
        json.dump(final_info, f)


if __name__ == "__main__":
    main()
