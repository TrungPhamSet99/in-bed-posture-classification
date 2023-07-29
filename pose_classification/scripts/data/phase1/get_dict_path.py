import os
import json
import sys
import numpy as np 
import argparse


MANUAL = """
Run script to get dict path from 9 class dataset to original dataset 

""".format(script=__file__)
def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser(MANUAL)
    parser.add_argument('--nine-class-path', type=str, 
                        help='Path to root of 9 class dataset', default="/data/users/trungpq/22A/pose_data/SLP_9_CLASS/data")
    parser.add_argument('--original-path', type=str,
                        help='Path to root of original dataset', default="/data/users/trungpq/22A/pose_data/SLP2022/SLP/danaLab")
    parser.add_argument('--save-path', type=str, 
                        help='Path to save output JSON file', default="./9_classes_label_v2.json")
    return parser


def get_class_from_9_class_path(fp):
    return fp.split("/")[-2]

def train_or_val(fp):
    return fp.split("/")[-3]

def get_condition(fp, original=True):
    if original:
        return fp.split("/")[-2]
    else:
        return fp.split("/")[-5].split("_")[-1].lower()

def get_class(original_sample_path, nine_class_root):
    print(original_sample_path)
    person_id = original_sample_path.split("/")[-4].lstrip("0")
    print(person_id)
    print("------------------------------------------------")
    image_id = original_sample_path.split("/")[-1].split("_")[-1].replace(".png","")
    for root, dirs, files in os.walk(nine_class_root):
        if "train_aug" in root:
            continue
        for file in files:
            if file == f"image_{image_id}_{person_id}.png":
                fp = os.path.join(root, file)
                if person_id == "80":
                    print(fp)
                return get_class_from_9_class_path(fp)
    return "Unknown"


def main():
    parser = parse_argument()
    args = parser.parse_args()
    nine_class_root = args.nine_class_path
    original_root = args.original_path
    save_path = args.save_path

    if not os.path.exists(nine_class_root):
        print(f"No such input path {nine_class_root}")
        sys.exit(0)
    if not os.path.exists(original_root):
        print(f"No such input path {original_root}")
        sys.exit(1)

    class_dict = {}
    for root, dirs, files in os.walk(original_root):
        print(f"Number of sample in class dict {len(class_dict)}", end="\r")
        if "IR/" not in root:
            continue
        for file in files:
            if ".png" in file or ".jpg" in file:
                fp = os.path.join(root, file)  
                class_index = get_class(fp, nine_class_root)
                class_dict[fp] = class_index

    count = 0
    for sample in class_dict:
        if class_dict[sample] == "Unknown":
            count += 1

    print(f"Number sample with Unknown label: {count}")
    print(f"Number of samples in label file: {len(class_dict)}")
    with open(save_path, "w") as f:
        json.dump(class_dict, f)

if __name__ == "__main__":
    main()