# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script mapping output of HRNet to images for end 2 end evaluation (Note: this is only used for testing phase)
import json
import os
import argparse
import sys
import numpy as np
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        description='mapping images to original dataset')
    parser.add_argument('--hrnet-output-json', type=str, default="../json_files/hrnet_test_output.json",
                        help='Path to JSON file that containing hrnet output')
    parser.add_argument('--mapping-to-original', type=str, default="../json_files/val_SLP_path.json",
                        help='Path to JSON mapping file for validation images')
    parser.add_argument('--mapping-to-keypoints', type=str, default="../single_module_data/single_module_mapping_file.json",
                        help='Path to JSON mapping file for validation images')
    parser.add_argument('--output-path', type=str, default="./pose_numpy/",
                        help='Path to JSON mapping file for validation images')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    hrnet_output = json.load(open(args.hrnet_output_json, "r"))
    mapping_to_original = json.load(open(args.mapping_to_original, "r"))
    mapping_to_keypoints = json.load(open(args.mapping_to_keypoints, "r"))
    mapping_to_test_keypoints = {(k, mapping_to_keypoints[k]) for k in mapping_to_keypoints.keys() if "train" not in k}
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    output = {}

    # print(len(hrnet_output))
    # print(len(mapping_to_original))
    # print(len(mapping_to_test_keypoints))
    # image_id_max = max([out["image_id"] for out in hrnet_output])
    # print(image_id_max)

    for out in hrnet_output:
        # print(out.keys())
        raw_keypoints = np.array(out["keypoints"])
        image_id = out["image_id"]
        file_name = f"{image_id}_pose.npy"
        save_path = os.path.join(output_path, file_name)
        # print("Raw output keypoints from HRNet:")
        # print(raw_keypoints)
        # Remove confidence score from raw keypoints
        raw_keypoints = raw_keypoints[raw_keypoints>1]
        x_index = np.array([2*i for i in range(14)])
        y_index = np.array([2*i+1 for i in range(14)])
        formated_keypoint = np.reshape(np.take(raw_keypoints, np.concatenate((x_index, y_index))),
                                       (2,14))
        
        np.save(save_path, formated_keypoint)
        

if __name__ == "__main__":
    main()
