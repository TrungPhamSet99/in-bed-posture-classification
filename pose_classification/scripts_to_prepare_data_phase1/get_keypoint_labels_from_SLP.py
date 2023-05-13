import os 
import json 
import argparse 
import shutil
import numpy as np 
import scipy.misc
from scipy import io
import cv2
import sys


POSTURE_CLASSSES = ["supine", "lying_left", "lying_right"]
JOINTS_FILE_NAME = "joints_gt_IR.mat"

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
    parser.add_argument('--SLP-root-path', type=str, 
                        help='Path to root of 9 class dataset', default="../pose_data/SLP2022/SLP/danaLab/")
    parser.add_argument('--merged-info-path', type=str,
                        help='Path to root of original dataset', default="./everything.json")
    parser.add_argument('--output-dir', type=str, 
                        help='Path to save output keypoint file', default="../pose_data/POSE_SLP2022/")
    return parser


def get_class_index(person_idx, image_idx, merged_info):
    person_idx = person_idx.lstrip("0")
    image_idx = image_idx.lstrip("0")
    for sample in merged_info:
        person_id = sample.split("/")[-4].lstrip("0")
        image_id = sample.split("/")[-1].split("_")[-1].replace(".png","").lstrip("0")
        if person_idx == person_id and image_idx == image_id:
            return merged_info[sample]['class']
        


def main():
    parser = parse_argument()
    args = parser.parse_args()

    SLP_root_path = args.SLP_root_path
    merged_info_path = args.merged_info_path
    output_dir = args.output_dir

    # Check whether input paths are valid
    if not os.path.exists(SLP_root_path):
        print(f"No such input path {SLP_root_path}")
        sys.exit(0)

    if not os.path.exists(merged_info_path):
        print(f"No such input path {merged_info_path}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read data from merged info JSON file
    merged_info = json.load(open(merged_info_path))
    class_index = []
    for sample in merged_info:
        if merged_info[sample]['class'] not in class_index:
            class_index.append(merged_info[sample]['class'])
        elif merged_info[sample]['class'] is None:
            print(merged_info[sample])

    print(f"Number of classes: {len(class_index)}")
    print(f"Class index: {class_index}")
    print(f"Number of samples in dataset: {len(merged_info)}")
    
    # Iterate SLP dataset to get keypoints label paths as numpy files    
    label_file_paths = []
    for root, dirs, files in os.walk(SLP_root_path):
        for file in files:
            if JOINTS_FILE_NAME in file and "old" not in file:
                label_file_paths.append(os.path.join(root, file))

    # Create sub folders for each classes
    for c in class_index:
        fp = os.path.join(output_dir, c)
        if not os.path.exists(fp):
            os.makedirs(fp)
    # Iterate label file to extract keypoint for each image and save as numpy file.
    for index, path in enumerate(label_file_paths):
        print("Process {0}/{1} files".format(index, len(label_file_paths)), end="\r")
        pose = io.loadmat(path)['joints_gt']
        person_id = path.split("/")[-2]
        for i in range(pose.shape[-1]):
            image_id = str(i+1)
            class_idx = get_class_index(person_id, image_id, merged_info)
            sub_pose = pose[0:2,:,i]
            save_path = os.path.join(output_dir, class_idx, "{}_{}.npy".format(person_id, image_id))
            np.save(save_path, sub_pose)
    print(f"Keypoint label files were saved to {output_dir}")


if __name__ == "__main__":
    main()