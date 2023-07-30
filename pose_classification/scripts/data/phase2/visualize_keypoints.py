import cv2  
import os 
import argparse
import json
import numpy as np


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
    parser.add_argument('--image-dir', type=str,
                        help='Path to training config file', default="/data/users/trungpq/coco/images/slp_val")
    parser.add_argument('--hrnet-feature', type=str,
                        help='Path to training config file', default="/data2/samba/public/TrungPQ/22B/pose_data/POSE_SLP2022")
    parser.add_argument("--mapping-file", type=str,
                        help='Path to training config file', default="./scripts_to_prepare_data_phase2/single_module_data/single_module_mapping_file.json")            
    parser.add_argument('--output-dir', type=str,
                        help='Path to training config file', default="./vis/")
    return parser


def visualize_keypoint_on_image(image, keypoints):
    for i in range(14):
        point = (int(keypoints[0][i]), int(keypoints[1][i]))
        image = cv2.circle(image, point, 2, (255,0,0), 1)
    return image

def main():
    parser = parse_argument()
    args = parser.parse_args()
    image_dir = args.image_dir
    hrnet_feature = args.hrnet_feature
    mapping_file = args.mapping_file
    output_dir = args.output_dir 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mapping = json.load(open(mapping_file, "r"))
    for file in os.listdir(image_dir):
        image_fp = os.path.join(image_dir, file)
        image = cv2.imread(image_fp)
        key = f"train_{file}"
        if key in mapping:
            pose = np.load(os.path.join(hrnet_feature, mapping[key]))
            pose = pose.tolist()
            vis_image = visualize_keypoint_on_image(image, pose)
            save_path = os.path.join(output_dir, file)
            cv2.imwrite(save_path, visualize_keypoint_on_image(image, pose))
        else:
            continue

if __name__ == "__main__":
    main()