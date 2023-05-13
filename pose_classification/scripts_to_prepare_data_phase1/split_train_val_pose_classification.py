import os 
import json 
import sys 
import numpy 
import argparse
import random

random.seed(10)

RAW_POSE = ["supine", "lying_left", "lying_right"]

def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose-data-root-path', type=str, 
                        help='Path to root of 9 class dataset', default="../pose_data/POSE_SLP2022/")
    parser.add_argument('--merged-info-path', type=str,
                        help='Path to root of original dataset', default="./everything.json")
    parser.add_argument('--output-train-list', type=str, 
                        help='Path to save output keypoint file', default="../pose_data/POSE_SLP2022/train_list.txt")
    parser.add_argument('--output-test-list', type=str, 
                        help='Path to save output keypoint file', default="../pose_data/POSE_SLP2022/test_list.txt")
    return parser

def get_category(person_idx, image_idx, merged_info):
    person_idx = person_idx.lstrip("0")
    image_idx = image_idx.lstrip("0")
    for sample in merged_info:
        person_id = sample.split("/")[-4].lstrip("0")
        image_id = sample.split("/")[-1].split("_")[-1].replace(".png","").lstrip("0")
        if person_idx == person_id and image_id == image_idx:
            return merged_info[sample]["is_train"]


def main():
    parser = parse_argument()
    args = parser.parse_args()
    pose_data_root = args.pose_data_root_path
    merged_info_path = args.merged_info_path
    output_train = args.output_train_list
    output_test = args.output_test_list

    # Read merged information from JSON file
    merged_info = json.load(open(merged_info_path))

    # Declare list and dict use to contain output data
    keypoint_label_paths = []
    train_list = []
    test_list = []
    distribution = {}
    distribution['train'] = {}
    distribution['test'] = {}
    class_list = []

    for root, dirs, files in os.walk(pose_data_root):
        for file in files:
            if file.endswith(".npy"):
                keypoint_label_paths.append(os.path.join(root, file))

    for path in keypoint_label_paths:
        class_idx = path.split("/")[-2]
        if class_idx not in class_list:
            class_list.append(class_idx)

    for c in class_list:
        distribution["train"][c] = 0 
        distribution["test"][c] = 0

    for path in keypoint_label_paths:
        person_id = path.split("/")[-1].split("_")[0]
        image_id = path.split("/")[-1].split("_")[-1].replace(".npy", "")
        class_id = path.split("/")[-2]
        if class_id == "Unknown":
            continue
        if get_category(person_id, image_id, merged_info):
            train_list.append(os.path.join(class_id, path.split("/")[-1]) + "\n")
            distribution["train"][class_id] += 1
        else: 
            test_list.append(os.path.join(class_id, path.split("/")[-1]) + "\n")
            distribution["test"][class_id] += 1

    for sample in distribution:
        for c in class_list:
            print(f"{sample}-{c}-{distribution[sample][c]}")

    for c in class_list:
        if c == "Unknown":
            continue
        ratio = distribution["test"][c]/distribution["train"][c]
        print(f"{c} {ratio}")

    random.shuffle(train_list)
    random.shuffle(test_list)

    supine_train_list = []
    supine_test_list = []
    lying_left_train_list = []
    lying_left_test_list = []
    lying_right_train_list = []
    lying_right_test_list = []

    for sample in train_list:
        if int(sample.split("/")[0])%3 == 0:
            raw_pose_name = RAW_POSE[(int(sample.split("/")[0])//3)-1]
        else: 
            raw_pose_name = RAW_POSE[(int(sample.split("/")[0])//3)]
        eval(f"{raw_pose_name}_train_list").append(sample)

    for sample in test_list:
        if int(sample.split("/")[0])%3 == 0:
            raw_pose_name = RAW_POSE[(int(sample.split("/")[0])//3)-1]
        else: 
            raw_pose_name = RAW_POSE[(int(sample.split("/")[0])//3)]
        eval(f"{raw_pose_name}_test_list").append(sample)

    for pose in RAW_POSE:
        output_train_path = output_train.replace("train_list", f"{pose}_train_list")
        output_test_path = output_test.replace("test_list", f"{pose}_test_list")
        print(output_train_path)
        print(output_test_path)
        with open(output_train_path, "w") as f:
            f.writelines(eval(f"{pose}_train_list"))
        with open(output_test_path, "w") as f:
            f.writelines(eval(f"{pose}_test_list"))

    # with open(output_train, "w") as f:
    #     f.writelines(train_list)

    # with open(output_test, "w") as f:
    #     f.writelines(test_list)

if __name__ == "__main__":
    main()