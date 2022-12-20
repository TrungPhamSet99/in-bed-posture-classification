import json
import os
import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description='mapping images to original dataset')
    parser.add_argument('--mapping-file-train', type=str, default="../json_files/train_SLP_path.json",
                        help='Path to JSON mapping file for training images')
    parser.add_argument('--mapping-file-val', type=str, default="../json_files/val_SLP_path.json",
                        help='Path to JSON mapping file for validation images')
    parser.add_argument('--keypoint-data-dir', type=str, default="../../../../POSE_SLP2022/",
                        help='Path to JSON mapping file for validation images')
    args = parser.parse_args()
    return args

def get_person_id(path):
    return path.split("/")[-4].lstrip("0")

def get_image_id(path):
    return path.split("/")[-1].replace(".png", "").replace("image_", "").lstrip("0")

def main():
    args = get_args()
    mapping_train = json.load(open(args.mapping_file_train, "r"))
    mapping_val = json.load(open(args.mapping_file_val, "r"))
    print(len(mapping_train))
    print(len(mapping_val))
    keypoint_data_dir = args.keypoint_data_dir
    # Process keypoints data
    test_list_files = []
    train_list_files = []
    keypoint_train_samples = []
    keypoint_test_samples = []
    for file in os.listdir(keypoint_data_dir):
        fp = os.path.join(keypoint_data_dir, file)
        if os.path.isdir(fp):
            continue
        if "test" in file:
            test_list_files.append(fp)
        elif "train" in file:
            train_list_files.append(fp)

    for path in test_list_files:
        samples = open(path).readlines()
        for s in samples:
            keypoint_test_samples.append(s.rstrip())

    for path in train_list_files:
        samples = open(path).readlines()
        for s in samples:
            keypoint_train_samples.append(s.rstrip())

    # print(len(keypoint_test_samples)*3)
    # print(len(keypoint_train_samples)*3)
    # print(keypoint_test_samples)
    # print(keypoint_train_samples)

    mapping_keypoints_train_samples = []
    mapping_keypoints_test_samples = []
    output = {}
    for element in mapping_train:
        # output[f"train_{mapping_train[element].split("/")[-1]}"] = f"{get_person_id(element)}_{get_image_id(element)}"
        image_name = mapping_train[element].split("/")[-1]
        output[f"train_{image_name}"] = f"{get_person_id(element)}_{get_image_id(element)}"
        mapping_keypoints_train_samples.append(f"{get_person_id(element)}_{get_image_id(element)}")

    for element in mapping_val:
        image_name = mapping_val[element].split("/")[-1]
        output[f"test_{image_name}"] = f"{get_person_id(element)}_{get_image_id(element)}"
        mapping_keypoints_test_samples.append(f"{get_person_id(element)}_{get_image_id(element)}")

    for element in output:
        keypoint_index = output[element]
        for _kp in keypoint_train_samples:
            if keypoint_index in _kp:
                output[element] = _kp
                continue 
        
        for _kp in keypoint_test_samples:
            if keypoint_index in _kp:
                output[element] = _kp 
                continue
    print(len(output))
    with open("single_module_mapping_file.json", "w") as f:
        json.dump(output, f)
    # for ele in keypoint_train_samples:
    #     format_ele = ele.split("/")[-1].replace(".npy","").lstrip("0")
    #     if format_ele not in mapping_keypoints_train_samples:
    #         print(format_ele)
    
    # for ele in keypoint_test_samples:
    #     format_ele = ele.split("/")[-1].replace(".npy","").lstrip("0")
    #     if format_ele not in mapping_keypoints_test_samples:
    #         print(format_ele)
        

if __name__ == "__main__":
    main()
