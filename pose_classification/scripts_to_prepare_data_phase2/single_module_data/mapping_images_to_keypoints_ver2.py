import os 
import json
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


def main():
    args = get_args()
    train_mapping_file = args.mapping_file_train
    test_mapping_file = args.mapping_file_val
    keypoint_data_dir = args.keypoint_data_dir

    train_mapping = json.load(open(train_mapping_file, "r"))
    test_mapping = json.load(open(test_mapping_file, "r"))

    feature_paths = []

    for root, dirs, files in os.walk(keypoint_data_dir):
        if "Unknown" in root:
            continue
        for file in files:
            feature_paths.append(os.path.join(root.split("/")[-1], file))
    output = {}
    for sample in train_mapping:
        if "simLab" in sample:
            continue
        person_id = sample.split("/")[5].lstrip("0")
        image_name = sample.split("/")[-1]
        image_id = image_name.replace("image_", "").replace(".png", "").lstrip("0")

        hr_train_image_name = train_mapping[sample].split("/")[-1]
        key = f"train_{hr_train_image_name}"
        print(person_id, image_id)
        feature_file = [feat for feat in feature_paths if f"00{person_id}_{image_id}.npy" in feat]
        if len(feature_file):
            output[key] = feature_file[0]

    for sample in test_mapping:
        if "simLab" in sample:
            continue
        person_id = sample.split("/")[5].lstrip("0")
        image_name = sample.split("/")[-1]
        image_id = image_name.replace("image_", "").replace(".png", "").lstrip("0")

        hr_train_image_name = test_mapping[sample].split("/")[-1]
        key = f"test_{hr_train_image_name}"
        print(person_id, image_id)
        feature_file = [feat for feat in feature_paths if f"00{person_id}_{image_id}.npy" in feat]
        print(feature_file)
        if len(feature_file):
            output[key] = feature_file[0]

    with open("single_module_mapping_file.json", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()