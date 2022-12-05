import os
import json
import argparse
import os
import numpy as np 

def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser("ABC")
    parser.add_argument('-m', '--mapping-file', type=str, 
                        help='Path to mapping JSON file', default="val_SLP_path.json")
    parser.add_argument('-i', '--JSON_input', type=str, 
                        help='Path to input JSON file', default="../output/coco/pose_hrnet/w32_256x192_adam_lr1e-3/results/keypoints_slp_val_results_0.json")
    parser.add_argument('-o', '--output-folder', type=str, 
                        help='Path to output numpy folder', default="../output/coco/pose_hrnet/w32_256x192_adam_lr1e-3/results/numpy_total/")
    parser.add_argument('-e', '--merged-file', type=str, 
                        help='Path to merged info file', default="../../../script_to_process_SLP/everything.json")
    return parser

def format_keypints(keypoints):
    filtered_list = []
    for i in range(14):
        filtered_list.append(keypoints[3*i])
    for i in range(14):
        filtered_list.append(keypoints[3*i+1])
    return np.reshape(filtered_list, (2,14))

def get_type(mapping, image_id):
    for k in mapping.keys():
        _image_id = int(mapping[k].split("/")[-1].replace(".png",""))
        if image_id == _image_id:
            image_name = int(k.split("/")[-1].replace(".png", "").replace("image_", ""))
            if image_name <= 15:
                return "supine", k
            elif image_name > 15 and image_name <= 30:
                return "lying_left", k 
            elif image_name <= 45 and image_name > 30:
                return "lying_right", k


def get_index_from_9_class(merged_info, original_path):
    for sample in merged_info:
        if sample == original_path:
            if merged_info[sample]['class'] != "Unknown":
                return merged_info[sample]['class']
            else:
                return 0


def main():
    parser = parse_argument()
    args = parser.parse_args()
    mapping_file = args.mapping_file
    JSON_input = args.JSON_input
    output_folder = args.output_folder
    merged_file = args.merged_file 

    merged_info = json.load(open(merged_file))
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    class_from_numpy_to_9_classes = {}

    outputs = json.load(open(JSON_input))
    mapping = json.load(open(mapping_file))
    for i, sample in enumerate(outputs):
        print("Process {0}/{1}".format(i, len(outputs)), end="\r")
        keypoints = sample['keypoints']
        posture, original_path = get_type(mapping, sample['image_id'])
        formated_keypoints = format_keypints(keypoints)
        save_path = os.path.join(output_folder, "{0}_{1}.npy".format(posture, i))
        class_index = get_index_from_9_class(merged_info, original_path)
        if "simLab" in original_path or not class_index :
            continue
        else:
            class_from_numpy_to_9_classes[save_path] = {}
            class_from_numpy_to_9_classes[save_path]["class"] = class_index 
            class_from_numpy_to_9_classes[save_path]["condition"] = original_path.split("/")[-2]
    class_list = []
    condition_list = []
    for sample in class_from_numpy_to_9_classes:
        if class_from_numpy_to_9_classes[sample]["class"] not in class_list:
            class_list.append(class_from_numpy_to_9_classes[sample]["class"])

        if class_from_numpy_to_9_classes[sample]["condition"] not in condition_list:
            condition_list.append(class_from_numpy_to_9_classes[sample]["condition"])

    with open("mapping.json", "w") as f:
        json.dump(class_from_numpy_to_9_classes, f)


if __name__=="__main__":
    main()