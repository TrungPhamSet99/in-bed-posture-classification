import os
import numpy as np
import json
import argparse
from utils.general import pose_to_embedding_v2, combine_pose_embedding_and_autoencoder_output

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
    parser.add_argument('--autoencoder-feature', type=str,
                        help='Path to training config file', default="./autoencoder_feature/test")
    parser.add_argument('--hrnet-feature', type=str,
                        help='Path to training config file', default="/data2/samba/public/TrungPQ/22B/pose_data/POSE_SLP2022")
    parser.add_argument("--output-dir", type=str,
                        help='Path to training config file', default="./combined_feature/")
    parser.add_argument("--mapping-file", type=str,
                        help='Path to training config file', default="./scripts_to_prepare_data_phase2/single_module_data/single_module_mapping_file.json")            
    return parser


def main():
    parser = parse_argument()
    args = parser.parse_args()
    autoencoder_feature_dir = args.autoencoder_feature
    hrnet_feature_dir = args.hrnet_feature
    output_dir = args.output_dir
    mapping_file = args.mapping_file

    mapping_info = json.load(open(mapping_file))
    if not os.path.exists(output_dir):
        # create output directory
        os.makedirs(output_dir, exist_ok=True)
        for i in range(1,10):
            os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True) 
    for idx, ae_file in enumerate(os.listdir(autoencoder_feature_dir)):
        print(idx)
        ae_fp = os.path.join(autoencoder_feature_dir, ae_file)
        ae_feature = np.load(open(ae_fp, "rb"))
        ae_feature = np.squeeze(ae_feature)
        query_str = f'test_{ae_file.replace(".npy", "")}'
        if query_str not in mapping_info:
            continue
        pose_feature_file = mapping_info[query_str]
        if ".npy" in pose_feature_file:
            pose_feature_fp = os.path.join(hrnet_feature_dir, pose_feature_file)
            pose_feature = np.load(open(pose_feature_fp, "rb"))
            pose_embedding = pose_to_embedding_v2(pose_feature)
            combined_feature = combine_pose_embedding_and_autoencoder_output(pose_embedding, ae_feature)
            combined_feature = np.asarray(combined_feature)
            save_path = os.path.join(output_dir, pose_feature_file)
            print(save_path)
            np.save(save_path, combined_feature)
        else:
            continue

if __name__ == "__main__":
    main()