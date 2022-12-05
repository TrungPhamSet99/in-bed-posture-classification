__author__ = "Trungpq"

import json
import os
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
Run script to preprocess SLP dataset for pose classification

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
    parser.add_argument('-i', '--input-folder', type=str, 
                        help='Path to input data folder', default="../../../SLP")
    parser.add_argument('-o', '--output-folder', type=str, 
                        help='Path to output data folder', default="../../../POSESLP/")
    return parser


def main():
    parser = parse_argument()
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(input_folder):
        print("No such input folder {0}".format(input_folder))
        sys.exit(0)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for c in POSTURE_CLASSSES:
        path = os.path.join(output_folder, c)
        if not os.path.exists(path):
            os.makedirs(path)

    pose_paths = list()
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if JOINTS_FILE_NAME in file and "old" not in file:
                pose_paths.append(os.path.join(root, file))

    count = 0
    for index, path in enumerate(pose_paths):
        print("Process {0}/{1} files".format(index, len(pose_paths)), end="\r")
        pose = io.loadmat(path)['joints_gt']
        for c in POSTURE_CLASSSES:
            sub_pose = pose[:,:,POSTURE_CLASSSES.index(c)*15:(POSTURE_CLASSSES.index(c)+1)*15]
            for i in range(15):
                save_path = os.path.join(output_folder, c, "%06d.npy"%count)
                _pose = sub_pose[0:2,:,i]
                np.save(save_path, _pose)
                count += 1


if __name__ == "__main__":
    main()