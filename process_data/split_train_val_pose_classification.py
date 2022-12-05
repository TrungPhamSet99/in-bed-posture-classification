import os
import numpy as np 
from sklearn.model_selection import train_test_split
import argparse
import random

POSTURE_CLASSES = ['supine', 'lying_left', 'lying_right']

MANUAL = """
Run script to split train/test data for pose classification

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
    parser.add_argument('-i', '--root-folder', type=str, 
                        help='Path to input data folder', default="../../../POSESLP")
    parser.add_argument('-o', '--save-folder', type=str, 
                        help='Path to output data folder', default="../../../POSESLP/")
    return parser


def main():
    parser = parse_argument()
    args = parser.parse_args()
    root_folder = args.root_folder
    save_folder = args.save_folder

    supine_samples = list()
    lying_left_samples = list()
    lying_right_samples = list()

    for c in POSTURE_CLASSES:
        sub_folder = os.path.join(root_folder,c)
        files = os.listdir(sub_folder)
        for file in files:
            eval("{}_samples".format(c)).append(os.path.join(sub_folder, file))

    supine_train, supine_test = train_test_split(supine_samples, test_size=.2, random_state=10)
    lying_right_train, lying_right_test = train_test_split(lying_right_samples, test_size=.2, random_state=10)
    lying_left_train, lying_left_test = train_test_split(lying_left_samples, test_size=.2, random_state=10)

    train_samples = supine_train + lying_left_train + lying_right_train
    test_samples = supine_test + lying_right_test + lying_left_test

    random.shuffle(train_samples)
    random.shuffle(test_samples)
    for i in range(len(train_samples)):
        train_samples[i] = train_samples[i][17:]

    for i in range(len(test_samples)):
        test_samples[i] = test_samples[i][17:]
    test_save_path = os.path.join(save_folder, "test_list.txt")
    train_save_path = os.path.join(save_folder, "train_list.txt")

    with open(train_save_path, "w") as f:
        for path in train_samples:
            f.write(path + "\n")
    with open(test_save_path, "w") as f:
        for path in test_samples:
            f.write(path + "\n")

if __name__ == "__main__":
    main()