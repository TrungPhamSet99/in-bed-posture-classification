import os
import shutil
import cv2
import json
import scipy.misc
from scipy import io
import argparse
import sys
import numpy as np 
from sklearn.model_selection import train_test_split
import random
random.seed(10)

IMAGE_CLASS = "IR/"
JOINTS_FILE_NAME = "joints_gt_IR.mat"
CON = ["cover1", "cover2", "uncover"]

MANUAL = """
Run script to split SLP dataset for HRnet
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
                        help='Path to output data folder', default="../../../HRSLPSub/")
    return parser


def main():
    parser = parse_argument()
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    train_image_folder = os.path.join(output_folder, "images", "train")
    val_image_folder = os.path.join(output_folder, "images", "val")

    train_label_folder = os.path.join(output_folder, "labels", "train")
    val_label_folder = os.path.join(output_folder, "labels", "val")

    if not os.path.exists(input_folder):
        print("Input folder {0} does not exists".format(input_folder))
        sys.exit(0)
    if not os.path.exists(train_image_folder):
        os.makedirs(train_image_folder, exist_ok=True)
    if not os.path.exists(val_image_folder):
        os.makedirs(val_image_folder, exist_ok=True)
    if not os.path.exists(train_label_folder):
        os.makedirs(train_label_folder, exist_ok=True)
    if not os.path.exists(val_label_folder):
        os.makedirs(val_label_folder, exist_ok=True)

    origin_image_paths = list()
    image_classes_paths = list()
    for root, dirs, files in os.walk(input_folder):
        if IMAGE_CLASS in root:
            for file in files:
                if ".png" in file:
                    origin_image_paths.append(os.path.join(root, file))
    cover1_images = list()
    cover2_images = list()
    uncover_images = list()
    train_mapping_path = dict()
    val_mapping_path = dict()

    for path in origin_image_paths:
        if "cover1" in path:
            cover1_images.append(path)
        elif "cover2" in path:
            cover2_images.append(path)
        elif "uncover" in path:
            uncover_images.append(path)
            
    cover1_images_train, cover1_images_val = train_test_split(cover1_images, test_size=.2,random_state=10)
    cover2_images_train, cover2_images_val = train_test_split(cover2_images, test_size=.2,random_state=10)
    uncover_images_train, uncover_images_val = train_test_split(uncover_images, test_size=.2,random_state=10)

    print("Number of cover1 images for training: ", len(cover1_images_train))
    print("Number of cover2 images for training: ", len(cover2_images_train))
    print("Number of uncover images for training: ", len(uncover_images_train))

    print("Number of cover1 images for validation: ", len(cover1_images_val))
    print("Number of cover2 images for validation: ", len(cover2_images_val))
    print("Number of uncover images for validation: ", len(uncover_images_val))

    train_images = cover1_images_train + cover2_images_train + uncover_images_train
    val_images = cover1_images_val + cover2_images_val + uncover_images_val
    random.shuffle(train_images, )
    random.shuffle(val_images, )
    
    print("Number of training images: ",len(train_images))
    print("Number of validation images: ",len(val_images))

    for i, path in enumerate(train_images):
        print("Process training data {0}/{1} samples".format(i,len(train_images)), end="\r")
        dirpath = os.path.dirname(path)
        label_filename = path.split("/")[-1][6:12]
        src_label_path = os.path.join(dirpath, "labels", "{0}.npy".format(label_filename))
        des_image_path = os.path.join(train_image_folder, "%012d.png"%i)
        des_label_path = os.path.join(train_label_folder, "%012d.npy"%i)
        train_mapping_path[path] = des_image_path
        shutil.copyfile(path, des_image_path)
        shutil.copyfile(src_label_path, des_label_path)
    
    for i, path in enumerate(val_images):
        print("Process validation data {0}/{1} samples".format(i,len(val_images)), end="\r")
        dirpath = os.path.dirname(path)
        label_filename = path.split("/")[-1][6:12]
        src_label_path = os.path.join(dirpath, "labels", "{0}.npy".format(label_filename))
        des_image_path = os.path.join(val_image_folder, "%012d.png"%i)
        des_label_path = os.path.join(val_label_folder, "%012d.npy"%i)
        val_mapping_path[path] = des_image_path
        shutil.copyfile(path, des_image_path)
        shutil.copyfile(src_label_path, des_label_path)
    with open("train_SLP_path.json", "w") as f:
        json.dump(train_mapping_path, f)
    with open("val_SLP_path.json", "w") as f:
        json.dump(val_mapping_path, f)

    # for path in origin_image_paths:
    #     condition_path = os.path.dirname(path)
    #     if os.path.dirname(os.path.dirname(condition_path)) not in image_classes_paths:
    #         image_classes_paths.append(os.path.dirname(os.path.dirname(condition_path)))
    # for path in image_classes_paths:
    #     label_folders = list()
    #     joints_file = os.path.join(path, JOINTS_FILE_NAME) 
    #     joints_gt = io.loadmat(joints_file)['joints_gt']
    #     for c in CON:
    #         folder_path = os.path.join(path, IMAGE_CLASS, c, "labels")
    #         os.makedirs(folder_path, exist_ok=True)
    #         label_folders.append(folder_path)
        
    #     for i in range(joints_gt.shape[2]):
    #         joints = joints_gt[:,:,i]
    #         for path in label_folders:
    #             save_path = os.path.join(path, "%06d.npy"%(i+1))
    #             np.save(save_path, joints)

if __name__ == "__main__":
    main()