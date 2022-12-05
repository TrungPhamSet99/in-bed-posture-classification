import random 
import shutil
import os
import argparse
import time

MANUAL = """
Run script to split original data into train/validation for HRNet 

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
    parser.add_argument('-i', '--images-folder', type=str, required=True,
                        help='Path to images folder')
    parser.add_argument('-v', '--val-folder', type=str, required=True,
                        help='Path to validation folder')
    return parser
    
def main():
    parser = parse_argument()
    args = parser.parse_args()
    image_folder = args.images_folder
    val_folder = args.val_folder
    train_folder = val_folder.replace('val','train')
    print('Checking input argument')
    if not os.path.isdir(image_folder):
        print('No such input image folder: {}'.format(args.images_folder))
    if not os.path.isdir(val_folder):
        os.makedirs(val_folder, exist_ok=True)
    
    if not os.path.isdir(train_folder):
        os.makedirs(train_folder, exist_ok=True)

    images = os.listdir(image_folder)
    
    val_images = random.sample(images, int(0.2*len(images)))
    print(len(images), len(val_images))
    for img in images:
        source = os.path.join(image_folder, img)
        if img in val_images:
            destination = os.path.join(val_folder,'0000'+img)
        else:
            destination = os.path.join(train_folder,'0000'+img)
        shutil.copyfile(source, destination)

if __name__ == "__main__":
    main()
