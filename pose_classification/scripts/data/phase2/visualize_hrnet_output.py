import os 
import cv2 
import json
import argparse
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
    parser.add_argument("--hrnet-output-file", type=str,
                        help='Path to training config file', default="./json_files/hrnet_test_output.json")            
    parser.add_argument('--output-dir', type=str,
                        help='Path to training config file', default="./vis/")
    return parser

def get_pose_by_image_id(hrnet_output, image_id):
    for element in hrnet_output:
        if element['image_id'] == int(image_id):
            return element['keypoints']
        
def visualize_pose_on_image(original_image, hrnet_pose):
    for i in [0,5]:
        x = int(hrnet_pose[2*i])
        y = int(hrnet_pose[2*i + 1])
        image = cv2.circle(original_image, (x,y), 2, (255,0,0), 1)
    return image
        
def main():
    parser = parse_argument()
    args = parser.parse_args()
    image_dir = args.image_dir
    hrnet_output_file = args.hrnet_output_file
    output_dir = args.output_dir

    hrnet_output = json.load(open(hrnet_output_file))
    images = os.listdir(image_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img in images:
        img_id = img.lstrip("0").replace(".png", "")
        if img_id == "":
            img_id = "0"
        original_image = cv2.imread(os.path.join(image_dir, img))
        pose = get_pose_by_image_id(hrnet_output, img_id)
        print(pose)
        pose = [element for element in pose if element > 2]
        print(len(pose))
        x_list = []
        y_list = []
        for i, element in enumerate(pose):
            if i in list(range(0,27,2)):
                x_list.append(element)
            else:
                y_list.append(element)
        vis_image = visualize_pose_on_image(original_image, pose)
        save_path = os.path.join(output_dir, img)
        
        pose = np.asarray([[x_list, y_list]])
        print(pose)
        cv2.imwrite(save_path, vis_image)
        break
if __name__ == "__main__":
    main()