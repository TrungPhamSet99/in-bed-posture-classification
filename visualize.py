import json
import argparse
import os
import cv2

SKELETON=[[1,2],[2,3],[5,6],[4,5],[7,8],[8,9],[11,12],[10,11],[9,13],[10,13],[13,14]]

def get_args():
    parser = argparse.ArgumentParser(description='Train region detection')
    parser.add_argument('--input_image', help="Path to input image",
                        default='data/coco/images/slp_val/')
    parser.add_argument('--label', help="Path to label file",
                        default='data/coco/annotations/person_keypoints_slp_val.json')
    parser.add_argument('--pred', help="Path to output file",
                        default='conf/train/yolo_trainer_conf.json')
    return parser.parse_args()
    
def visualize_image(image, keypoints):
    keypoints = [int(ele) for ele in keypoints]
    for k in SKELETON:
        image = cv2.line(image, (keypoints[(k[0]-1)*3], keypoints[(k[0]-1)*3+1]), (keypoints[(k[1]-1)*3], keypoints[(k[1]-1)*3+1]),
                        (255,0,0), 2)
    for i in range(14):
        image = cv2.circle(image, (keypoints[i*3], keypoints[i*3+1]), radius=0, color=(0,255,0), thickness=-1)
    return image

def main():
    args = get_args()
    input_image = args.input_image
    label = args.label
    pred = args.pred

    if not os.path.exists("pred_visualize/"):
        os.makedirs("pred_visualize/")
    try:
        annotation = json.load(open(label))['annotations']
    except:
        annotation = json.load(open(label))
    for image in os.listdir(input_image):
        fp = os.path.join(input_image, image)
        image_id = int(image.replace(".png", ""))
        img = cv2.imread(fp)
        for anno in annotation:
            if anno['image_id'] == image_id:
                keypoints = anno['keypoints']
                vis_image = visualize_image(img, keypoints)
                save_path = os.path.join("pred_visualize/", image)
                cv2.imwrite(save_path, vis_image)

if __name__ == "__main__":
    main()



