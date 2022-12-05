import json
import os
import argparse
import scipy.misc
from scipy import io
import cv2
import numpy as np 

SLP_KEYPOINTS = ["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee",
                 "left_ankle", "right_wrist", "right_elbow", "right_shoulder",
                 "left_shoulder", "left_elbow", "left_wrist", "thorax", "head_top"]
SLP_CONNECTION = [[1,2], [2,3], [5,6], [4,5],
                  [7,8], [8,9], [11,12], [10,11],
                  [9,13], [10,13], [13,14]]
DATA_ROOT = "../../../HRSLPSub/"
SAVE_PATH = "person_keypoints_slp_val.json"
def parse_args():
    pass

def get_image_sample(path, id):
    image = dict()
    img = cv2.imread(path)
    image['license'] = 1
    image['file_name']=path.split("/")[-1]
    image['coco_url']=path.split("/")[-1]
    image['flickr_url']=path.split("/")[-1]
    image['id']=id
    image['width']=img.shape[1]
    image['height']=img.shape[0]
    return image

def get_keypoints(joints):
    keypoints = list()
    joints = joints.tolist()
    for i in range(len(joints[2])):
        if joints[2][i] == 0:
            joints[2][i]=2
        else:
            joints[2][i]=1
    for i in range(14):
        keypoints.append(joints[0][i])
        keypoints.append(joints[1][i])
        keypoints.append(joints[2][i])
    return keypoints

def get_bbox(joints):
    joints = joints.tolist()
    xmax = max(joints[0])
    ymax = max(joints[1])
    xmin = min(joints[0])
    ymin = min(joints[1])
    return [xmin,ymin,xmax-xmin,ymax-ymin]

def get_segment(joints):
    return [get_bbox(joints)]

def get_area(joints):
    bbox = get_bbox(joints)
    return bbox[2]*bbox[3]

def get_annotation_sample(joints, image_id, anno_id):
    anno = dict()
    anno['num_keypoints']=14
    anno['iscrowd']=0
    anno['image_id']=image_id
    anno['id']=anno_id
    anno['category_id']=1
    anno['bbox'] = get_bbox(joints)
    anno['segmentation'] = get_segment(joints)
    anno['keypoints'] = get_keypoints(joints)
    anno['area'] = get_area(joints)
    return anno

def main():
    json_file = "../data/tsdvcoco/annotations/person_keypoints_tsdv_val.json"
    json_data = json.load(open(json_file, "r"))
    coco_keys = list(json_data.keys())
    # print(json_data['images'][0])
    # print(json_data['annotations'][0])
    SLP_data = dict()
    SLP_data['images'] = list()
    SLP_data['annotations'] = list()
    SLP_data['licenses'] = json_data['licenses']
    SLP_data['categories'] = [{'supercategory': 'person',
                               'id': 1,
                               'name': 'person',
                               'keypoints': SLP_KEYPOINTS,
                               'skeleton': SLP_CONNECTION}]

    train_images_folder = os.path.join(DATA_ROOT, "images", "train")
    train_labels_folder = os.path.join(DATA_ROOT, "labels", "train")

    val_images_folder = os.path.join(DATA_ROOT, "images", "val")
    val_labels_folder = os.path.join(DATA_ROOT, "labels", "val")

    for i,image in enumerate(os.listdir(val_images_folder)):
        print("Process {0}/{1} samples {2}".format(i, len(os.listdir(val_images_folder)), image), end="\r")
        fp = os.path.join(val_images_folder, image)
        img = cv2.imread(fp)
        name = image.replace(".png", "")
        try:
            image_id = int(name.lstrip("0"))
        except:
            print(name)
            image_id = 0
        img_obj = get_image_sample(fp, image_id)
        SLP_data['images'].append(img_obj)

        label_path = os.path.join(val_labels_folder, "{0}.npy".format(name))
        joints = np.load(label_path)
        anno_obj = get_annotation_sample(joints, image_id, i)
        SLP_data['annotations'].append(anno_obj)
        
    
    with open(SAVE_PATH, "w") as f:
        json.dump(SLP_data, f)
    
    # for path in data_dict['joints_label'][:20]:
    #     dirpath = os.path.dirname(path)
    #     lab, index = dirpath.split("/")[-2], dirpath.split("/")[-1]
    #     joints_gt = io.loadmat(path)['joints_gt']
    #     n_samples = joints_gt.shape[2]
    #     for i in range(n_samples):
    #         _joints = joints_gt[:,:,i]
    #         for cat in CONDITION:
    #             image_path = os.path.join(dirpath, IMAGE_CLASS, cat, "image_%06d.png"%(i+1))
    #             vis_dir = os.path.join(VIS_ROOT, lab, index)
    #             if not os.path.exists(os.path.join(vis_dir, IMAGE_CLASS, cat)):
    #                 os.makedirs(os.path.join(vis_dir, IMAGE_CLASS, cat), exist_ok=True)
    #             save_path = os.path.join(vis_dir, IMAGE_CLASS, cat, "vis_image_%06d.png"%(i+1))
    #             image = cv2.imread(image_path)
    #             for connect in SLP_CONNECTION:
    #                 start=tuple(_joints[:2,connect[0]-1])
    #                 end=tuple(_joints[:2,connect[1]-1])
    #                 start = [int(ele) for ele in start]
    #                 end = [int(ele) for ele in end]
    #                 image = cv2.line(image, tuple(start), tuple(end), (0,255,0), 2)
    #             print(save_path)
    #             cv2.imwrite(save_path, image)


if __name__ == "__main__":

    main()