import random 
import shutil
import os
import argparse
import time
import cv2
import json
import sys
import numpy as np

openpose_dict = {"Nose": 0, "Right Shoulder": 1, "Right Elbow": 2, "Right Wrist": 3, "Left Shoulder": 4, "Left Elbow": 5, "Left Wrist": 6, "Right Hip": 7,
				"Right Knee": 8, "Right Ankle": 9, "Left Hip": 10, "Left Knee": 11, "Left Ankle": 12, "Right Eye": 13, "Left Eye": 14, "Right Ear": 15, "Left Ear": 16}

HRNet_dict = {"Nose": 0, "Left Eye": 1, "Right Eye": 2, "Left Ear": 3, "Right Ear": 4, "Left Shoulder": 5, "Right Shoulder": 6, "Left Elbow": 7,
			  "Right Elbow": 8, "Left Wrist": 9, "Right Wrist": 10, "Left Hip": 11, "Right Hip": 12, "Left Knee": 13, "Right Knee": 14, "Left Ankle": 15, "Right Ankle": 16}

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
	parser.add_argument('-o', '--openpose-output', type=str, required=True,
						help='Path to openpose output')
	parser.add_argument('--COCO-sample-annotation', type=str, required=True,
						help='Path to COCO sample annotation')
	parser.add_argument('--TSDV-sample-annotation', type=str, required=True,
						help='Path to TSDV sample annotation')
	parser.add_argument('--annotation-output', type=str, required=True,
						help='Path to save annotation output')
	parser.add_argument('--bbox-output', type=str, required=True,
						help='Path to save bbox output')
	return parser

def get_name_by_id(id, info):
	for ele in info:
		if ele['id'] == id:
			return ele['filename']

def get_id_by_name(name, info):
	for ele in info:
		if ele['file_name'] == name:
			return ele['id']

def get_bbox_by_keypoint(keypoints):
	pose_for_action = np.array(keypoints).reshape(17,3)
	top_left_x, top_left_y = min(pose_for_action[:,0][np.nonzero(pose_for_action[:,0])]), min(pose_for_action[:,1][np.nonzero(pose_for_action[:,1])])
	w, h = max(pose_for_action[:,0])-top_left_x, max(pose_for_action[:,1])- top_left_y
	return [float(top_left_x),float(top_left_y),float(w), float(h)]

def get_frame_id_by_image_name(img):
	mark=0
	for ele in img:
		if ele == '0':
			continue
		else:
			mark = img.index(ele)
			break
	if len(img[mark:img.index('.')]) != 0:
		result = img[mark:img.index('.')]
	else:
		result = '0'
	return result

def get_keypoint_dict(openpose_output_folder):
	keypoint_file_list = os.listdir(openpose_output_folder)
	data_dict = dict()
	for file in keypoint_file_list:
		frame_id = file[0:file.index('_')]
		pose_list = list()
		data = json.load(open(os.path.join(openpose_output_folder,file)))
		people = data['people']
		for person in people:
			hrnet_format_poses = list()
			poses = person['pose_keypoints_2d'][:3] + person['pose_keypoints_2d'][6:]
			for i in range(len(poses)):
				if ((i+1)%3 == 0) and (poses[i] != 0):
					poses[i] = 2
			poses = [int(ele) for ele in poses]
			for position in list(HRNet_dict.keys()):
				openpose_id = openpose_dict[position]
				hrnet_format_poses.append(poses[openpose_id*3])
				hrnet_format_poses.append(poses[openpose_id*3+1])
				hrnet_format_poses.append(poses[openpose_id*3+2])
			pose_list.append(hrnet_format_poses)
		data_dict[frame_id] = pose_list
	return data_dict

def main():
	parser = parse_argument()
	args = parser.parse_args()
	image_folder = args.images_folder
	openpose_output_folder = args.openpose_output
	COCOsample_annotation = args.COCO_sample_annotation
	TSDVsample_annotation = args.TSDV_sample_annotation
	annotation_output = args.annotation_output
	bbox_output = args.bbox_output
	print('Checking input argument')
	if not os.path.isdir(image_folder):
		print('No such input image folder {}'.format(args.image_folder))
		sys.exit(1)
	if not os.path.isdir(openpose_output_folder):
		print('No such openpose output folder {}'.format(args.openpose_output))
		sys.exit(2)
	if not os.path.isfile(COCOsample_annotation):
		print('No such sample COCO annotation file {}'.format(args.COCO_sample_annotation))
		sys.exit(3)
	if not os.path.isfile(TSDVsample_annotation):
		print('No such sample TSDV annotation file {}'.format(args.TSDV_sample_annotation))
		sys.exit(4)
	print('Valid input parameter')
	
	TSDVSample = json.load(open(TSDVsample_annotation))
	COCOSample = json.load(open(COCOsample_annotation))
	annotation_data = dict()
	bbox_data = list()
	annotation_data['images'] = list()
	for i, img in enumerate(os.listdir(image_folder)):
		fp = os.path.join(image_folder, img)
		IMG = dict()
		frame_id = int(get_frame_id_by_image_name(img))
		IMG['license'] = 1
		IMG['width'] = cv2.imread(fp).shape[1]
		IMG['height'] = cv2.imread(fp).shape[0]
		IMG['file_name'] = img
		IMG['coco_url'] = img
		IMG['flickr_url'] = img
		IMG['date_captured'] = '2021-10-13 00:00:00'
		IMG['id'] = frame_id
		annotation_data['images'].append(IMG)

	keypoints_dict = get_keypoint_dict(openpose_output_folder)
	
	dict_keys = list(keypoints_dict.keys())
	annotation_data['annotations'] = list()
	anno_id = 0
	for image in os.listdir(image_folder):
		fp = os.path.join(image_folder, image)
		frame_id = get_frame_id_by_image_name(image)
		image_id = get_id_by_name(image, annotation_data['images'])
		for k in dict_keys:
			if k == frame_id:
				keypoints = keypoints_dict[k]
				for keypoint in keypoints:
					bbox = get_bbox_by_keypoint(keypoint)
					ANNO = dict()
					ANNO['num_keypoints'] = 17
					ANNO['iscrowd'] = 0
					ANNO['category_id'] = 1
					ANNO['id'] = anno_id
					ANNO['image_id'] = image_id
					ANNO['keypoints'] = keypoint
					ANNO['bbox'] = bbox
					ANNO['segmentation'] = [bbox]
					ANNO['area'] = bbox[2]*bbox[3]
					anno_id += 1
					annotation_data['annotations'].append(ANNO)
					BBOX = dict()
					BBOX['image_id'] = image_id
					BBOX['category_id'] = 1
					BBOX['bbox'] = bbox
					BBOX['score'] = 1
					bbox_data.append(BBOX)
	annotation_data['licenses'] = TSDVSample['licenses']
	annotation_data['categories'] = COCOSample['categories']

	with open(annotation_output, 'w') as f:
		json.dump(annotation_data, f)
	with open(bbox_output, 'w') as f:
		str = json.dumps(bbox_data)
		f.write(str)
		
if __name__ == "__main__":
    main()
