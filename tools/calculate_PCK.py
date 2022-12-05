import os
import json
import math
import argparse
from collections import namedtuple
import numpy as np
import sys

HRNet_dict = {"Nose": 0, "Left Eye": 1, "Right Eye": 2, "Left Ear": 3, "Right Ear": 4, "Left Shoulder": 5, "Right Shoulder": 6, "Left Elbow": 7,
			  "Right Elbow": 8, "Left Wrist": 9, "Right Wrist": 10, "Left Hip": 11, "Right Hip": 12, "Left Knee": 13, "Right Knee": 14, "Left Ankle": 15, "Right Ankle": 16}

SLP_dict = {"Right Ankle":0, "Right Knee": 1, "Right Hip": 2, "Left Hip": 3, "Left Knee": 4, "Left Ankle": 5, "Right Wrist": 6,
			"Right Elbow": 7, "Right Shoulder": 8, "Left Shoulder": 9, "Left Elbow": 10, "Left Wrist": 11, "Thorax": 12, "Head Top": 13}
MANUAL = """
Run script to calculate PCK metric for pose estimation 

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
	parser.add_argument('-g', '--groundtruth-file', type=str, required=True,
						help='Path to images folder')
	parser.add_argument('-p', '--pred-file', type=str, required=True,
						help='Path to openpose output')
	return parser


def get_bbox_by_keypoint(keypoints):
	pose_for_action = np.array(keypoints).reshape(14,3)
	top_left_x, top_left_y = min(pose_for_action[:,0][np.nonzero(pose_for_action[:,0])]), min(pose_for_action[:,1][np.nonzero(pose_for_action[:,1])])
	w, h = max(pose_for_action[:,0])-top_left_x, max(pose_for_action[:,1])- top_left_y
	return [float(top_left_x), float(top_left_y), float(w), float(h)]


def gtImageId2Anns(json_file_name):
	results = {}
	with open(os.path.join(json_file_name), 'r') as f:
		data = json.load(f)
	for ann in data['annotations']:
		if ann['image_id'] not in results:
			results[ann['image_id']] = []
		results[ann['image_id']].append({'bbox': ann['bbox'], 'keypoints': ann['keypoints'], 'image_id': ann['image_id']})
	return results


def predImageId2Anns(json_file_name):
	results = {}
	with open(os.path.join(json_file_name), 'r') as f:
		data = json.load(f)
	for ann in data:
		bbox = get_bbox_by_keypoint(ann['keypoints'])
		if ann['image_id'] not in results:
			results[ann['image_id']] = []
		results[ann['image_id']].append({'bbox': bbox, 'keypoints': ann['keypoints'], 'image_id': ann['image_id']})
	return results


def getData(gt_file_name, pred_file_name):
	gt_data = gtImageId2Anns(gt_file_name)
	pred_data = predImageId2Anns(pred_file_name)

	Detection = namedtuple("Detection", ["gt", "pred"])
	results = []
	for image_index in gt_data:
		if image_index in pred_data:
			if gt_data[image_index] is not None and pred_data[image_index] is not None:
				example = Detection(gt_data[image_index], pred_data[image_index])
				results.append(example)
	return results


def IoU(boxA, boxB):
	xA = max(float(boxA[0]), float(boxB[0]))
	yA = max(float(boxA[1]), float(boxB[1]))
	xB = min(float(boxA[2]+boxA[0]), float(boxB[2]+boxB[0]))
	yB = min(float(boxA[3]+boxA[1]), float(boxB[3]+boxB[1]))

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2]+boxA[0] - boxA[0] + 1) * (boxA[1]+boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2]+boxB[0] - boxB[0] + 1) * (boxB[1]+boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou


def distance(x1, x2, w, h):
	dx = (x1[0] - x2[0]) / w
	dy = (x1[1] - x2[1]) / h
	return math.sqrt(dx*dx + dy*dy)


def TT_keypoint(Data):
	temp = 0
	temp_list = [0]*14
	for detection in Data:
		for i in range(len(detection.gt)):
			z = detection.gt[i]['keypoints'][2:51:3]
			index = -1
			for k in z:
				index = index + 1
				if k > 0:
					temp += 1
					temp_list[index] += 1
	return temp, temp_list


def mAP(gt_file_name, pred_file_name, thres):
	Data = getData(gt_file_name, pred_file_name)
	TP = 0
	FP = 0
	Precision = []
	Recall = []
	tt_bbox_gt = TT_bboxes(Data)

	for detection in Data:
		for i in range(len(detection.pred)):
			maxiou = 0
			for j in range(len(detection.gt)):
				iou = IoU(detection.gt[j]['bbox'], detection.pred[i]['bbox'])
				if iou > maxiou:
					maxiou = iou
			if maxiou > thres:
				TP = TP + 1
			else:
				# for abc in detection.gt:
				#     print("{:012d}.jpg".format(abc['image_id']))
				FP = FP + 1
		precision = TP/(TP+FP)
		recall = TP/tt_bbox_gt

		Precision.append(precision)
		Recall.append(recall)
		AP = 0
		for i in range(len(Precision)-1):
			AP = AP + (Recall[i+1]-Recall[i])*Precision[i]
	return AP*100


def TT_bboxes(Data):
	bboxes_gt = []
	for detection in Data:
		for i in range(len(detection.gt)):
			bboxes_gt.append(detection.gt[i])
	return len(bboxes_gt)


def pck(gt_file_name, pred_file_name, a):
	Data = getData(gt_file_name, pred_file_name)
	TT_correct_kp = 0
	temp = np.array([0]*14)
	TT_kp, TT_kp_list = TT_keypoint(Data)
	TT_kp_list = np.array(TT_kp_list)
	for detection in Data:
		for i in range(len(detection.gt)):
			x1 = detection.gt[i]['keypoints'][0:49:3]
			y1 = detection.gt[i]['keypoints'][1:50:3]
			z1 = detection.gt[i]['keypoints'][2:51:3]
			maxiou = 0
			maxidx = 0
			for j in range(len(detection.pred)):
				iou = IoU(detection.gt[i]['bbox'], detection.pred[j]['bbox'])
				if iou > maxiou:
					maxiou = iou
					maxidx = j
			x2 = detection.pred[maxidx]['keypoints'][0:49:3]
			y2 = detection.pred[maxidx]['keypoints'][1:50:3]
			z2 = detection.pred[maxidx]['keypoints'][2:51:3]
			width = detection.pred[maxidx]['bbox'][2]
			height = detection.pred[maxidx]['bbox'][3]

			threshold = a
			index = -1
			for k in range(len(z1)):
				index = index + 1
				if z1[k] > 0 and z2[k] > 0:
					xa1, ya1 = (int(x1[k]), int(y1[k]))
					xa2, ya2 = (int(x2[k]), int(y2[k]))
					if distance((xa1, ya1), (xa2, ya2), width, height) <= threshold:
						TT_correct_kp += 1
						temp[index] += 1

	tt_ratio = (temp/TT_kp_list).tolist()
	return TT_correct_kp/TT_kp, tt_ratio


if __name__ == '__main__':

	parser = parse_argument()
	args = parser.parse_args()
	print('Checking input argument...')
	pred_file = args.pred_file
	gt_file = args.groundtruth_file
	if not os.path.isfile(pred_file):
		print('No such input prediction file {}'.format(pred_file))
		sys.exit(1)
	if not os.path.isfile(gt_file):
		print('No such input groundtruth file {}'.format(gt_file))
		sys.exit(2)

	keypoint_list = list(SLP_dict.keys())
	alpha = 0.15
	b, c = pck(gt_file, pred_file, alpha)
	print('Alpha value: {}'.format(alpha))
	print('PCK: {}'.format(b))
	for i in range(len(keypoint_list)):
		print('PCK {0}: {1}'.format(keypoint_list[i], str(c[i])))
	
	