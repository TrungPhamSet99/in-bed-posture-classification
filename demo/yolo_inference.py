from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil
import json
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import sys
sys.path.append('/data2/samba/public/TrungPQ/internal/Action-Pantry/yolov5-6.0')
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression, check_img_size, scale_coords
from utils.torch_utils import select_device
sys.path.append('../lib')
import time

# import _init_paths
import Models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from Utils.transforms import get_affine_transform



CTX = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#os.environ['DISPLAY']=':0'
weights = '/data/users/tinhvv/haar/yolov5_deepsort/yolov5/weights/best.pt'
COCO_KEYPOINT_INDEXES = {
	0: 'nose',
	1: 'left_eye',
	2: 'right_eye',
	3: 'left_ear',
	4: 'right_ear',
	5: 'left_shoulder',
	6: 'right_shoulder',
	7: 'left_elbow',
	8: 'right_elbow',
	9: 'left_wrist',
	10: 'right_wrist',
	11: 'left_hip',
	12: 'right_hip',
	13: 'left_knee',
	14: 'right_knee',
	15: 'left_ankle',
	16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
	'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
	'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
	'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
	'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
	'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
	'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
	'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
	'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
	'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
	'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_max_preds(batch_heatmaps):
	'''
	get predictions from score maps
	heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
	'''
	assert isinstance(batch_heatmaps, np.ndarray), \
		'batch_heatmaps should be numpy.ndarray'
	assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

	batch_size = batch_heatmaps.shape[0]
	num_joints = batch_heatmaps.shape[1]
	width = batch_heatmaps.shape[3]
	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
	idx = np.argmax(heatmaps_reshaped, 2)
	maxvals = np.amax(heatmaps_reshaped, 2)

	maxvals = maxvals.reshape((batch_size, num_joints, 1))
	idx = idx.reshape((batch_size, num_joints, 1))

	preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

	preds[:, :, 0] = (preds[:, :, 0]) % width
	preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

	pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
	pred_mask = pred_mask.astype(np.float32)

	preds *= pred_mask
	return preds, maxvals



def get_person_detection_boxes(model, img, device, threshold=0.5):
	stride = int(model.stride.max())
	imgsz = check_img_size([640,640], s=stride)
	if device.type !='cpu':
		model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
	Img = letterbox(img, imgsz, stride, auto=True)[0]
	Img = Img.transpose((2, 0, 1))[::-1]
	Img = np.ascontiguousarray(Img)
	Img = torch.from_numpy(Img).to('cuda:0')
	Img = Img.float()
	Img = Img/250.0
	if len(Img.shape)==3:
		Img = Img[None]
	pred = model(Img, augment=False, visualize=False)[0]  # Pass the image to the model

	return pred
	

def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
	rotation = 0

	# pose estimation transformation
	model_inputs = []
	for center, scale in zip(centers, scales):
		trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
		# Crop smaller image of people
		model_input = cv2.warpAffine(
			image,
			trans,
			(int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
			flags=cv2.INTER_LINEAR)

		# hwc -> 1chw
		model_input = transform(model_input)#.unsqueeze(0)
		model_inputs.append(model_input)

	# n * 1chw -> nchw
	model_inputs = torch.stack(model_inputs)

	# compute output heatmap
	output = pose_model(model_inputs.to(CTX))
	coords, _ = get_final_preds(
		cfg,
		output.cpu().detach().numpy(),
		np.asarray(centers),
		np.asarray(scales))

	return coords


def box_to_center_scale(box, model_image_width, model_image_height):
	"""convert a box to center,scale information required for pose transformation
	Parameters
	----------
	box : list of tuple
		list of length 2 with two tuples of floats representing
		bottom left and top right corner of a box
	model_image_width : int
	model_image_height : int

	Returns
	-------
	(numpy array, numpy array)
		Two numpy arrays, coordinates for the center of the box and the scale of the box
	"""
	center = np.zeros((2), dtype=np.float32)

	bottom_left_corner = box[0]
	top_right_corner = box[1]
	box_width = top_right_corner[0]-bottom_left_corner[0]
	box_height = top_right_corner[1]-bottom_left_corner[1]
	bottom_left_x = bottom_left_corner[0]
	bottom_left_y = bottom_left_corner[1]
	center[0] = bottom_left_x + box_width * 0.5
	center[1] = bottom_left_y + box_height * 0.5

	aspect_ratio = model_image_width * 1.0 / model_image_height
	pixel_std = 200

	if box_width > aspect_ratio * box_height:
		box_height = box_width * 1.0 / aspect_ratio
	elif box_width < aspect_ratio * box_height:
		box_width = box_height * aspect_ratio
	scale = np.array(
		[box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
		dtype=np.float32)
	if center[0] != -1:
		scale = scale * 1.25

	return center, scale


def prepare_output_dirs(prefix='/output/'):
	pose_dir = os.path.join(prefix, "pose")
	pose_json_dir = os.path.join(prefix, "pose_json")
	if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
		shutil.rmtree(pose_dir)
	if os.path.exists(pose_json_dir) and os.path.isdir(pose_json_dir):
		shutil.rmtree(pose_json_dir)
	os.makedirs(pose_dir, exist_ok=True)
	os.makedirs(pose_json_dir, exist_ok=True)
	return pose_dir, pose_json_dir


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
	# Resize and pad image while meeting stride-multiple constraints
	shape = im.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better val mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return im, ratio, (dw, dh)

def parse_args():
	parser = argparse.ArgumentParser(description='Train keypoints network')
	# general
	parser.add_argument('--cfg', type=str, required=True)
	parser.add_argument('--videoFile', type=str, required=True)
	parser.add_argument('--outputDir', type=str, default='/output/')
	parser.add_argument('--inferenceFps', type=int, default=10)
	parser.add_argument('--writeBoxFrames', action='store_true')

	parser.add_argument('opts',
						help='Modify config options using the command-line',
						default=None,
						nargs=argparse.REMAINDER)

	args = parser.parse_args()

	# args expected by supporting codebase
	args.modelDir = ''
	args.logDir = ''
	args.dataDir = ''
	args.prevModelDir = ''
	return args


def main():
	# transformation
	pose_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225]),
	])
	device = select_device('cuda:0')
	# cudnn related setting
	cudnn.benchmark = cfg.CUDNN.BENCHMARK
	torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
	torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

	args = parse_args()
	update_config(cfg, args)
	pose_dir, pose_json_dir = prepare_output_dirs(args.outputDir)
	#csv_output_rows = []

	#box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	# Load yolo model
	box_model = attempt_load(weights, map_location=device)
	box_model.eval()
	
	pose_model = eval('Models.'+cfg.MODEL.NAME+'.get_pose_net')(
		cfg, is_train=False
	)

	if cfg.TEST.MODEL_FILE:
		print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
		pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
	else:
		print('expected model defined in config at TEST.MODEL_FILE')

	pose_model.to(CTX)
	pose_model.eval()

	# Loading an video
	vidcap = cv2.VideoCapture(args.videoFile)
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	if fps < args.inferenceFps:
		print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
		exit()
	skip_frame_cnt = round(fps / args.inferenceFps)
	frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
							 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

	count = 0
	length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(length)
	while count <= length-1:
		total_now = time.time()
		ret, image_bgr = vidcap.read()

		count += 1
		print(count)

		if not ret:
			continue

		if count % skip_frame_cnt != 0:
			continue

		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

		# Clone 2 image for person detection and pose estimation
		if cfg.DATASET.COLOR_RGB:
			image_per = image_rgb.copy()
			image_pose = image_rgb.copy()
		else:
			image_per = image_bgr.copy()
			image_pose = image_bgr.copy()

		# Clone 1 image for debugging purpose
		image_debug = image_bgr.copy()

		# object detection box
		now = time.time()
		pred = get_person_detection_boxes(box_model, image_bgr, device, threshold=0.9)
		then = time.time()
		print("Find person bbox in: {} sec".format(then - now))
		
		conf_thres=0.25
		iou_thres=0.45
		classes=0
		agnostic_nms=False
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)
		
		a = torch.ones([1,3,384, 640], dtype=torch.float)
		pred_boxes = list()
		for i, det in enumerate(pred):
			if len(det):
				det[:, :4] = scale_coords(a.shape[2:], det[:, :4], image_rgb.shape).round()
				for *xyxy, conf, cls in reversed(det):
					coor = list() 
					for ele in xyxy:
						ele = ele.cpu().detach().numpy().tolist()
						coor.append(ele)
					x0y0 = (coor[0],coor[1])
					x1y1 = (coor[2],coor[3])
					pred_boxes.append([x0y0,x1y1])
		
				
		# Can not find people. Move to next frame
		if not pred_boxes:
			count += 1
			continue
		#if args.writeBoxFrames:
		for box in pred_boxes:
			box[0] = [int(ele) for ele in box[0]]
			box[1] = [int(ele) for ele in box[1]]
			cv2.rectangle(image_debug, box[0], box[1], color=(0, 255, 0),
							thickness=3)  # Draw Rectangle with the coordinates

		# pose estimation : for multiple people
		centers = []
		scales = []
		for box in pred_boxes:
			center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
			centers.append(center)
			scales.append(scale)

		now = time.time()
		pose_preds = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
		then = time.time()
		print("Find person pose in: {} sec".format(then - now))

		json_data = dict()
		json_data['version'] = 1.3
		json_data['people'] = list()

		for coords in pose_preds:
			# Draw each point on image
			person = dict()
			person['person_id'] = [-1]
			person['pose_keypoints_2d'] = list()
			person['face_keypoints_2d'] = list()
			person['hand_left_keypoints_2d'] = list()
			person['hand_right_keypoints_2d'] = list()
			person['pose_keypoints_3d'] = list()
			person['face_keypoints_3d'] = list()
			person['hand_left_keypoints_3d'] = list()
			person['hand_right_keypoints_3d'] = list()
			for coord in coords:
				x_coord, y_coord = int(coord[0]), int(coord[1])
				cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
				person['pose_keypoints_2d'].append(x_coord)
				person['pose_keypoints_2d'].append(y_coord)
				person['pose_keypoints_2d'].append(1)
			json_data['people'].append(person)
		json_output_file = os.path.join(pose_json_dir,str(count-1)+'_keypoints.json')
		with open(json_output_file, 'w') as f:
			json.dump(json_data, f)

		total_then = time.time()

		text = "{:03.2f} sec".format(total_then - total_now)
		cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
							1, (0, 0, 255), 2, cv2.LINE_AA)
		#csv_output_rows.append(new_csv_row)
		img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
		#cv2.imwrite(img_file, image_debug)
		outcap.write(image_debug)

	vidcap.release()
	outcap.release()

	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
