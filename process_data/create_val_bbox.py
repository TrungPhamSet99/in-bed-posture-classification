import json
import os

SAMPLE_FILE = "../data/tsdvcoco/person_detection_results/bbox_train.json"
SOURCE_FILE = "../data/coco/annotations/person_keypoints_slp_val.json"
SAVE_PATH = "../data/coco/person_detection_results/bbox_val.json"

def main():
	source_data = json.load(open(SOURCE_FILE, "r"))
	sample_data = json.load(open(SAMPLE_FILE, "r"))

	result = list()
	for anno in source_data['annotations']:
		obj = dict()
		obj['image_id'] = anno['image_id']
		obj['category_id'] = anno['category_id']
		obj['bbox'] = anno['bbox']
		obj['score'] = 1
		result.append(obj)
	with open(SAVE_PATH, "w") as f:
		json.dump(result, f)

if __name__ == "__main__":
	main()