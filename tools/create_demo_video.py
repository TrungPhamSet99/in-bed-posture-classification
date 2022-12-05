import cv2
import numpy as np
import glob
import argparse
import sys
import os

MANUAL = """
Run script to create demo video for HRNet output 

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
	parser.add_argument('-i', '--visualize-image-folder', type=str, required=True,
						help='Path to visualize image folder')
	parser.add_argument('-o', '--output', type=str, required=True,
						help='Path to save demo vide')
	return parser

def get_frame_id(img):
	mark=0
	img=img[22:-1]
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

def main():
	parser = parse_argument()
	args = parser.parse_args()
	if not os.path.isdir(args.visualize_image_folder):
		print('No such input visualize image folder {}'.format(args.visualize_image_folder))
		sys.exit(1)
	img_list = [0]*(len(glob.glob(args.visualize_image_folder+'/*png'))) 
	print(len(img_list))
	ids = []
	for img in glob.glob(args.visualize_image_folder+'/*.png'):
		img_name = os.path.basename(img)
		index = int(get_frame_id(img_name))
		ids.append(index)
		img_list[index] = img
	img_array = []
	for filename in img_list:
		print(filename)
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)
	out = cv2.VideoWriter('test_demo.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	

if __name__ == "__main__":
	main()
