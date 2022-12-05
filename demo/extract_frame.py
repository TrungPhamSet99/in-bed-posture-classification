import cv2
import os
import pickle

vidcap = cv2.VideoCapture('/data2/user/datkt/Project/Age/data_video/2.mp4')
save_path = '/data2/samba/public/TrungPQ/Action-Pantry/deep-high-resolution-net.pytorch/demo/images_demo/'
if not os.path.isdir(save_path):
	os.makedirs(save_path, exist_ok=True)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(save_path+"frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
'''
img = cv2.imread('/data2/samba/public/TrungPQ/Action-Pantry/deep-high-resolution-net.pytorch/demo/images/frame0.jpg')
img = cv2.rectangle(img,(1856,791),(1920,1049),(0,0,255),thickness=3)
cv2.imwrite('test.jpg',img)


class HUMAN:
    def __init__(self, pose, frame, bbox):
        self.pose = []
        self.pose.append(pose)
        self.frame = []
        self.frame.append(frame)
        self.bbox = []
        self.bbox.append(bbox)
    def update(self, pose, frame, bbox):
        self.pose.append(pose)
        self.frame.append(frame)
        self.bbox.append(bbox)
with open('/data2/samba/public/TrungPQ/Action-Pantry/ST-GCN-TF_old/dict_human_new.pkl', 'rb') as f:
	data = pickle.load(f)
human = data[1]
print(len(human.frame),len(human.pose), len(human.bbox))
'''