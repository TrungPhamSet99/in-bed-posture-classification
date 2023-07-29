import os
import json
import argparse
import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor, Compose
from utils.general import load_config, accuracy, plot_confusion_matrix, colorstr
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from data.normal_dataset import NormalPoseDataset 

CLASSES = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
OUTPUT_DIR = "./outputs/refine_modules"

class PoseRefinder:
    def __init__(self, pose, classes):
        self.pose = pose[0, :, :]
        self.classes = classes
    
    @staticmethod
    def calculate_angle(first_line, second_line):
        def slop(line):
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            return (y2-y1)/(x2-x1)
        
        def angle(first_slope, second_slope):
            return math.degrees(math.atan((second_slope-first_slope)/(1+(second_slope*first_slope))))

        first_slope = slop(first_line)
        second_slope = slop(second_line)

        return abs(angle(first_slope, second_slope))
    

    def __call__(self):
        neck_coord = self.pose[:,12]
        sub_pose = self.pose[:,:6]
        center_of_hip = (sub_pose[:,2] + sub_pose[:,5]) / 2
        base_distance = np.linalg.norm(neck_coord - center_of_hip) # Distance from neck to center of hip
        avg_length_of_legs = (np.linalg.norm(sub_pose[:,2] - sub_pose[:,0]) + np.linalg.norm(sub_pose[:,2] - sub_pose[:,0])) / 2 
        knee_distance = np.linalg.norm(sub_pose[:,1] - sub_pose[:,4]) # Distance between 2 knees
        feet_distance = np.linalg.norm(sub_pose[:,0] - sub_pose[:,5]) # Distance between 2 feet

        knee_y = (sub_pose[1][1], sub_pose[1][4])
        feet_y = (sub_pose[1][0], sub_pose[1][5])
        hip_y = (sub_pose[1][2], sub_pose[1][3])
        

        knee_and_hip_diff_y = ((knee_y[0] - hip_y[0]) + (knee_y[0] - hip_y[1])) / 2
        

        knee_x = (sub_pose[0][1], sub_pose[0][4])
        feet_x = (sub_pose[0][0], sub_pose[0][5])
        knee_and_feet_diff_x = ((knee_x[0] - feet_x[0]) + (knee_x[0] - feet_x[1])) / 2

        knee_and_feet_diff_y = ((knee_y[0] - feet_y[0]) + (knee_y[0] - feet_y[1])) / 2

        right_side = [sub_pose[0][0], sub_pose[1][0],
                    sub_pose[0][1], sub_pose[1][1],
                    sub_pose[0][2], sub_pose[1][2]]
        left_side = [sub_pose[0][3], sub_pose[1][3],
                    sub_pose[0][4], sub_pose[1][4],
                    sub_pose[0][5], sub_pose[1][5]]
        left_angle = self.calculate_angle((left_side[0], left_side[1], left_side[2], left_side[3]),
                                        (left_side[2], left_side[3], left_side[4], left_side[5]))
        
        right_angle = self.calculate_angle((right_side[0], right_side[1], right_side[2], right_side[3]),
                                        (right_side[2], right_side[3], right_side[4], right_side[5]))
        

        angle_diff = abs(left_angle - right_angle)
        if math.isnan(right_angle):
            right_angle = 0
        if math.isnan(left_angle):
            left_angle = 0

        if self.classes == ["1","2","3"]:
            angle_thresh = [35, 25]
            distance_thresh = [5, 42]
            if left_angle <= angle_thresh[0] and right_angle <= angle_thresh[0]:
                if abs(sub_pose[1][2] - sub_pose[1][3]) < distance_thresh[0] and \
                abs(sub_pose[1][0] - sub_pose[1][5]) < distance_thresh[0] and \
                avg_length_of_legs > distance_thresh[1]:
                    pred = 0
                else: 
                    if knee_distance - feet_distance < distance_thresh[0] and avg_length_of_legs > distance_thresh[1]:
                        pred = 1
                    else: 
                        pred = 2
            else:
                if (left_angle >= angle_thresh[1] and right_angle <= angle_thresh[1]) or \
                (left_angle <= angle_thresh[1] and right_angle >= angle_thresh[1]):
                    if feet_distance < 10:
                        pred = 2
                    else:
                        pred = 1
                elif (left_angle >= angle_thresh[1]) and (right_angle >= angle_thresh[1]):
                    pred = 2

        elif self.classes == ["4","5","6"] or self.classes == ["7","8","9"]:
            angle_thresh = [35, 25]
            if knee_and_hip_diff_y < 15:
                # Pred can be 1 or 2
                if knee_distance < 25 and (left_angle > 35 and right_angle > 35):
                    pred = 2
                else:
                    pred = 1
            else:
                # Pred can be 0, 1 or 2
                if knee_distance < 15 and feet_distance < 20:
                    # Pred can be 0 or 2
                    if (left_angle + right_angle) / 2 > 60:
                        if (left_angle + right_angle) / 2 > 70 or abs(knee_and_feet_diff_x) > 18:
                            pred = 2
                        else:
                            pred = 0
                    elif abs(knee_and_feet_diff_y) < 10:
                        pred = 2
                    else: 
                        pred = 0
                else:
                    if (knee_and_hip_diff_y > 25 and knee_distance < 15 and feet_distance < 20) or \
                    (left_angle < 45 and right_angle < 45 and angle_diff < 35) or \
                    (left_angle < 60 and right_angle < 60 and abs(feet_y[0] - feet_y[1]) < 10):
                        pred = 0
                    elif (left_angle > 60 and right_angle > 60 and angle_diff < 15) or \
                        (knee_distance < 20 and abs(feet_y[0] - feet_y[1]) < 10) or \
                        ((left_angle + right_angle) / 2 > 60 and knee_distance < 5):
                        pred = 2
                    else:
                        pred = 1  
                        
        # if pred != label:
        #     print(image_file, left_angle, right_angle, angle_diff, knee_distance, feet_distance, knee_y, feet_y, pred, label)
        #     cv2.imwrite(f"./vis/{label}_{image_file}", image)
        
        return pred


def get_distribution(dataloader):
    angle_left_distribution = {"0": [], "1": [], "2": []}
    angle_right_distribution = {"0": [], "1": [], "2": []}
    knee_distance_distribution = {"0": [], "1": [], "2": []}
    feet_distance_distribution = {"0": [], "1": [], "2": []}
    angle_diff_distribution = {"0": [], "1": [], "2": []}
    knee_and_hip_diff_y_distribution = {"0": [], "1": [], "2": []}

    for i, batch in dataloader:
        _, labels = batch
        # angle_left_distribution[str(labels[0])].append(output[0].item())
        # angle_right_distribution[str(labels[0])].append(output[1].item())
        # knee_distance_distribution[str(labels[0])].append(output[2].item())
        # feet_distance_distribution[str(labels[0])].append(output[3].item())
        # angle_diff_distribution[str(labels[0])].append(abs(output[1].item() - output[0].item()))
        # knee_and_hip_diff_y_distribution[str(labels[0])].append(output[4].item())

    for i in range(3):
        print(f"Label {i}")
        print("Angle diff: ", min(angle_diff_distribution[str(i)]), max(angle_diff_distribution[str(i)]), sum(angle_diff_distribution[str(i)])/len(angle_diff_distribution[str(i)]))
        print("Angle left: ", min(angle_left_distribution[str(i)]), max(angle_left_distribution[str(i)]), sum(angle_left_distribution[str(i)])/len(angle_left_distribution[str(i)]))
        print("Angle right: ", min(angle_right_distribution[str(i)]), max(angle_right_distribution[str(i)]), sum(angle_right_distribution[str(i)])/len(angle_right_distribution[str(i)]))
        print("Knee distance", min(knee_distance_distribution[str(i)]), max(knee_distance_distribution[str(i)]), sum(knee_distance_distribution[str(i)])/len(knee_distance_distribution[str(i)]))
        print("Feet distance", min(feet_distance_distribution[str(i)]), max(feet_distance_distribution[str(i)]), sum(feet_distance_distribution[str(i)])/len(feet_distance_distribution[str(i)]))
        print("Knee and hip diff y", min(knee_and_hip_diff_y_distribution[str(i)]), max(knee_and_hip_diff_y_distribution[str(i)]), sum(knee_and_hip_diff_y_distribution[str(i)])/len(knee_and_hip_diff_y_distribution[str(i)]))
        print("-------------------------------------------------------------------------------")


def main():
    config_path = "./cfg/eval/end2end_config_v2.json"
    config = load_config(config_path)
    data_config = config["data"]

    for c in CLASSES:
        print(colorstr(f"Target class: {c}"))
        out_dir = os.path.join(OUTPUT_DIR, "_".join(c))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        test_dataset = NormalPoseDataset(data_config['data_dir'],
                                     data_config['test_list'],
                                     data_config['mapping_file_test'],
                                     data_config['image_dir'],
                                     classes = c,
                                     augment_config_path=data_config['augmentation_config_path'],
                                     transform = Compose([eval(data_config["test_transform"])()]))
    
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=data_config['batch_size'],
                                                    shuffle=data_config['shuffle'],
                                                    num_workers=data_config['num_workers'],
                                                    pin_memory=data_config['pin_memory'])
        prediction = []
        labels = []
        for i, batch in enumerate(testloader):
            pose, label = batch
            output = PoseRefinder(pose, c)()

            label = label.numpy()
            labels += label.tolist()
            prediction.append(output)


        precision, recall, fscore, support = score(labels, prediction)
        report = classification_report(labels, prediction)
        print('\nprecision: {}'.format(precision))
        print('\nrecall: {}'.format(recall))
        print('\nfscore: {}'.format(fscore))
        print('\nsupport: {}\n'.format(support))
        print(report)

        plot_confusion_matrix(labels, prediction, c, normalize=False, title="Non-normalized confusion matrix (all)",
                            savepath=f"{out_dir}/non_normalize.png")
        plot_confusion_matrix(labels, prediction, c, normalize=True, title="Normalized confusion matrix (all)",
                            savepath=f"{out_dir}/normalize.png")
        print("------------------------------------------------------")
    

if __name__ == "__main__":
    main()