import torch
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

SLP_dict = {"Right Ankle": 0, "Right Knee": 1, "Right Hip": 2, "Left Hip": 3, "Left Knee": 4, "Left Ankle": 5, "Right Wrist": 6,
            "Right Elbow": 7, "Right Shoulder": 8, "Left Shoulder": 9, "Left Elbow": 10, "Left Wrist": 11, "Thorax": 12, "Head Top": 13}


def get_distance(lmk_from, lmk_to):
    return lmk_to - lmk_from


def get_distance_by_name(pose, name_from, name_to):
    lmk_from = pose[SLP_dict[name_from], :]
    lmk_to = pose[SLP_dict[name_to], :]
    return get_distance(lmk_from, lmk_to)


def get_center_points(pose, left_point, right_point):
    left = pose[SLP_dict[left_point], :]
    right = pose[SLP_dict[right_point], :]
    return left*0.5 + right*0.5


def get_pose_size(pose, torso_size_multiplier=2.5):
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().detach().numpy()

    hips_center = get_center_points(pose, "Left Hip", "Right Hip")
    shoulder_center = get_center_points(
        pose, "Left Shoulder", "Right Shoulder")
    torso_size = np.linalg.norm(shoulder_center - hips_center)

    pose_center_new = get_center_points(pose, "Left Hip", "Right Hip")
    d = np.take(pose - pose_center_new, 0, axis=0)
    max_dis = np.amax(np.linalg.norm(d, axis=0))
    # length of body = torso_size * torso_size_multiplier
    pose_size = np.maximum(torso_size*torso_size_multiplier, max_dis)

    return pose_size


def normalize_pose(pose):
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().detach().numpy()
    pose_center = get_center_points(pose, "Left Hip", "Right Hip")
    pose = pose - pose_center
    pose_size = get_pose_size(pose)

    pose /= pose_size
    return pose


def build_embedding_from_distance(pose):
    distance_embedding = np.array([
        get_distance_by_name(pose, "Left Shoulder", "Right Shoulder"),
        get_distance_by_name(pose, "Left Elbow", "Right Elbow"),
        get_distance_by_name(pose, "Left Wrist", "Right Wrist"),
        get_distance_by_name(pose, "Left Hip", "Right Hip"),
        get_distance_by_name(pose, "Left Knee", "Right Knee"),
        get_distance_by_name(pose, "Left Ankle", "Right Ankle"),
        get_distance_by_name(pose, "Thorax", "Left Wrist"),
        get_distance_by_name(pose, "Thorax", "Right Wrist")
    ])
    return distance_embedding


def pose_to_embedding_v1(pose):
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().detach().numpy()
    reshaped_inputs = np.reshape(pose, (14, 2))
    norm_inputs = normalize_pose(reshaped_inputs)
    return torch.from_numpy(norm_inputs.flatten())


def pose_to_embedding_v2(pose):
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().detach().numpy()
    reshaped_input = np.reshape(pose, (14, 2))
    norm_input = normalize_pose(reshaped_input)
    distance_embedding = build_embedding_from_distance(norm_input)

    # return torch.from_numpy(np.concatenate((norm_input, distance_embedding), axis=0).flatten())
    return torch.from_numpy(np.transpose(np.concatenate((norm_input, distance_embedding), axis=0)))


def load_config(path):
    return json.load(open(path, "r"))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          savepath="confusion_matrix.png"):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Computing confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

        # Visualizing
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotating the tick labels and setting their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Looping over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    print("Saving confusion matrix...")
    plt.savefig(savepath)
    return ax


def visualize_keypoint(image, keypoint):
    pass


def main():
    sample_file = "../../../POSESLP/lying_right/000030.npy"
    pose = np.load(sample_file)
    # print(pose)
    # print(pose.shape)
    tensor = torch.from_numpy(np.transpose(pose, (1, 0)))
    pose = pose_to_embedding(tensor)
    print("Normalize pose: ", pose)


if __name__ == "__main__":
    main()
