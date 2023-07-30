dataset_config = {
    "dataset_name": "NormalPoseDataset",
    "data_dir": "/data/users/trungpq/22A/pose_data/POSE_SLP2022_v2",
    "train_list": "/data/users/trungpq/22A/pose_data/POSE_SLP2022/lying_left_train_list.txt",
    "test_list": "/data/users/trungpq/22A/pose_data/POSE_SLP2022/lying_left_test_list.txt",
    "mapping_file_train": "/data/users/trungpq/23A/in-bed-posture-classification/pose_classification/scripts/data/phase2/single_module_data/single_module_mapping_file_train.json",
    "mapping_file_test": "/data/users/trungpq/23A/in-bed-posture-classification/pose_classification/scripts/data/phase2/single_module_data/single_module_mapping_file_test.json",
    "image_dir": "/data/users/trungpq/coco/images",
    "batch_size": 32,
    "shuffle": False,
    "num_workers": 4,
    "pin_memory": False,
    "train_transform": "ToTensor",
    "test_transform": "ToTensor",
    "augmentation_config_path": "/data/users/trungpq/23A/in-bed-posture-classification/pose_classification/cfg/augmentation.json"
}