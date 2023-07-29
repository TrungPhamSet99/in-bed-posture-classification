#!/bin/bash
export PYTHONPATH=$PYTHONPATH:'./'

if [ "$1" = "predict" ]; then
	python3 main_script/core/predict.py
elif [ "$1" = "train" ]; then
	python3 main_script/core/train.py
elif [ "$1" = "end2end_eval" ]; then
	python3 lib/utils/angle_evaluate.py
elif [ "$1" = "single_eval" ]; then
	python3 api/evaluator.py
elif [ "$1" = "autoencoder" ]; then
	python3 scripts_to_prepare_data_phase2/run_ae_and_save_feature.py
elif [ "$1" = "combine" ]; then
	python3 scripts_to_prepare_data_phase2/combine_hrnet_and_ae_feature.py
elif [ "$1" = "normalize" ]; then
	python3 data/normal_dataset.py
elif [ "$1" = "visualize" ]; then
	python3 scripts_to_prepare_data_phase2/visualize_keypoints.py
else
  echo "Not supported mode "
  printf "\nUsage: ./run_demo.sh mode\n"
  printf "mode is train or predict\n"
fi
