#!/bin/bash
export PYTHONPATH=$PYTHONPATH:'./'

if [ "$1" = "predict" ]; then
	python3 demo/predict.py
elif [ "$1" = "train" ]; then
	python3 demo/train.py
elif [ "$1" = "eval" ]; then
	python3 utils/run_end2end_evaluation.py
else
  echo "Not supported mode "
  printf "\nUsage: ./run_demo.sh mode\n"
  printf "mode is train or predict\n"
fi
