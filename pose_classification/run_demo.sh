#!/bin/bash
export PYTHONPATH=$PYTHONPATH:'./'

if [ "$1" = "predict" ]; then
	python3 demo/predict.py
elif [ "$1" = "train" ]; then
	python3 demo/train.py
elif [ "$1" = "end2end_eval" ]; then
	python3 utils/run_end2end_evaluation.py
elif [ "$1" = "single_eval" ]; then
	python3 api/evaluator.py
else
  echo "Not supported mode "
  printf "\nUsage: ./run_demo.sh mode\n"
  printf "mode is train or predict\n"
fi
