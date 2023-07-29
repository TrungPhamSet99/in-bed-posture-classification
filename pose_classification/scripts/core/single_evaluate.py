import os 
import argparse
import json 
import warnings

from api.evaluate import SingeModuleEvaluator
warnings.filterwarnings('ignore')

def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser(
        "Run script to predict pose classification model")
    parser.add_argument('--config-path', type=str,
                        help='Path to training config file', default="./cfg/eval/evaluate_config.json")
    return parser


def main():
    evaluator = SingeModuleEvaluator(args.config_path)
    evaluator.initialize()
    label, prediction = evaluator.run_evaluate()    

if __name__ == "__main__":
    main()