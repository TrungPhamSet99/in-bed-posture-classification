# -*- coding: utf-8 -*-
# author: Trung Pham (EDABK lab - HUST)
# description: Script define Focal loss for Cross Entropy as nn.Module
import os 
import logging
import sys 
from utils.general import load_config

class Logger:
    def __init__(self, config):
        if isinstance(config, str):
            self.config = load_config(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("Invalid type for config parameter")
        for attr in self.config:
            setattr(self, attr, self.config[attr])
        if os.path.exists(self.output_path):
            os.remove(self.output_path) 

        self.logger = logging.getLogger(__name__)
        if not len(self.logger.handlers):
            self.logger.handlers.clear()
            self.logger.setLevel(logging.DEBUG if self.log_verbose else logging.INFO)
            formatter = logging.Formatter(
                '[%(levelname)s] [%(asctime)s] %(message)s', '%m-%d-%Y %H:%M:%S')
            self.logger.propagate = False
            std_handler = logging.StreamHandler()
            std_handler.setLevel(self.logger.level)
            std_handler.setFormatter(formatter)

            file_handler = logging.FileHandler(self.output_path)
            file_handler.setLevel(self.logger.level)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(std_handler)
            self.logger.addHandler(file_handler)


