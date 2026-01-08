import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.config.configuration import ConfigurationManager
from src.components.features import build_features

class DataTransformation():
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_transformation()
    def initiate_data_transformation(self, raw_array_path):
        logging.info("Initiating data transformation.")
        try:
            logging.info("Reading data.")
            df = pd.read_csv(raw_array_path)

            build_features(data=df,
                           feature_names=self.config.features
                           )
            
        except Exception as e:
            raise CustomException(e, sys)