import os
import sys
import yfinance as yf

from src.exception import CustomException
from src.logger import logging
from src.config.configuration import ConfigurationManager

class DataIngestion():
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_ingestion_config()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion.")

        try:
            logging.info("Obtaining data.")
            df = yf.download(tickers=self.config.ticker,
                             start=self.config.start_date,
                             end=self.config.end_date,
                             auto_adjust=False)
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False, header=True)
            



        except Exception as e:
            raise CustomException(e, sys)
