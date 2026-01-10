import os
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.config.configuration import ConfigurationManager
from src.components.features import build_features
from src.utils import save_pkl

class DataTransformation():
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_transformation()

    def get_data_transformation(self, df: pd.DataFrame):

        df["Returns"] = np.log(df["Close"]).diff()

        df = build_features(
            data=df,
            feature_names=self.config.features,
            include_ohlc=self.config.include_ohlc
        )

        df = df.dropna().reset_index(drop=True)

        y = df["Returns"].values.reshape(-1, 1)
        X = df.drop(columns=["Returns"]).values

        training_len = int(len(df) * self.config.train_test_split)

        X_train_raw = X[:training_len]
        y_train_raw = y[:training_len]

        X_test_raw = X[training_len - self.config.lookback:]
        y_test_raw = y[training_len - self.config.lookback:]

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = X_scaler.fit_transform(X_train_raw)
        X_test_scaled = X_scaler.transform(X_test_raw)

        y_train_scaled = y_scaler.fit_transform(y_train_raw)
        y_test_scaled = y_scaler.transform(y_test_raw)

        X_train, y_train = [], []
        for i in range(self.config.lookback, len(X_train_scaled) - self.config.horizon + 1):
            X_train.append(X_train_scaled[i - self.config.lookback:i, :])
            y_train.append(y_train_scaled[i + self.config.horizon - 1, 0])

        X_test, y_test = [], []
        for i in range(self.config.lookback, len(X_test_scaled) - self.config.horizon + 1):
            X_test.append(X_test_scaled[i - self.config.lookback:i, :])
            y_test.append(y_test_scaled[i + self.config.horizon - 1, 0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        return X_train, y_train, X_test, y_test, X_scaler, y_scaler


    def initiate_data_transformation(self, raw_array_path):
        logging.info("Initiating data transformation.")
        try:
            logging.info("Reading data.")
            df = pd.read_csv(raw_array_path)
            
            X_train, y_train, X_test, y_test, X_scaler, y_scaler = self.get_data_transformation(df)


            logging.info("Saving preprocessors.")
            save_pkl(X_scaler, self.config.preprocessor_path + "/preprocessor_x.pkl")
            save_pkl(y_scaler, self.config.preprocessor_path + "/preprocessor_y.pkl")

            logging.info("Finished data transformation.")

            return X_train, y_train, X_test, y_test, self.config.preprocessor_path
        except Exception as e:
            raise CustomException(e, sys)