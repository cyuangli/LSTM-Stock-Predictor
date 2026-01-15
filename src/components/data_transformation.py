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

class DataTransformation:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_transformation()

    def get_data_transformation(self, df: pd.DataFrame):
        # Compute log returns
        df["Returns"] = np.log(df["Close"]).diff()

        # Add features
        df = build_features(
            data=df,
            feature_names=self.config.features,
            include_ohlc=self.config.include_ohlc
        )

        df = df.dropna().reset_index(drop=True)

        y = df["Returns"].values.reshape(-1, 1)
        X = df.drop(columns=["Returns"]).values
        prices = df["Close"].values  # Keep prices for financial metrics

        training_len = int(len(df) * self.config.train_test_split)

        # Split raw arrays
        X_train_raw = X[:training_len]
        y_train_raw = y[:training_len]
        X_test_raw = X[training_len - self.config.lookback:]
        y_test_raw = y[training_len - self.config.lookback:]
        test_prices_raw = prices[training_len - self.config.lookback:]

        # Scale
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = X_scaler.fit_transform(X_train_raw)
        X_test_scaled = X_scaler.transform(X_test_raw)

        y_train_scaled = y_scaler.fit_transform(y_train_raw)
        y_test_scaled = y_scaler.transform(y_test_raw)

        # Create sequences
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

        # Align test prices to predictions
        test_prices_aligned = test_prices_raw[self.config.lookback - 1 + self.config.horizon - 1:]
        test_prices_aligned = test_prices_aligned[:len(y_test)]

        return X_train, y_train, X_test, y_test, X_scaler, y_scaler, test_prices_aligned

    def initiate_data_transformation(self, raw_array_path):
        logging.info("Initiating data transformation.")
        try:
            df = pd.read_csv(raw_array_path, parse_dates=["Date"])
            df = df.sort_values("Date")  # important for VWAP
            df.set_index("Date", inplace=True)
            X_train, y_train, X_test, y_test, X_scaler, y_scaler, test_prices_aligned = self.get_data_transformation(df)

            logging.info("Saving preprocessors.")
            save_pkl(self.config.X_preprocessor_path, X_scaler)
            save_pkl(self.config.y_preprocessor_path, y_scaler)

            logging.info("Finished data transformation.")
            return X_train, y_train, X_test, y_test, self.config.X_preprocessor_path, self.config.y_preprocessor_path, test_prices_aligned

        except Exception as e:
            raise CustomException(e, sys)