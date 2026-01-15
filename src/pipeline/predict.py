import os
import sys
import pandas as pd
import numpy as np
from src.utils import load_keras, load_pkl
from src.components.features import build_features
from src.exception import CustomException
from src.logger import logging

class PredictionPipeline:
    def __init__(self, model_path, X_scaler_path, y_scaler_path, features=None, include_ohlc=True, lookback=20):

        self.model_path = model_path
        self.X_scaler_path = X_scaler_path
        self.y_scaler_path = y_scaler_path
        self.features = features or []
        self.include_ohlc = include_ohlc
        self.lookback = lookback

        self._load_artifacts()

    def _load_artifacts(self):
        try:
            logging.info("Loading model and scalers for prediction.")
            self.model = load_keras(self.model_path)
            self.X_scaler = load_pkl(self.X_scaler_path)
            self.y_scaler = load_pkl(self.y_scaler_path)
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_features(self, df):

        try:
            df = build_features(df, feature_names=self.features, include_ohlc=self.include_ohlc)
            df = df.dropna().reset_index(drop=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def create_sequences(self, X_scaled):

        X_seq = []
        for i in range(self.lookback, len(X_scaled)):
            X_seq.append(X_scaled[i - self.lookback:i, :])
        return np.array(X_seq)

    def predict(self, df):

        try:
            logging.info("Preparing features for prediction.")
            df_feat = self.prepare_features(df)
            X = df_feat.drop(columns=["Returns"], errors="ignore").values

            X_scaled = self.X_scaler.transform(X)

            X_seq = self.create_sequences(X_scaled)

            y_pred_scaled = self.model.predict(X_seq)
            y_pred = self.y_scaler.inverse_transform(y_pred_scaled)

            return y_pred.flatten()

        except Exception as e:
            raise CustomException(e, sys)
