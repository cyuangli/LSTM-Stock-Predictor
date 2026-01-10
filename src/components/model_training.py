import os
import sys
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.exception import CustomException
from src.logger import logging
from src.config.configuration import ConfigurationManager
from src.utils import save_keras, save_json, load_pkl

class ModelTrainer:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_model_training()

    def build_model(self, input_shape):
        """Builds the LSTM model"""
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(keras.layers.LSTM(64, return_sequences=False))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))

        model.compile(
            optimizer="adam",
            loss="mae",
            metrics=[keras.metrics.RootMeanSquaredError()]
        )
        return model

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, y_scaler_path):
        try:
            logging.info("Loading scalers.")
            y_scaler = load_pkl(y_scaler_path)

            logging.info("Building model.")
            model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

            logging.info("Training model.")
            
            history = model.fit(
                X_train, y_train, 
                epochs=self.config.epochs, 
                batch_size=self.config.batch_size,
                verbose=1
            )

            logging.info("Generating evaluations.")
            predictions_scaled = model.predict(X_test)
            predictions_actual = y_scaler.inverse_transform(predictions_scaled)
            y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1))

            # Evaluate predictions
            mse = np.mean((predictions_actual - y_test_actual)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions_actual - y_test_actual))
            r2 = 1 - np.sum((y_test_actual - predictions_actual)**2) / np.sum((y_test_actual - np.mean(y_test_actual))**2)

            metrics = {
                "MSE": float(mse),
                "RMSE": float(rmse),
                "MAE": float(mae),
                "R2": float(r2)
            }

            logging.info("Saving trained model.")
            save_keras(model, self.config.model_save_path)

            logging.info("Saving evaluation metrics.")
            save_json(metrics, self.config.evaluations_save_path)

            logging.info("Finished model training.")

            return model, predictions_actual, metrics

        except Exception as e:
            raise CustomException(e, sys)
