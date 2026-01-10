import os
import sys
import numpy as np
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
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        model.add(keras.layers.LSTM(64, return_sequences=True))
        model.add(keras.layers.LSTM(64))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))

        model.compile(
            optimizer="adam",
            loss="mae",
            metrics=[keras.metrics.RootMeanSquaredError()]
        )
        return model

    def calculate_financial_metrics(self, predictions, actual_prices):
        # predictions and actual_prices are aligned
        pred_returns = np.diff(predictions.flatten()) / actual_prices[:-1]
        actual_returns = np.diff(np.log(actual_prices))

        cumulative_return = np.prod(1 + pred_returns) - 1
        profit_factor = pred_returns[pred_returns > 0].sum() / -pred_returns[pred_returns < 0].sum() if np.any(pred_returns < 0) else np.inf
        sharpe_ratio = np.mean(pred_returns) / np.std(pred_returns) * np.sqrt(252) if np.std(pred_returns) != 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(pred_returns.cumsum()) - pred_returns.cumsum())

        return {
            "Profit factor": float(profit_factor),
            "Cumulative return": float(cumulative_return),
            "Sharpe ratio": float(sharpe_ratio),
            "Max drawdown": float(max_drawdown)
        }

    def initiate_model_training(self, X_train, y_train, X_test, y_test, y_scaler_path, test_prices_aligned):
        try:
            logging.info("Loading scalers.")
            y_scaler = load_pkl(y_scaler_path)

            logging.info("Building model.")
            model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

            logging.info("Training model.")
            model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                verbose=1
            )

            logging.info("Generating predictions and evaluations.")
            predictions_scaled = model.predict(X_test)
            predictions_actual = y_scaler.inverse_transform(predictions_scaled)
            y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1))

            # Standard regression metrics
            mse = np.mean((predictions_actual - y_test_actual) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions_actual - y_test_actual))
            r2 = 1 - np.sum((y_test_actual - predictions_actual) ** 2) / np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)

            regression_metrics = {
                "MSE": float(mse),
                "RMSE": float(rmse),
                "MAE": float(mae),
                "R2": float(r2)
            }

            # Financial metrics
            financial_metrics = self.calculate_financial_metrics(predictions_actual, test_prices_aligned)

            all_metrics = {**regression_metrics, **financial_metrics}

            logging.info("Saving trained model and metrics.")
            save_keras(self.config.model_save_path, model)
            save_json(self.config.evaluations_save_path, all_metrics)

            logging.info("Finished model training.")
            return model, predictions_actual, all_metrics

        except Exception as e:
            raise CustomException(e, sys)
