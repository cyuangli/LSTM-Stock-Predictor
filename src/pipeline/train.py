import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.logger import logging
from src.exception import CustomException

class TrainingPipeline:

    def initiate_pipeline(self):
        logging.info("Initiating the training pipeline.")

        try:
            # 1️⃣ Data ingestion
            data_ingestion = DataIngestion()
            raw_path = data_ingestion.initiate_data_ingestion()

            # 2️⃣ Data transformation
            data_transformation = DataTransformation()
            X_train, y_train, X_test, y_test, _, y_scaler_path, test_prices_aligned = \
                data_transformation.initiate_data_transformation(raw_array_path=raw_path)

            # 3️⃣ Model training
            model_trainer = ModelTrainer()
            model, predictions_actual, metrics = model_trainer.initiate_model_training(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                y_scaler_path=y_scaler_path,
                test_prices_aligned=test_prices_aligned
            )

            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    tp = TrainingPipeline()
    tp.initiate_pipeline()
