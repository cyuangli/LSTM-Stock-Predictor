import yaml
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    ticker: str
    start_date: str
    end_date: str
    raw_data_path: str

@dataclass
class DataTransformationConfig:
    features: list
    train_test_split: float
    lookback: int
    horizon: int
    include_ohlc: bool
    X_preprocessor_path: str
    y_preprocessor_path: str
@dataclass
class ModelTrainingConfig:
    epochs: int
    batch_size: int
    model_save_path: str
    evaluations_save_path: str

class ConfigurationManager():
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            ticker=self.config["data_ingestion"]["ticker"],
            start_date=self.config["data_ingestion"]["start_date"],
            end_date=self.config["data_ingestion"]["end_date"],
            raw_data_path=self.config["data_ingestion"]["raw_data_path"]
        )
    
    def get_data_transformation(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            features=self.config["data_transformation"]["features"],
            train_test_split=self.config["data_transformation"]["train_test_split"],
            lookback=self.config["data_transformation"]["lookback"],
            horizon=self.config["data_transformation"]["horizon"],
            include_ochl=self.config["data_transformation"]["include_ohlc"],
            X_preprocessor_path=self.config["data_transformation"]["X_preprocessor_path"],
            y_preprocessor_path=self.config["data_transformation"]["y_preprocessor_path"]
        )
    
    def get_model_training(self) -> ModelTrainingConfig:
        return ModelTrainingConfig(
            epochs=self.config["model_training"]["epochs"],
            batch_size=self.config["model_training"]["batch_size"],
            model_save_path=self.config["model_training"]["model_save_path"],
            evaluations_save_path=self.config["model_training"]["evaluations_save_path"]
        )
