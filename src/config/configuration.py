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
    preprocessor_path: str

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
            preprocessor_path=self.config["data_transformation"]["preprocessor_path"]

        )
