import yaml
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    ticker: str
    start_date: str
    end_date: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    train_test_ratio: float


class ConfigurationManager():
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            ticker=self.config["data_ingestion"]["ticker"],
            start_date=self.config["data_ingestion"]["start_date"],
            end_date=self.config["data_ingestion"]["end_date"],
            raw_data_path=self.config["data_ingestion"]["raw_data_path"],
            train_data_path=self.config["data_ingestion"]["train_data_path"],
            test_data_path=self.config["data_ingestion"]["test_data_path"],
            train_test_ratio=self.config["data_ingestion"]["train_test_ratio"]
        )
