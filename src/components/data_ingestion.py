import os
import pandas as pd
from src.logger import get_logger
from src.utils.common import read_yaml, create_directories

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config_path: str):
        self.config = read_yaml(config_path)
        self.ingestion_config = self.config["data_ingestion"]

        create_directories([self.ingestion_config["artifact_dir"]])

    def ingest_data(self) -> str:
        """
        Reads raw CSV and saves it into artifacts folder
        """
        logger.info("Starting data ingestion")
        print("[Data Ingestion] Started")

        raw_data_path = self.ingestion_config["raw_data_path"]
        source_data_path = self.ingestion_config["source_data_path"]

        df = pd.read_csv(source_data_path)

        df.to_csv(raw_data_path, index=False)

        logger.info(f"Raw data saved at {raw_data_path}")
        print(f"[Data Ingestion] Raw data saved at: {raw_data_path}")

        return raw_data_path

# -----------------------------------------------------------------------------------
# import os
# import shutil
# from src.logger import get_logger
# from src.utils.common import read_yaml

# logger = get_logger(__name__)

# class DataIngestion:
#     def __init__(self, config_path: str):
#         self.config = read_yaml(config_path)
#         self.ingestion_config = self.config["data_ingestion"]

#     def initiate_data_ingestion(self):
#         try:
#             source_path = self.ingestion_config["source_data_path"]
#             target_path = self.ingestion_config["raw_data_path"]

#             os.makedirs(os.path.dirname(target_path), exist_ok=True)

#             shutil.copy(source_path, target_path)

#             logger.info("Data ingestion completed successfully")
            

#         except Exception as e:
#             logger.error(f"Error during data ingestion: {e}")
#             raise e