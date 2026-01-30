import os
import pandas as pd
from src.logger import get_logger
from src.utils.common import read_yaml

logger = get_logger(__name__)

class DataValidation:
    def __init__(self, config_path: str):
        self.config = read_yaml(config_path)
        self.validation_config = self.config["data_validation"]

        # REQUIRED by config.yaml
        self.required_columns = self.validation_config["required_columns"]
        self.artifacts_dir = self.validation_config["artifacts_dir"]

        os.makedirs(self.artifacts_dir, exist_ok=True)

    def validate_data(self, data_path: str) -> dict:
        """
        Validates schema, nulls, duplicates
        """
        logger.info("Starting data validation")
        print("[Data Validation] Started")

        df = pd.read_csv(data_path)
        report = {}

        # 1️⃣ Schema check
        missing_columns = list(
            set(self.required_columns) - set(df.columns)
        )
        report["missing_columns"] = missing_columns

        # 2️⃣ Null check
        null_counts = df.isnull().sum().to_dict()
        report["null_values"] = null_counts

        # 3️⃣ Duplicate check
        duplicates = df.duplicated().sum()
        report["duplicate_rows"] = duplicates

        # Save report
        report_path = os.path.join(self.artifacts_dir, "validation_report.txt")
        with open(report_path, "w") as f:
            f.write(str(report))

        logger.info(f"Validation report saved at {report_path}")

        print("[Data Validation] Missing Columns:", missing_columns)
        print("[Data Validation] Duplicate Rows:", duplicates)
        print(f"[Data Validation] Report saved at {report_path}")

        return report

#----------------------------------------------------------------------------------------
# import os
# import pandas as pd
# from src.logger import get_logger
# from src.utils.common import read_yaml

# logger = get_logger(__name__)

# class DataValidation:
#     def __init__(self, config_path: str):
#         self.config = read_yaml(config_path)
#         self.validation_config = self.config["data_validation"]
#         self.required_columns = self.validation_config["required_columns"]
#         self.artifacts_dir = self.validation_config["artifacts_dir"]
#         os.makedirs(self.artifacts_dir, exist_ok=True)

#     def validate_data(self, data_path: str):
#         try:
#             df = pd.read_csv(data_path)
#             report = {}

#             # Check required columns
#             missing_columns = [col for col in self.required_columns if col not in df.columns]
#             report["missing_columns"] = missing_columns

#             # Check null values
#             null_counts = df.isnull().sum().to_dict()
#             report["null_counts"] = null_counts

#             # Check duplicates
#             duplicates = df.duplicated().sum()
#             report["duplicates"] = duplicates

#             # Save a simple report
#             report_path = os.path.join(self.artifacts_dir, "validation_report.txt")
#             with open(report_path, "w") as f:
#                 f.write(str(report))

#             if missing_columns:
#                 logger.error(f"Missing columns: {missing_columns}")
#             else:
#                 logger.info("All required columns are present")

#             logger.info(f"Null value summary: {null_counts}")
#             logger.info(f"Number of duplicate rows: {duplicates}")
#             logger.info(f"Validation report saved at {report_path}")

#             return report

#         except Exception as e:
#             logger.error(f"Error in data validation: {e}")
#             raise e