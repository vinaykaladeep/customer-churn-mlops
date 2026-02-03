import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.utils.common import read_yaml
from src.logger import get_logger

logger = get_logger(__name__)


class DataTransformation:
    def __init__(self, config_path: str):
        self.config = read_yaml(config_path)
        self.transformation_config = self.config["data_transformation"]

        self.artifact_dir = self.transformation_config["artifact_dir"]
        self.target_column = self.transformation_config["target_column"]

        os.makedirs(self.artifact_dir, exist_ok=True)

    def transform_data(self, data_path: str):
        logger.info("Starting data transformation")
        print("\n[Data Transformation] Started")

        # 1️⃣ Load data
        df = pd.read_csv(data_path)

        # 2️⃣ Separate features & target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # 2️⃣.5 🔥 FIX: Encode target ONCE (CRITICAL)
        # Contract: Yes → 1, No → 0
        y = y.map({"No": 0, "Yes": 1}).astype(int)

        # 3️⃣ Identify column types
        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(exclude=["object"]).columns

        # 4️⃣ Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )

        # 5️⃣ Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 6️⃣ Fit preprocessor ONLY on training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # 7️⃣ Save preprocessor
        preprocessor_path = os.path.join(self.artifact_dir, "preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)

        # 8️⃣ Save transformed datasets
        X_train_df = pd.DataFrame(
            X_train_processed.toarray()
            if hasattr(X_train_processed, "toarray")
            else X_train_processed
        )
        X_test_df = pd.DataFrame(
            X_test_processed.toarray()
            if hasattr(X_test_processed, "toarray")
            else X_test_processed
        )

        X_train_df.to_csv(os.path.join(self.artifact_dir, "X_train.csv"), index=False)
        X_test_df.to_csv(os.path.join(self.artifact_dir, "X_test.csv"), index=False)

        y_train.to_csv(os.path.join(self.artifact_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.artifact_dir, "y_test.csv"), index=False)

        print("✅ [Data Transformation] Completed")
        print(f"✅ Preprocessor saved at: {preprocessor_path}")
        logger.info("Data transformation completed")

        return X_train_processed, X_test_processed, y_train, y_test

    def run(self, data_path: str):
        return self.transform_data(data_path)


# import os
# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer

# from src.utils.common import read_yaml
# from src.logger import get_logger

# logger = get_logger(__name__)


# class DataTransformation:
#     def __init__(self, config_path: str):
#         self.config = read_yaml(config_path)
#         self.transformation_config = self.config["data_transformation"]

#         self.artifact_dir = self.transformation_config["artifact_dir"]
#         self.target_column = self.transformation_config["target_column"]

#         os.makedirs(self.artifact_dir, exist_ok=True)

#     def transform_data(self, data_path: str):
#         """
#         Performs feature transformation and train-test split
#         """

#         logger.info("Starting data transformation")
#         print("\n[Data Transformation] Started")

#         # 1️⃣ Load data
#         df = pd.read_csv(data_path)

#         # 2️⃣ Separate features & target
#         X = df.drop(columns=[self.target_column])
#         y = df[self.target_column]

#         # 3️⃣ Identify column types
#         categorical_cols = X.select_dtypes(include=["object"]).columns
#         numerical_cols = X.select_dtypes(exclude=["object"]).columns

#         # 4️⃣ Create preprocessor
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", StandardScaler(), numerical_cols),
#                 ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
#             ]
#         )

#         # 5️⃣ Train-test split (IMPORTANT: before fitting preprocessor)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         # 6️⃣ Fit ONLY on training data
#         X_train_processed = preprocessor.fit_transform(X_train)
#         X_test_processed = preprocessor.transform(X_test)

#         # 7️⃣ Save preprocessor (CRITICAL for production inference)
#         preprocessor_path = os.path.join(self.artifact_dir, "preprocessor.pkl")
#         joblib.dump(preprocessor, preprocessor_path)

#         # 8️⃣ Save transformed datasets
#         X_train_df = pd.DataFrame(
#             X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
#         )
#         X_test_df = pd.DataFrame(
#             X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed
#         )

#         X_train_df.to_csv(
#             os.path.join(self.artifact_dir, "X_train.csv"), index=False
#         )
#         X_test_df.to_csv(
#             os.path.join(self.artifact_dir, "X_test.csv"), index=False
#         )

#         y_train.to_csv(
#             os.path.join(self.artifact_dir, "y_train.csv"), index=False
#         )
#         y_test.to_csv(
#             os.path.join(self.artifact_dir, "y_test.csv"), index=False
#         )

#         print("✅ [Data Transformation] Completed")
#         print(f"✅ [Data Transformation] Preprocessor saved at: {preprocessor_path}")
#         logger.info("Data transformation completed")

#         return X_train_processed, X_test_processed, y_train, y_test

#     def run(self, data_path: str):
#         """
#         Pipeline entry point (called from main.py)
#         """
#         return self.transform_data(data_path)