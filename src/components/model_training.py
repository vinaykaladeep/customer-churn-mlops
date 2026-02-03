# src/components/model_training.py

import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from src.utils.common import read_yaml
from src.logger import get_logger

logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, config_path: str):
        self.config = read_yaml(config_path)
        self.model_config = self.config["model_training"]

        self.artifact_dir = self.model_config["artifact_dir"]
        self.model_name = self.model_config["model_name"]
        self.random_state = self.model_config["random_state"]

        # <mlflow_metadata>
        self.input_example = None
        self.signature = None

        os.makedirs(self.artifact_dir, exist_ok=True)

    def train(self, X_train, y_train):
        logger.info("Starting model training")
        print("\n[Model Training] Started")

        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_train, y_train)

        # --------------------------------------------------
        # <mlflow_input_example>
        # X_train is numpy.ndarray → slice numpy-style
        # --------------------------------------------------
        self.input_example = X_train[:1]  # shape: (1, n_features)

        model_path = os.path.join(self.artifact_dir, self.model_name)
        joblib.dump(model, model_path)

        print("✅ [Model Training] Completed")
        print(f"✅ [Model Training] Model saved at: {model_path}")
        logger.info("Model training completed")

        return model

    def run(self, X_train, y_train):
        return self.train(X_train, y_train)


# # src/components/model_training.py

# import os
# import joblib
# import pandas as pd

# from sklearn.linear_model import LogisticRegression
# from src.utils.common import read_yaml
# from src.logger import get_logger

# # <phase2_step1> MLflow imports (PREPARATION ONLY – no logging yet)
# import mlflow
# from mlflow.models.signature import infer_signature
# # </phase2_step1>

# logger = get_logger(__name__)


# class ModelTraining:
#     def __init__(self, config_path: str):
#         self.config = read_yaml(config_path)
#         self.model_config = self.config["model_training"]

#         self.artifact_dir = self.model_config["artifact_dir"]
#         self.model_name = self.model_config["model_name"]
#         self.random_state = self.model_config["random_state"]

#         os.makedirs(self.artifact_dir, exist_ok=True)

#     def train(self, X_train, y_train):
#         logger.info("Starting model training")
#         print("\n[Model Training] Started")

#         # <unchanged> Core sklearn model training
#         model = LogisticRegression(random_state=self.random_state)
#         model.fit(X_train, y_train)
#         # </unchanged>

#         # <unchanged> Save raw sklearn model locally (for debugging / backup)
#         model_path = os.path.join(self.artifact_dir, self.model_name)
#         joblib.dump(model, model_path)
#         # </unchanged>

#         print("✅ [Model Training] Completed")
#         print(f"✅ [Model Training] Model saved at: {model_path}")
#         logger.info("Model training completed")

#         # <phase2_step1>
#         # Store training schema reference (used later for MLflow signature)
#         # We DO NOT log anything yet — only prepare metadata
#         self.input_example = X_train.iloc[:1]
#         self.signature = infer_signature(
#             X_train,
#             model.predict_proba(X_train)[:, 1]
#         )
#         # </phase2_step1>

#         return model

#     def run(self, X_train, y_train):
#         return self.train(X_train, y_train)