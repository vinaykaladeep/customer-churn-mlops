import os
import joblib
import pandas as pd

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

        os.makedirs(self.artifact_dir, exist_ok=True)

    def train(self, X_train, y_train):
        logger.info("Starting model training")
        print("\n[Model Training] Started")

        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_train, y_train)

        model_path = os.path.join(self.artifact_dir, self.model_name)
        joblib.dump(model, model_path)

        print("✅ [Model Training] Completed")
        print(f"✅ [Model Training] Model saved at: {model_path}")
        logger.info("Model training completed")

        return model

    def run(self, X_train, y_train):
        return self.train(X_train, y_train)