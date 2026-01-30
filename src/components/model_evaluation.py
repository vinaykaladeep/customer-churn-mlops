import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class ModelEvaluation:
    def __init__(self, config_path):
        self.artifact_dir = "artifacts/model_evaluation"
        os.makedirs(self.artifact_dir, exist_ok=True)

    def run(self, model, X_test, y_test):
        # Predictions
        y_pred = model.predict(X_test)

        # -------------------------
        # Metrics (IMPORTANT FIX)
        # -------------------------
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="Yes"),
            "recall": recall_score(y_test, y_pred, pos_label="Yes"),
            "f1_score": f1_score(y_test, y_pred, pos_label="Yes"),
        }

        # -------------------------
        # Save metrics.json
        # -------------------------
        metrics_path = os.path.join(self.artifact_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # -------------------------
        # Confusion Matrix
        # -------------------------
        cm = confusion_matrix(y_test, y_pred, labels=["No", "Yes"])

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_path = os.path.join(self.artifact_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        return metrics


# import os
# import json
# import joblib

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix
# )

# from src.utils.common import read_yaml
# from src.logger import get_logger

# logger = get_logger(__name__)


# class ModelEvaluation:
#     def __init__(self, config_path: str):
#         self.config = read_yaml(config_path)
#         self.eval_config = self.config["model_evaluation"]

#         self.artifact_dir = self.eval_config["artifact_dir"]
#         os.makedirs(self.artifact_dir, exist_ok=True)

#     def evaluate(self, model, X_test, y_test):
#         logger.info("Starting model evaluation")
#         print("\n[Model Evaluation] Started")

#         # Predictions
#         y_pred = model.predict(X_test)

#         # Metrics
#         metrics = {
#             "accuracy": accuracy_score(y_test, y_pred),
#             "precision_score":precision_score(y_test,y_pred,pos_label="Yes"),
#             "recall_score":recall_score(y_test, y_pred, pos_label="Yes"),
#             "f1_score":f1_score(y_test, y_pred, pos_label="Yes"),
#             "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
#         }

#         # Save metrics
#         metrics_path = os.path.join(self.artifact_dir, "metrics.json")
#         with open(metrics_path, "w") as f:
#             json.dump(metrics, f, indent=4)

#         # Console output (your requirement ✅)
#         print("✅ Model Evaluation Metrics:")
#         for k, v in metrics.items():
#             print(f"{k}: {v}")

#         print(f"📁 Metrics saved at: {metrics_path}")
#         logger.info(f"Evaluation metrics: {metrics}")

#         return metrics

#     def run(self, model, X_test, y_test):
#         return self.evaluate(model, X_test, y_test)