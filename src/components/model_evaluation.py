import os
import json
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


class ModelEvaluation:
    def __init__(self, config_path: str):
        self.artifact_dir = "artifacts/model_evaluation"
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.model_name = "CustomerChurnModel"
        self.client = MlflowClient()

    def run(self, model, X_test, y_test):
        """
        Performs threshold tuning using nested MLflow runs,
        selects best threshold, logs artifacts,
        registers model, and promotes to STAGING (Option A).
        """

        # -----------------------------
        # 1️⃣ Predict probabilities
        # -----------------------------
        y_prob = model.predict_proba(X_test)[:, 1]

        thresholds = np.arange(0.3, 0.81, 0.05)

        best_threshold = 0.5
        best_f1 = -1.0

        threshold_metrics = {}

        # -----------------------------
        # 2️⃣ Nested MLflow runs (threshold tuning)
        # -----------------------------
        for threshold in thresholds:
            with mlflow.start_run(
                run_name=f"threshold_{threshold:.2f}",
                nested=True
            ):
                y_pred = ["Yes" if p >= threshold else "No" for p in y_prob]

                precision = precision_score(
                    y_test, y_pred, pos_label="Yes", zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, pos_label="Yes", zero_division=0
                )
                f1 = f1_score(
                    y_test, y_pred, pos_label="Yes", zero_division=0
                )

                mlflow.log_param("threshold", round(threshold, 2))
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                threshold_metrics[str(round(threshold, 2))] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        # -----------------------------
        # 3️⃣ Final evaluation with best threshold
        # -----------------------------
        y_pred_final = [
            "Yes" if p >= best_threshold else "No" for p in y_prob
        ]

        accuracy = accuracy_score(y_test, y_pred_final)
        precision = precision_score(
            y_test, y_pred_final, pos_label="Yes", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred_final, pos_label="Yes", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred_final, pos_label="Yes", zero_division=0
        )

        cm = confusion_matrix(
            y_test, y_pred_final, labels=["No", "Yes"]
        )

        # -----------------------------
        # 4️⃣ Save confusion matrix
        # -----------------------------
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        )
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix (Threshold = {best_threshold:.2f})")

        cm_path = os.path.join(self.artifact_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # -----------------------------
        # 5️⃣ Save metrics JSON (CONSISTENT NAMES)
        # -----------------------------
        metrics = {
            "best_threshold": round(best_threshold, 2),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "threshold_metrics": threshold_metrics
        }

        metrics_path = os.path.join(self.artifact_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # -----------------------------
        # 6️⃣ Log metrics & artifacts
        # -----------------------------
        mlflow.log_param("best_threshold", round(best_threshold, 2))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_path)

        # -----------------------------
        # 7️⃣ Log model ONCE (CRITICAL FIX)
        # -----------------------------
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        # -----------------------------
        # 8️⃣ Register model (ONCE)
        # -----------------------------
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name
        )

        model_version = registered_model.version

        # -----------------------------
        # 9️⃣ Store version-level metadata
        # -----------------------------
        self.client.set_model_version_tag(
            self.model_name,
            model_version,
            "f1_score",
            str(f1)
        )
        self.client.set_model_version_tag(
            self.model_name,
            model_version,
            "best_threshold",
            str(round(best_threshold, 2))
        )

        # -----------------------------
        # 🔟 Promote to STAGING (Option A)
        # -----------------------------
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )

        return metrics

#---------------------------------------------------------------
# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     ConfusionMatrixDisplay
# )


# class ModelEvaluation:
#     def __init__(self, config_path: str):
#         self.artifact_dir = "artifacts/model_evaluation"
#         os.makedirs(self.artifact_dir, exist_ok=True)

#     def run(self, model, X_test, y_test):
#         """
#         Performs threshold tuning, evaluates model,
#         saves metrics + confusion matrix, and returns metrics.
#         """

#         # -----------------------------
#         # 1️⃣ Predict probabilities
#         # -----------------------------
#         y_prob = model.predict_proba(X_test)[:, 1]

#         thresholds = np.arange(0.3, 0.81, 0.05)

#         best_threshold = 0.5
#         best_f1 = -1

#         threshold_results = {}

#         # -----------------------------
#         # 2️⃣ Threshold tuning loop
#         # -----------------------------
#         for threshold in thresholds:
#             y_pred = ["Yes" if p >= threshold else "No" for p in y_prob]

#             precision = precision_score(
#                 y_test, y_pred, pos_label="Yes", zero_division=0
#             )
#             recall = recall_score(
#                 y_test, y_pred, pos_label="Yes", zero_division=0
#             )
#             f1 = f1_score(
#                 y_test, y_pred, pos_label="Yes", zero_division=0
#             )

#             threshold_results[str(round(threshold, 2))] = {
#                 "precision": precision,
#                 "recall": recall,
#                 "f1_score": f1
#             }

#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold

#         # -----------------------------
#         # 3️⃣ Final prediction with best threshold
#         # -----------------------------
#         y_pred_final = [
#             "Yes" if p >= best_threshold else "No" for p in y_prob
#         ]

#         accuracy = accuracy_score(y_test, y_pred_final)
#         precision = precision_score(
#             y_test, y_pred_final, pos_label="Yes", zero_division=0
#         )
#         recall = recall_score(
#             y_test, y_pred_final, pos_label="Yes", zero_division=0
#         )
#         f1 = f1_score(
#             y_test, y_pred_final, pos_label="Yes", zero_division=0
#         )

#         cm = confusion_matrix(
#             y_test, y_pred_final, labels=["No", "Yes"]
#         )

#         # -----------------------------
#         # 4️⃣ Save confusion matrix PNG
#         # -----------------------------
#         disp = ConfusionMatrixDisplay(
#             confusion_matrix=cm,
#             display_labels=["No", "Yes"]
#         )
#         disp.plot(cmap="Blues")
#         plt.title(f"Confusion Matrix (Threshold = {best_threshold:.2f})")

#         cm_path = os.path.join(self.artifact_dir, "confusion_matrix.png")
#         plt.savefig(cm_path)
#         plt.close()

#         # -----------------------------
#         # 5️⃣ Save metrics JSON
#         # -----------------------------
#         metrics = {
#             "best_threshold": round(best_threshold, 2),
#             "accuracy": accuracy,
#             "precision_score": precision,
#             "recall_score": recall,
#             "f1_score": f1,
#             "confusion_matrix": cm.tolist(),
#             "threshold_tuning": threshold_results
#         }

#         metrics_path = os.path.join(self.artifact_dir, "metrics.json")
#         with open(metrics_path, "w") as f:
#             json.dump(metrics, f, indent=4)

#         return metrics

#---------------------------------------------------------------
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