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
        ✔ Nested threshold tuning
        ✔ Final evaluation in parent run
        ✔ Single model registration
        ✔ Promote to STAGING only
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
        # 3️⃣ Final evaluation (parent run)
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

        # -----------------------------
        # 4️⃣ Confusion Matrix
        # -----------------------------
        cm = confusion_matrix(
            y_test, y_pred_final, labels=["No", "Yes"]
        )

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
        # 5️⃣ Save metrics JSON
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
        # 6️⃣ Log parent metrics & artifacts
        # -----------------------------
        mlflow.log_param("best_threshold", round(best_threshold, 2))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metrics_path)

        # Get version safely
        latest_versions = self.client.get_latest_versions(
            self.model_name, stages=["None"]
        )
        model_version = latest_versions[0].version

        # -----------------------------
        #  7️⃣ Version metadata
        # -----------------------------
        self.client.set_model_version_tag(
            self.model_name, model_version, "f1_score", str(f1)
        )
        self.client.set_model_version_tag(
            self.model_name, model_version, "best_threshold", str(round(best_threshold, 2))
        )

        # -----------------------------
        # 8️⃣ Promote to STAGING only
        # -----------------------------
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )

        return metrics

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
#         saves metrics & confusion matrix, and returns metrics.
#         """

#         # -----------------------------
#         # 1️⃣ Predict probabilities
#         # -----------------------------
#         y_prob = model.predict_proba(X_test)[:, 1]
#         thresholds = np.arange(0.3, 0.81, 0.05)

#         best_threshold = 0.5
#         best_f1 = -1.0
#         threshold_metrics = {}

#         # -----------------------------
#         # 2️⃣ Threshold tuning
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

#             threshold_metrics[str(round(threshold, 2))] = {
#                 "precision": precision,
#                 "recall": recall,
#                 "f1_score": f1
#             }

#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold

#         # -----------------------------
#         # 3️⃣ Final evaluation
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

#         # -----------------------------
#         # 4️⃣ Confusion matrix
#         # -----------------------------
#         cm = confusion_matrix(
#             y_test, y_pred_final, labels=["No", "Yes"]
#         )

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
#         # 5️⃣ Save metrics (CONSISTENT NAMING)
#         # -----------------------------
#         metrics = {
#             "best_threshold": round(best_threshold, 2),
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1,
#             "threshold_metrics": threshold_metrics
#         }

#         metrics_path = os.path.join(self.artifact_dir, "metrics.json")
#         with open(metrics_path, "w") as f:
#             json.dump(metrics, f, indent=4)

#         return metrics