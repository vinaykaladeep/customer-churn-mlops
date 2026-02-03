import mlflow
from mlflow.tracking import MlflowClient

# 🔒 FORCE SAME TRACKING STORE AS TRAINING
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "CustomerChurnModel"
MODEL_ALIAS = "production"


def load_production_model():
    client = MlflowClient()

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    # For now: TEMP threshold so API can boot
    threshold = 0.5

    print("✅ Production model loaded successfully")
    return model, threshold


# # src/serving/model_loader.py

# import json
# import os
# import tempfile
# import mlflow
# from mlflow.tracking import MlflowClient

# # 🔒 FORCE SAME TRACKING STORE AS TRAINING
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# MODEL_NAME = "CustomerChurnModel"
# MODEL_ALIAS = "Production"


# def load_production_model():
#     model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

#     # Load ML model
#     model = mlflow.pyfunc.load_model(model_uri)

#     # Download model artifacts
#     with tempfile.TemporaryDirectory() as tmpdir:
#         local_path = mlflow.artifacts.download_artifacts(
#             artifact_uri=model_uri,
#             dst_path=tmpdir,
#         )

#         threshold_file = os.path.join(local_path, "threshold.json")

#         if not os.path.exists(threshold_file):
#             raise RuntimeError(
#                 "❌ threshold.json not found in model artifacts"
#             )

#         with open(threshold_file, "r") as f:
#             threshold = json.load(f)["threshold"]

#     print(f"✅ Loaded model @Production with threshold={threshold}")

#     return model, threshold

#---
# # src/serving/model_loader.py
# import os
# import mlflow
# from mlflow.tracking import MlflowClient

# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# MODEL_NAME = "CustomerChurnModel"
# MODEL_ALIAS = "production"  # MUST match MLflow UI exactly


# def load_production_model():
#     """
#     Loads the production model and its threshold from MLflow Model Registry.
#     Returns:
#         model: Loaded MLflow pyfunc model
#         threshold: float
#     """
#     client = MlflowClient()

#     # 1️⃣ Load model via alias
#     model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
#     model = mlflow.pyfunc.load_model(model_uri)

#     # 2️⃣ Resolve version behind alias
#     model_version = client.get_model_version_by_alias(
#         MODEL_NAME, MODEL_ALIAS
#     )

#     # 3️⃣ Fetch tags (best_threshold)
#     tags = client.get_model_version_tags(
#         MODEL_NAME, model_version.version
#     )

#     if "best_threshold" not in tags:
#         raise RuntimeError(
#             "❌ best_threshold tag missing on Production model"
#         )

#     threshold = float(tags["best_threshold"])

#     print(
#         f"✅ Loaded {MODEL_NAME} v{model_version.version} "
#         f"(threshold={threshold})"
#     )

#     return model, threshold