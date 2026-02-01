import mlflow
from mlflow.tracking import MlflowClient

# -------------------------------------------------
# 🔧 MUST MATCH mlflow_runner.py EXACTLY
# -------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "CustomerChurnModel"

# Change this manually when promoting
VERSION_TO_PROMOTE = "4"   # <-- your visible version


def promote_to_production(version: str):
    client = MlflowClient()

    # -----------------------------
    # 1️⃣ Verify model version exists
    # -----------------------------
    try:
        client.get_model_version(
            name=MODEL_NAME,
            version=version
        )
    except Exception as e:
        raise RuntimeError(
            f"❌ Model '{MODEL_NAME}' version '{version}' not found.\n"
            f"Check tracking URI and version number."
        ) from e

    # -----------------------------
    # 2️⃣ Assign production alias
    # -----------------------------
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=version
    )

    print(
        f"✅ Model '{MODEL_NAME}' version {version} "
        f"successfully promoted to PRODUCTION"
    )


if __name__ == "__main__":
    promote_to_production(VERSION_TO_PROMOTE)


# import mlflow
# from mlflow.tracking import MlflowClient


# MODEL_NAME = "CustomerChurnModel"

# # ---- Promotion rules (metric guardrails) ----
# MIN_F1_SCORE = 0.80
# MIN_PRECISION = 0.75


# def promote_to_production(model_version: str):
#     """
#     Promote a specific model version to PRODUCTION using MLflow aliases.
#     Industry best practice (MLflow 2.9+ compatible).
#     """

#     client = MlflowClient()

#     # -----------------------------
#     # 1️⃣ Fetch model version details
#     # -----------------------------
#     version_info = client.get_model_version(
#         name=MODEL_NAME,
#         version=model_version
#     )

#     run_id = version_info.run_id

#     # -----------------------------
#     # 2️⃣ Fetch metrics from run
#     # -----------------------------
#     run = client.get_run(run_id)
#     metrics = run.data.metrics

#     f1 = metrics.get("f1_score")
#     precision = metrics.get("precision")

#     if f1 is None or precision is None:
#         raise ValueError("Required metrics missing for promotion")

#     # -----------------------------
#     # 3️⃣ Metric guardrails
#     # -----------------------------
#     if f1 < MIN_F1_SCORE:
#         raise ValueError(
#             f"❌ Promotion blocked: f1_score {f1:.3f} < {MIN_F1_SCORE}"
#         )

#     if precision < MIN_PRECISION:
#         raise ValueError(
#             f"❌ Promotion blocked: precision {precision:.3f} < {MIN_PRECISION}"
#         )

#     # -----------------------------
#     # 4️⃣ Assign aliases
#     # -----------------------------
#     client.set_registered_model_alias(
#         name=MODEL_NAME,
#         alias="production",
#         version=model_version
#     )

#     print(
#         f"✅ Model '{MODEL_NAME}' version {model_version} "
#         f"successfully promoted to PRODUCTION"
#     )


# if __name__ == "__main__":
#     """
#     Usage:
#     python -m src.model_promotion
#     """

#     # 👇 CHANGE ONLY THIS WHEN PROMOTING
#     VERSION_TO_PROMOTE = "1"

#     promote_to_production(VERSION_TO_PROMOTE)