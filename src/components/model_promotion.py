from mlflow.tracking import MlflowClient

MODEL_NAME = "CustomerChurnModel"
METRIC_KEY = "roc_auc"

client = MlflowClient()

def promote_to_production():
    staging_models = client.get_latest_versions(
        name=MODEL_NAME,
        stages=["Staging"]
    )

    if not staging_models:
        print("No model in STAGING.")
        return

    staging_model = staging_models[0]
    staging_auc = float(staging_model.tags.get(METRIC_KEY, 0))

    prod_models = client.get_latest_versions(
        name=MODEL_NAME,
        stages=["Production"]
    )

    if not prod_models:
        print("No Production model. Promoting STAGING → PRODUCTION.")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        return

    prod_model = prod_models[0]
    prod_auc = float(prod_model.tags.get(METRIC_KEY, 0))

    print(f"Staging AUC: {staging_auc}")
    print(f"Production AUC: {prod_auc}")

    if staging_auc > prod_auc:
        print("Staging model is better. Promoting to PRODUCTION.")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_model.version,
            stage="Production",
            archive_existing_versions=True
        )
    else:
        print("Production model remains unchanged.")


if __name__ == "__main__":
    promote_to_production()