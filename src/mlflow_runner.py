import mlflow
import mlflow.sklearn

from src.components.model_training import train_model
from components.model_evaluation import evaluate_model

def run_experiment():
    """
    Orchestrates a full MLflow experiment run:
    - trains the model
    - evaluates it
    - logs params, metrics, and model
    """

    # 1️⃣ Set experiment (creates if not exists)
    mlflow.set_experiment("customer_churn_experiment")

    # 2️⃣ Start MLflow run
    with mlflow.start_run():

        # 3️⃣ Train model
        model, params = train_model()

        # 4️⃣ Evaluate model
        metrics = evaluate_model(model)

        # 5️⃣ Log hyperparameters
        mlflow.log_params(params)

        # 6️⃣ Log evaluation metrics
        mlflow.log_metrics(metrics)

        # 7️⃣ Log trained model as MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )


if __name__ == "__main__":
    run_experiment()
