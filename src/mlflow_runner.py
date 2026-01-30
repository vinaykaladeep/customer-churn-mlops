import mlflow
import mlflow.sklearn

# -------------------------------
# MLflow Configuration
# -------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer_churn_experiment")

# -------------------------------
# Pipeline Components
# -------------------------------
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation


def main():
    config_path = "config/config.yaml"

    # -------------------------------
    # Start MLflow Run
    # -------------------------------
    with mlflow.start_run(run_name="logistic_regression_run"):

        # ===============================
        # 1️⃣ Data Ingestion
        # ===============================
        ingestion = DataIngestion(config_path)
        raw_data_path = ingestion.ingest_data()

        mlflow.log_artifact(
            raw_data_path,
            artifact_path="data_ingestion"
        )

        # ===============================
        # 2️⃣ Data Validation
        # ===============================
        validation = DataValidation(config_path)
        validation_report = validation.validate_data(raw_data_path)

        mlflow.log_dict(
            validation_report,
            artifact_file="data_validation/validation_report.json"
        )

        # ===============================
        # 3️⃣ Data Transformation
        # ===============================
        transformation = DataTransformation(config_path)
        X_train, X_test, y_train, y_test = transformation.run(raw_data_path)

        mlflow.log_artifacts(
            transformation.artifact_dir,
            artifact_path="data_transformation"
        )

        # ===============================
        # 4️⃣ Model Training
        # ===============================
        trainer = ModelTraining(config_path)
        model = trainer.run(X_train, y_train)

        # Log model parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("random_state", trainer.random_state)

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CustomerChurnModel"
        )

        # ===============================
        # 5️⃣ Model Evaluation
        # ===============================
        evaluator = ModelEvaluation(config_path)
        metrics = evaluator.run(model, X_test, y_test)

        # Log metrics individually (MLflow requirement)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log evaluation artifacts (metrics.json + confusion_matrix.png)
        mlflow.log_artifacts(
            evaluator.artifact_dir,
            artifact_path="model_evaluation"
        )

        print("\n✅ MLflow pipeline executed successfully")


if __name__ == "__main__":
    main()