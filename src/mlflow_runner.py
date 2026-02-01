import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer_churn_experiment")

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation


def main():
    config_path = "config/config.yaml"

    with mlflow.start_run(run_name="logistic_regression_run"):

        # 1️⃣ Data Ingestion
        ingestion = DataIngestion(config_path)
        raw_data_path = ingestion.ingest_data()
        mlflow.log_artifact(raw_data_path, artifact_path="data_ingestion")

        # 2️⃣ Data Validation
        validation = DataValidation(config_path)
        validation_report = validation.validate_data(raw_data_path)
        mlflow.log_dict(
            validation_report,
            "data_validation/validation_report.json"
        )

        # 3️⃣ Data Transformation
        transformation = DataTransformation(config_path)
        X_train, X_test, y_train, y_test = transformation.run(raw_data_path)
        mlflow.log_artifacts(
            transformation.artifact_dir,
            artifact_path="data_transformation"
        )

        # 4️⃣ Model Training (🚨 ONLY PLACE MODEL IS LOGGED)
        trainer = ModelTraining(config_path)
        model = trainer.run(X_train, y_train)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("random_state", trainer.random_state)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="CustomerChurnModel"
        )

        # 5️⃣ Model Evaluation (NO model logging here)
        evaluator = ModelEvaluation(config_path)
        metrics = evaluator.run(model, X_test, y_test)

        # Log final metrics (consistent names)
        mlflow.log_param("best_threshold", metrics["best_threshold"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        # Log evaluation artifacts
        mlflow.log_artifact(
            f"{evaluator.artifact_dir}/confusion_matrix.png",
            artifact_path="model_evaluation"
        )

        mlflow.log_artifact(
            f"{evaluator.artifact_dir}/metrics.json",
            artifact_path="model_evaluation"
        )

        mlflow.log_dict(
            metrics["threshold_metrics"],
            "model_evaluation/threshold_metrics.json"
        )

        print("\n✅ MLflow pipeline executed successfully")


if __name__ == "__main__":
    main()