# src/mlflow_runner.py

import mlflow
import mlflow.pyfunc

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer_churn_experiment")

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation


# <phase2_step2>
class ChurnPyFuncModel(mlflow.pyfunc.PythonModel):
    """
    Wraps sklearn model so predict() always returns churn probability
    """

    def __init__(self, sklearn_model):
        self.model = sklearn_model

    def predict(self, context, model_input):
        # Always return probability of positive class
        return self.model.predict_proba(model_input)[:, 1]
# </phase2_step2>


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

        # 4️⃣ Model Training
        trainer = ModelTraining(config_path)
        sklearn_model = trainer.run(X_train, y_train)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("random_state", trainer.random_state)

        # <phase2_step2>
        pyfunc_model = ChurnPyFuncModel(sklearn_model)

        mlflow.pyfunc.log_model(
            name="model",                          # ✅ new API
            python_model=pyfunc_model,
            input_example=trainer.input_example,   # ✅ from Step-1
            signature=trainer.signature,           # ✅ from Step-1
            registered_model_name="CustomerChurnModel"
        )
        # </phase2_step2>

        # 5️⃣ Model Evaluation (UNCHANGED)
        evaluator = ModelEvaluation(config_path)
        metrics = evaluator.run(sklearn_model, X_test, y_test)

        mlflow.log_param("best_threshold", metrics["best_threshold"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])

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
