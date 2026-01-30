import os

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.logger import get_logger
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation

logger = get_logger(__name__)

def main():
    print("\n🚀 Starting Customer Churn MLOps Pipeline\n")

    # ✅ DEFINE CONFIG PATH
    config_path = os.path.join("config", "config.yaml")

    # ------------------ Data Ingestion ------------------
    print("📥 Running Data Ingestion...")
    data_ingestion = DataIngestion(config_path=config_path)
    raw_data_path = data_ingestion.ingest_data()
    print(f"✅ Data Ingested at: {raw_data_path}\n")

    # ------------------ Data Validation ------------------
    print("🔍 Running Data Validation...")
    validator = DataValidation(config_path=config_path)
    report = validator.validate_data(raw_data_path)
    print("✅ Data Validation Completed\n")

    print("📊 Validation Summary:")
    print(report)

    # ------------------ Data Transformation ------------------
    data_transformation = DataTransformation(config_path=config_path)
    X_train, X_test, y_train, y_test = data_transformation.run(data_path=raw_data_path)

    # ------------------ Data Transformation ------------------
    model_trainer = ModelTraining(config_path=config_path)
    model = model_trainer.run(X_train, y_train)

    # ------------------ Model Evaluation ------------------
    evaluator = ModelEvaluation(config_path=config_path)
    metrics = evaluator.run(model, X_test, y_test)
    print("✅🎉 Model Evaluation Completed Successfully")




    print("\n🎯 Pipeline Step Completed Successfully")

if __name__ == "__main__":
    main()