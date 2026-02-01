from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="file:E:/vinay/MLOPS/projects/customer-churn-mlops/mlruns")

models = client.search_registered_models()

print("Registered Models:")
for m in models:
    print("-", m.name)