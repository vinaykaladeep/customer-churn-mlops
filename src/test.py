# import mlflow

# deps = mlflow.pyfunc.get_model_dependencies(
#     "models:/CustomerChurnModel/Production"
# )
# print(deps)

# import mlflow
# print(mlflow.get_tracking_uri())

# from mlflow.tracking import MlflowClient

# client = MlflowClient()

# for m in client.search_registered_models():
#     print(m.name)

# from mlflow.tracking import MlflowClient

# client = MlflowClient()
# exp = client.get_experiment_by_name("customer_churn_experiment")
# runs = client.search_runs(exp.experiment_id, max_results=1)

# print(runs[0].info.run_id)

# python - << EOF
import mlflow
mlflow.pyfunc.load_model("models:/CustomerChurnModel@production")
print("MODEL LOAD OK")
# EOF

