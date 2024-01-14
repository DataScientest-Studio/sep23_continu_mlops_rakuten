import mlflow

path = "/app/MLflow/mlruns"
mlflow.set_tracking_uri("file://" + path)