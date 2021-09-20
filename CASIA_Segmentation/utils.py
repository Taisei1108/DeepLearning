import mlflow


def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard """
    mlflow.log_metric(name, value)