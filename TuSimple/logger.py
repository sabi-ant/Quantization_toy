import mlflow
from typing import Dict, Union
import torch
class MLflowLogger:
    def __init__(self, run_name: str = None, autostart: bool = True):
        self.run_name = run_name
        self.run = None
        self.save_chk_at =[49]
        if autostart:
            self.start_run()

    def start_run(self):
        if self.run is None:
            self.run = mlflow.start_run(run_name=self.run_name)
    
    def end_run(self):
        if self.run is not None:
            mlflow.end_run()
            self.run = None

    def log_metric(self, name: str, value: Union[int, float], step: int = None):
        mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: int = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: Union[str, int, float]):
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Union[str, int, float]]):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_image(self, name: str, image: Union[str, bytes], step: int = None):
        mlflow.log_image(image, key=name, step=step)
    
    def log_latest_state_dict(self, model, filename: str = "latest_model.pth", artifact_path: str = 'latest_model'):
        """
        Save the latest state_dict of the model and log it as an artifact.
        """
        torch.save(model.state_dict(), filename)
        mlflow.log_artifact(filename, artifact_path=artifact_path)

    def log_model_state_dict(self, epoch, model, filename: str = "model.pth", artifact_path: str = None):
        """
        Save PyTorch model's state_dict and log it as an artifact.
        """
        if epoch in self.save_chk_at:
            torch.save(model.state_dict(), filename)
            mlflow.log_artifact(filename, artifact_path=artifact_path)

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()