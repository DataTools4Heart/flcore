from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.regression import MeanSquaredError
import torch


def get_metrics_collection(task_type="binary", device="cpu"):

    if task_type.lower() == "binary":
        return MetricCollection(
            {
                "accuracy": BinaryAccuracy().to(device),
                "precision": BinaryPrecision().to(device),
                "recall": BinaryRecall().to(device),
                "specificity": BinarySpecificity().to(device),
                "f1": BinaryF1Score().to(device),
            }
        )
    elif task_type.lower() == "reg":
        return MetricCollection({
            "mse": MeanSquaredError().to(device),
        })

def calculate_metrics(y_true, y_pred, task_type="binary"):
    metrics_collection = get_metrics_collection(task_type)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true.tolist())
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred.tolist())
    metrics_collection.update(y_pred, y_true)

    metrics = metrics_collection.compute()
    metrics = {k: v.item() for k, v in metrics.items()}

    return metrics
