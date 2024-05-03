import numpy as np
import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from torchmetrics.functional.classification.precision_recall import (
    _precision_recall_reduce,
)
from torchmetrics.functional.classification.specificity import _specificity_reduce
from torchmetrics.classification.stat_scores import BinaryStatScores
from torchmetrics.regression import MeanSquaredError


class BinaryBalancedAccuracy(BinaryStatScores):
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        """Computes balanced accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()

        recall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
        )
        specificity = _specificity_reduce(
            tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average
        )

        return (recall + specificity) / 2


def get_metrics_collection(task_type="binary", device="cpu"):

    if task_type.lower() == "binary":
        return MetricCollection(
            {
                "accuracy": BinaryAccuracy().to(device),
                "precision": BinaryPrecision().to(device),
                "recall": BinaryRecall().to(device),
                "specificity": BinarySpecificity().to(device),
                "f1": BinaryF1Score().to(device),
                "balanced_accuracy": BinaryBalancedAccuracy().to(device),
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

def evaluate_metrics_aggregation_fn(eval_metrics):
    print(eval_metrics[0][1].keys())
    keys_names = eval_metrics[0][1].keys()
    keys_names = list(keys_names)

    metrics ={}

    for kn in keys_names:
        results = [ evaluate_res[kn] for _, evaluate_res in eval_metrics]
        metrics[kn] = np.mean(results)
        metrics['per client ' + kn] = results
        #print(f"Metric {kn} in aggregation evaluate: {metrics[kn]}\n")

    metrics['per client n samples'] = [res[0] for res in eval_metrics]

    return metrics