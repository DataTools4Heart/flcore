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
    BinaryAUROC,
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


def get_metrics_collection(task_type="binary", device="cpu", threshold=0.5):

    if task_type.lower() == "binary":
        return MetricCollection(
            {
                "accuracy": BinaryAccuracy(threshold=threshold).to(device),
                "precision": BinaryPrecision(threshold=threshold).to(device),
                "recall": BinaryRecall(threshold=threshold).to(device),
                "specificity": BinarySpecificity(threshold=threshold).to(device),
                "f1": BinaryF1Score(threshold=threshold).to(device),
                "balanced_accuracy": BinaryBalancedAccuracy(threshold=threshold).to(device),
                "auroc": BinaryAUROC().to(device),
            }
        )
    elif task_type.lower() == "reg":
        return MetricCollection({
            "mse": MeanSquaredError().to(device),
        })


def calculate_metrics(y_true, y_pred_proba, task_type="binary", threshold=0.5):
    metrics_collection = get_metrics_collection(task_type, threshold=threshold)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true.tolist())
    if not torch.is_tensor(y_pred_proba):
        y_pred_proba = torch.tensor(y_pred_proba.tolist())
    
    # Extract probabilities for the positive class if shape>1
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 1]

    metrics_collection.update(y_pred_proba, y_true)

    metrics = metrics_collection.compute()
    metrics = {k: v.item() for k, v in metrics.items()}

    return metrics

def metrics_aggregation_fn(distributed_metrics):
    print(distributed_metrics[0][1].keys())
    keys_names = distributed_metrics[0][1].keys()
    keys_names = list(keys_names)

    metrics ={}

    for kn in keys_names:
        results = [ evaluate_res[kn] for _, evaluate_res in distributed_metrics]
        metrics[kn] = np.mean(results)
        metrics['per client ' + kn] = results
        #print(f"Metric {kn} in aggregation evaluate: {metrics[kn]}\n")

    metrics['per client n samples'] = [res[0] for res in distributed_metrics]

    return metrics