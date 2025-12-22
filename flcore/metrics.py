import numpy as np
import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
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


def get_metrics_collection(config):
    device = config["device"]
    if config["task"] == "classification":
        if config["n_out"] == 1: # Binaria
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

        elif config["n_out"] > 1: # Multiclase
            num_classes = config["n_out"]
            return MetricCollection(
                {
                    # Overall accuracy
                    "accuracy": MulticlassAccuracy(
                        num_classes=num_classes,
                        average="micro",
                    ).to(device),

                    # Macro metrics (robust to imbalance)
                    "precision": MulticlassPrecision(
                        num_classes=num_classes,
                        average="macro",
                    ).to(device),

                    "recall": MulticlassRecall(
                        num_classes=num_classes,
                        average="macro",
                    ).to(device),

                    "f1": MulticlassF1Score(
                        num_classes=num_classes,
                        average="macro",
                    ).to(device),
                }
            )

    elif config["task"] == "regression":
        return MetricCollection({
            "mse": MeanSquaredError().to(device),
        })

def calculate_metrics(y_true, y_pred, config):
    metrics_collection = get_metrics_collection(config)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true.tolist())
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred.tolist())
    metrics_collection.update(y_pred, y_true)

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