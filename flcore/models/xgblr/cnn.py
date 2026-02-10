# ## Centralized Federated XGBoost
# #### Create 1D convolutional neural network on trees prediction results.
# #### 1D kernel size == client_tree_num
# #### Make the learning rate of the tree ensembles learnable.

from collections import OrderedDict
from typing import Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanSquaredError
from flcore.metrics import calculate_metrics, find_best_threshold
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(
        self, client_num=5, client_tree_num=100, n_channel: int = 64, task_type="BINARY"
    ) -> None:
        super(CNN, self).__init__()
        n_out = 1
        self.task_type = task_type
        self.conv1d = nn.Conv1d(
            1, n_channel, kernel_size=client_tree_num, stride=client_tree_num, padding=0
        )
        self.layer_direct = nn.Linear(n_channel * client_num, n_out)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.Identity = nn.Identity()

        # Add weight initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ReLU(self.conv1d(x))
        x = x.flatten(start_dim=1)
        x = self.ReLU(x)
        if self.task_type == "BINARY":
            x = self.Sigmoid(self.layer_direct(x))
        elif self.task_type == "REG":
            x = self.Identity(self.layer_direct(x))
        return x

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [
            np.array(val.cpu().numpy(), copy=True)
            for _, val in self.state_dict().items()
        ]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        layer_dict = {}
        for k, v in zip(self.state_dict().keys(), weights):
            if v.ndim != 0:
                layer_dict[k] = torch.Tensor(np.array(v, copy=True))
        state_dict = OrderedDict(layer_dict)
        self.load_state_dict(state_dict, strict=True)


def train(
    task_type: str,
    net: CNN,
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    log_progress: bool = True,
) -> Tuple[float, float, int]:
    # Define loss and optimizer
    if task_type == "BINARY":
        criterion = nn.BCELoss()
    elif task_type == "REG":
        criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

    # Train the network
    net.train()
    total_loss, total_result, n_samples = 0.0, 0.0, 0
    pbar = (
        tqdm(iter(cycle(trainloader)), total=num_iterations, desc="TRAIN")
        if log_progress
        else iter(cycle(trainloader))
    )

    # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
    # by the network. This will be helpful later on, when partitioning the data across clients: resulting
    # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
    for i, data in zip(range(num_iterations), pbar):
        tree_outputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(tree_outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        total_loss += loss.item()
        n_samples += labels.size(0)

        if task_type == "BINARY":
            acc = Accuracy(task="binary")(outputs, labels.type(torch.int))
            total_result += acc * labels.size(0)
        elif task_type == "REG":
            mse = MeanSquaredError()(outputs, labels.type(torch.int))
            total_result += mse * labels.size(0)
        total_result = total_result.item()

        if log_progress:
            if task_type == "BINARY":
                pbar.set_postfix(
                    {
                        "train_loss": total_loss / n_samples,
                        "train_acc": total_result / n_samples,
                    }
                )
            elif task_type == "REG":
                pbar.set_postfix(
                    {
                        "train_loss": total_loss / n_samples,
                        "train_mse": total_result / n_samples,
                    }
                )
    if log_progress:
        print("\n")

    return total_loss / n_samples, total_result / n_samples, n_samples


def test(
    task_type: str,
    net: CNN,
    testloader: DataLoader,
    device: torch.device,
    valloader: DataLoader = None,
    log_progress: bool = True,
) -> Tuple[float, float, int]:
    """Evaluates the network on test data."""
    if task_type == "BINARY":
        criterion = nn.BCELoss()
    if task_type == "MULTICLASS":
        criterion = nn.CrossEntropyLoss()
    elif task_type == "REG":
        criterion = nn.MSELoss()

    net.eval()

    # Collect predictions and true labels for the entire test set, to compute metrics at the end of the epoch

    def get_pred_proba(dataloader):
        y_pred_list = []
        y_true_list = []
        total_loss, total_result, n_samples = 0.0, 0.0, 0
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="TEST") if log_progress else dataloader
            for data in pbar:
                tree_outputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(tree_outputs)
                # Collected testing loss and accuracy statistics
                total_loss += criterion(outputs, labels).item()
                n_samples += labels.size(0)
                num_classes = np.unique(labels.cpu().numpy()).size

                y_pred = outputs.cpu()
                y_true = labels.cpu()
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)

        return y_true_list, y_pred_list, total_loss, n_samples

    metrics = {}
    if valloader is not None:
        y_true_val, y_pred_proba_val, val_loss, val_n_samples = get_pred_proba(valloader)
        best_threshold = find_best_threshold(y_true_val, y_pred_proba_val, metric="balanced_accuracy")
        metrics_val = calculate_metrics(y_true_val, y_pred_proba_val, task_type=task_type, threshold=best_threshold)
        metrics_val = {f"val {key}": metrics_val[key] for key in metrics_val}
        metrics.update(metrics_val)
    else:
        best_threshold = 0.5

    # Add validation metrics to the evaluation metrics with a prefix
    y_true, y_pred_proba, total_loss, n_samples = get_pred_proba(testloader)
    metrics_test = calculate_metrics(y_true, y_pred_proba, task_type=task_type, threshold=best_threshold)
    metrics_not_tuned = calculate_metrics(y_true, y_pred_proba, task_type=task_type, threshold=0.5)
    metrics_not_tuned = {f"not tuned {key}": metrics_not_tuned[key] for key in metrics_not_tuned}
    metrics.update(metrics_test)
    metrics.update(metrics_not_tuned)

    if log_progress:
        print("\n")

    return total_loss / n_samples, metrics, n_samples


def print_model_layers(model: nn.Module) -> None:
    print(model)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
