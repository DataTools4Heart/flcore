import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xgboost as xgb
from flwr.common import (
    NDArray,
    bytes_to_ndarray,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import Parameters
from matplotlib import pyplot as plt  # pylint: disable=E0401
from torch.utils.data import DataLoader, Dataset, random_split
from xgboost import XGBClassifier, XGBRegressor
from flcore.metrics import calculate_metrics



def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def do_fl_partitioning(
    trainset: Dataset,
    testset: Dataset,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // pool_size
    lengths = [partition_size] * pool_size
    if sum(lengths) != len(trainset):
        lengths[-1] = len(trainset) - sum(lengths[0:-1])
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = int(len(ds) * val_ratio)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(0))
        trainloaders.append(get_dataloader(ds_train, "train", batch_size))
        if len_val != 0:
            valloaders.append(get_dataloader(ds_val, "val", batch_size))
        else:
            valloaders = None
    testloader = get_dataloader(testset, "test", batch_size)
    return trainloaders, valloaders, testloader


def plot_xgbtree(tree: Union[XGBClassifier, XGBRegressor], n_tree: int) -> None:
    """Visualize the built xgboost tree."""
    xgb.plot_tree(tree, num_trees=n_tree)
    plt.rcParams["figure.figsize"] = [50, 10]
    plt.show()


def construct_tree(
    dataset: Dataset, label: NDArray, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset."""
    tree = get_tree(n_estimators, tree_type)
    tree.fit(dataset, label)
    return tree


def get_tree(n_estimators: int, tree_type: str) -> Union[XGBClassifier, XGBRegressor]:
    """Instantiate XGBoost model."""
    if tree_type == "REG":
        tree = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
        )
    else:
        if tree_type == "BINARY":
            objective = "binary:logistic"
        elif tree_type == "MULTICLASS":
            objective = "multi:softprob"
        else:
            raise ValueError("Unknown tree type.")

        tree = xgb.XGBClassifier(
            objective=objective,
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
        )

    return tree


def construct_tree_from_loader(
    dataset_loader: DataLoader, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]
    return construct_tree(data, label, n_estimators, tree_type)


def single_tree_prediction(
    tree: Union[XGBClassifier, XGBRegressor], n_tree: int, dataset: NDArray
) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the xgboost tree
    ensemble."""
    # How to access a single tree
    # https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    return tree.predict(  # type: ignore
        dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True
    )


def tree_encoding(  # pylint: disable=R0914
    trainloader: DataLoader,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients."""
    if trainloader is None:
        return None

    for local_dataset in trainloader:
        x_train, y_train = local_dataset[0], local_dataset[1]

    x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num))
    x_train_enc = np.array(x_train_enc, copy=True)

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids = []
        temp_trees = []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j in range(client_tree_num):
            predictions = single_tree_prediction(temp_trees[i], j, x_train)
            if len(predictions.shape) != 1:
                predictions = np.argmax(predictions, 1)
            x_train_enc[:, i * client_tree_num + j] = predictions
            # x_train_enc[:, i * client_tree_num + j] = single_tree_prediction(
            #     temp_trees[i], j, x_train
            # )

    x_train_enc32: Any = np.float32(x_train_enc)
    y_train32: Any = np.float32(y_train)

    x_train_enc32, y_train32 = torch.from_numpy(
        np.expand_dims(x_train_enc32, axis=1)  # type: ignore
    ), torch.from_numpy(
        np.expand_dims(y_train32, axis=-1)  # type: ignore
    )
    return x_train_enc32, y_train32


class TreeDataset(Dataset):
    def __init__(self, data: NDArray, labels: NDArray) -> None:
        self.labels = labels
        self.data = data

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[int, NDArray]:
        label = self.labels[idx]
        data = self.data[idx, :]
        sample = {0: data, 1: label}
        return sample


def tree_encoding_loader(
    dataloader: DataLoader,
    batch_size: int,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
) -> DataLoader:
    encoding = tree_encoding(dataloader, client_trees, client_tree_num, client_num)
    if encoding is None:
        return None
    data, labels = encoding
    tree_dataset = TreeDataset(data, labels)
    return get_dataloader(tree_dataset, "tree", batch_size)


def serialize_objects_to_parameters(objects_list: List, tmp_dir="") -> Parameters:
    net_weights = objects_list[0]
    if type(net_weights) is Parameters:
        net_weights = parameters_to_ndarrays(net_weights)
    net_json = json.dumps(net_weights, cls=NumpyEncoder)

    if type(objects_list[1]) is list:
        trees_json = []
        cids = []
        for tree, cid in objects_list[1]:
            trees_json.append(tree_to_json(tree, tmp_dir))
            cids.append(cid)
        tree_json = trees_json
        cid = cids
    else:
        tree_json = tree_to_json(objects_list[1][0], tmp_dir)
        cid = objects_list[1][1]

    parameters = ndarrays_to_parameters([net_json, tree_json, cid])

    return parameters


def parameters_to_objects(parameters: Parameters, tree_config_dict, tmp_dir="") -> List:
    # Begin data deserialization
    weights_binary = parameters.tensors[0]
    tree_binary = parameters.tensors[1]
    cid_binary = parameters.tensors[2]

    weights_json = bytes_to_ndarray(weights_binary)
    tree_json = bytes_to_ndarray(tree_binary)
    cid_data = bytes_to_ndarray(cid_binary)

    weights_json = json.loads(str(weights_json))
    weights_array = [np.asarray(layer_weights) for layer_weights in weights_json]
    weights_parameters = ndarrays_to_parameters(weights_array)

    client_tree_num = tree_config_dict["client_tree_num"]
    task_type = tree_config_dict["task_type"]

    if len(tree_json.shape) != 0:
        trees = []
        cids = []
        for tree_from_ensemble, cid in zip(tree_json, cid_data):
            cids.append(cid)
            trees.append(
                json_to_tree(tree_from_ensemble, client_tree_num, task_type, tmp_dir)
            )
        tree_parameters = [(tree, cid) for tree, cid in zip(trees, cids)]
    else:
        cid = int(cid_data.item())
        tree = json_to_tree(tree_json, client_tree_num, task_type, tmp_dir)
        tree_parameters = (tree, cid)

    return [weights_parameters, tree_parameters]


def tree_to_json(tree, tmp_directory=""):
    tmp_path = os.path.join(tmp_directory, str(uuid.uuid4()) + ".json")
    tree.get_booster().save_model(tmp_path)
    with open(tmp_path, "r") as fr:
        tree_params_obj = json.load(fr)
        tree_json = json.dumps(tree_params_obj)
    os.remove(tmp_path)

    return tree_json


def json_to_tree(tree_json, client_tree_num, task_type, tmp_directory=""):
    tree_json = json.loads(str(tree_json))
    tmp_path = os.path.join(tmp_directory, str(uuid.uuid4()) + ".json")
    with open(tmp_path, "w") as fw:
        json.dump(tree_json, fw)
    tree = get_tree(
        client_tree_num,
        task_type,
    )
    tree.load_model(tmp_path)
    os.remove(tmp_path)

    return tree

def train_test(data, client_tree_num):
    (X_train, y_train), (X_test, y_test) = data

    X_train.flags.writeable = True
    y_train.flags.writeable = True
    X_test.flags.writeable = True
    y_test.flags.writeable = True

    # If the feature dimensions of the trainset and testset do not agree,
    # specify n_features in the load_svmlight_file function in the above cell.
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
    # print("Feature dimension of the dataset:", X_train.shape[1])
    print("Size of the trainset:", X_train.shape[0])
    print("Size of the testset:", X_test.shape[0])
    assert X_train.shape[1] == X_test.shape[1]

    # Try to automatically determine the type of task
    n_classes = np.unique(y_train).shape[0]
    if n_classes == 2:
        task_type = "BINARY"
    elif n_classes > 2 and n_classes < 100:
        task_type = "MULTICLASS"
    else:
        task_type = "REG"

    if task_type == "BINARY":
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

    trainset = TreeDataset(np.array(X_train, copy=True), np.array(y_train, copy=True))
    testset = TreeDataset(np.array(X_test, copy=True), np.array(y_test, copy=True))

    # ## Conduct tabular dataset partition for Federated Learning

    # ## Define global variables for Federated XGBoost Learning

    # ## Build global XGBoost tree for comparison
    global_tree = construct_tree(X_train, y_train, client_tree_num, task_type)
    preds_train = global_tree.predict(X_train)
    preds_test = global_tree.predict(X_test)

    # metrics = calculate_metrics(y_train, preds_train, task_type)
    # print("Global XGBoost Training Metrics:", metrics)
    metrics = calculate_metrics(y_test, preds_test, task_type)
    return metrics
    # if task_type == "BINARY":
    #     result_train = accuracy_score(y_train, preds_train)
    #     result_test = accuracy_score(y_test, preds_test)
    #     print("Global XGBoost Training Accuracy: %f" % (result_train))
    #     print("Global XGBoost Testing Accuracy: %f" % (result_test))
    # elif task_type == "REG":
    #     result_train = mean_squared_error(y_train, preds_train)
    #     result_test = mean_squared_error(y_test, preds_test)
    #     print("Global XGBoost Training MSE: %f" % (result_train))
    #     print("Global XGBoost Testing MSE: %f" % (result_test))
