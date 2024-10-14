## Create Flower custom client

from typing import List, Tuple, Union
import time
import flwr as fl
import numpy as np
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import Parameters
from torch.utils.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor

from flcore.models.xgb.cnn import CNN, test, train
from flcore.models.xgb.utils import (
    NumpyEncoder,
    TreeDataset,
    construct_tree_from_loader,
    get_dataloader,
    parameters_to_objects,
    serialize_objects_to_parameters,
    tree_encoding_loader,
    train_test
)


class FL_Client(fl.client.Client):
    def __init__(
        self,
        task_type: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        client_tree_num: int,
        client_num: int,
        cid: str,
        log_progress: bool = False,
    ):
        """
        Creates a client for training `network.Net` on tabular dataset.
        """
        self.task_type = task_type
        self.cid = cid
        self.tree = construct_tree_from_loader(trainloader, client_tree_num, task_type)
        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.trainloader = None
        self.valloader = None
        self.client_tree_num = client_tree_num
        self.client_num = client_num
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress
        self.tree_config_dict = {
            "client_tree_num": self.client_tree_num,
            "task_type": self.task_type,
        }
        self.tmp_dir = ""

        # instantiate model
        self.net = CNN(client_num=client_num, client_tree_num=client_tree_num)

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.round_time = -1

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(
        self, ins: GetParametersIns
    ) -> Tuple[
        GetParametersRes, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]
    ]:
        net_params = self.net.get_weights()
        parameters = serialize_objects_to_parameters(
            [net_params, (self.tree, self.cid)], self.tmp_dir
        )

        return GetParametersRes(
            status=Status(Code.OK, ""),
            parameters=parameters,
        )

    def set_parameters(
        self,
        parameters: Tuple[
            Parameters,
            Union[
                Tuple[XGBClassifier, int],
                Tuple[XGBRegressor, int],
                List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
            ],
        ],
    ) -> Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ]:
        self.net.set_weights(parameters_to_ndarrays(parameters[0]))
        return parameters[1]

    def fit(self, fit_params: FitIns) -> FitRes:
        # Process incoming request to train
        num_iterations = fit_params.config["num_iterations"]
        batch_size = fit_params.config["batch_size"]

        objects = parameters_to_objects(
            fit_params.parameters, self.tree_config_dict, self.tmp_dir
        )

        aggregated_trees = self.set_parameters(objects)

        if type(aggregated_trees) is list:
            print("Client " + self.cid + ": recieved", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")
        self.trainloader = tree_encoding_loader(
            self.trainloader_original,
            batch_size,
            aggregated_trees,
            self.client_tree_num,
            self.client_num,
        )
        self.valloader = tree_encoding_loader(
            self.valloader_original,
            batch_size,
            aggregated_trees,
            self.client_tree_num,
            self.client_num,
        )

        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(self.trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        start_time = time.time()
        self.net.to(self.device)
        train_loss, train_result, num_examples = train(
            self.task_type,
            self.net,
            self.trainloader,
            device=self.device,
            num_iterations=num_iterations,
            log_progress=self.log_progress,
        )
        print(
            f"Client {self.cid}: training round complete, {num_examples} examples processed"
        )
        
        self.round_time = (time.time() - start_time)

        # Return training information: model, number of examples processed and metrics
        if self.task_type == "BINARY":
            return FitRes(
                status=Status(Code.OK, ""),
                # parameters=self.get_parameters(fit_params.config),
                parameters=self.get_parameters(fit_params.config).parameters,
                num_examples=num_examples,
                metrics={"loss": train_loss, "accuracy": train_result, "running_time":self.round_time},
            )
        elif self.task_type == "REG":
            return FitRes(
                status=Status(Code.OK, ""),
                parameters=self.get_parameters(fit_params.config),
                num_examples=num_examples,
                metrics={"loss": train_loss, "mse": train_result, "running_time":self.round_time},
            )

    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:

        print(
            f"Client {self.cid}: Start evaluation round"
        )
        # Process incoming request to evaluate
        objects = parameters_to_objects(
            eval_params.parameters, self.tree_config_dict, self.tmp_dir
        )
        self.set_parameters(objects)

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = test(
            self.task_type,
            self.net,
            self.valloader,
            device=self.device,
            log_progress=self.log_progress,
        )

        metrics = result
        metrics["client_id"] = int(self.cid)
        metrics["round_time [s]"] = self.round_time

        # Return evaluation information
        if self.task_type == "BINARY":
            accuracy = metrics["accuracy"]
            print(
                f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, accuracy={accuracy:.4f}"
            )
            return EvaluateRes(
                status=Status(Code.OK, ""),
                loss=loss,
                num_examples=num_examples,
                # metrics={"accuracy": result},
                metrics=metrics,
            )
        elif self.task_type == "REG":
            print(
                f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, mse={result:.4f}"
            )
            return EvaluateRes(
                status=Status(Code.OK, ""),
                loss=loss,
                num_examples=num_examples,
                metrics=metrics,
            )


def get_client(config, data, client_id) -> fl.client.Client:
    (X_train, y_train), (X_test, y_test) = data
    task_type = config["xgb"]["task_type"]
    client_num = config["num_clients"]
    client_tree_num = config["xgb"]["tree_num"] // client_num
    batch_size = "whole"
    cid = str(client_id)
    trainset = TreeDataset(np.array(X_train, copy=True), np.array(y_train, copy=True))
    valset = TreeDataset(np.array(X_test, copy=True), np.array(y_test, copy=True))
    trainloader = get_dataloader(trainset, "train", batch_size)
    valloader = get_dataloader(valset, "test", batch_size)

    metrics = train_test(data, client_tree_num)
    from flcore import datasets
    if client_id == 1:
        cross_id = 2
    else:
        cross_id = 1
    _, (X_test, y_test) = datasets.load_dataset(config, cross_id)

    data = (X_train, y_train), (X_test, y_test)
    metrics_cross = train_test(data, client_tree_num)
    print("Client " + cid + " non-federated training results:")
    print(metrics)
    print("Cross testing model on client " + str(cross_id) + ":")
    print(metrics_cross)

    client = FL_Client(
        task_type,
        trainloader,
        valloader,
        client_tree_num,
        client_num,
        cid,
        log_progress=False,
    )
    return client
