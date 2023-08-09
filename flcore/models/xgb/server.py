# ## Create Flower custom server

import functools
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Scalar,
    Status,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns, Parameters
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import evaluate_clients, fit_clients
from flwr.server.strategy import FedXgbNnAvg, Strategy
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import DataLoader
from xgboost import XGBClassifier, XGBRegressor

import flcore.datasets as datasets
from flcore.models.xgb.client import FL_Client
from flcore.models.xgb.cnn import CNN, test
from flcore.models.xgb.utils import (
    TreeDataset,
    construct_tree,
    do_fl_partitioning,
    parameters_to_objects,
    serialize_objects_to_parameters,
    tree_encoding_loader,
)

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]


class FL_Server(fl.server.Server):
    """Flower server."""

    def __init__(
        self, *, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy
        self.max_workers: Optional[int] = None
        self.tree_config_dict = {
            "client_tree_num": self.strategy.evaluate_fn.keywords["client_tree_num"],
            "task_type": self.strategy.evaluate_fn.keywords["task_type"],
        }

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)

        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[
            Optional[
                Tuple[
                    Parameters,
                    Union[
                        Tuple[XGBClassifier, int],
                        Tuple[XGBRegressor, int],
                        List[
                            Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]
                        ],
                    ],
                ]
            ],
            Dict[str, Scalar],
            FitResultsAndFailures,
        ]
    ]:
        """Perform a single round of federated averaging."""
        parameters_packed = serialize_objects_to_parameters(self.parameters)
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            # parameters=self.parameters,
            parameters=parameters_packed,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        for result in results:
            result[1].parameters = self.serialized_to_parameters(result[1])

        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        NN_aggregated: Parameters
        trees_aggregated: Union[
            Tuple[XGBClassifier, int],
            Tuple[XGBRegressor, int],
            List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
        ]
        metrics_aggregated: Dict[str, Scalar]
        aggregated, metrics_aggregated = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        NN_aggregated, trees_aggregated = aggregated[0], aggregated[1]

        if type(trees_aggregated) is list:
            print("Server side aggregated", len(trees_aggregated), "trees.")
        else:
            print("Server side did not aggregate trees.")

        return (
            [NN_aggregated, trees_aggregated],
            metrics_aggregated,
            (results, failures),
        )

    # def list_to_packed_parameters(self, parameters: List):
    #     net_weights = parameters_to_ndarrays(parameters[0])
    #     tree_json = parameters[1][0]
    #     cid = parameters[1][1]

    #     return ndarrays_to_parameters([net_weights, tree_json, cid])

    def serialized_to_parameters(self, get_parameters_res_tree):
        objects = parameters_to_objects(
            get_parameters_res_tree.parameters, self.tree_config_dict
        )

        weights_parameters = objects[0]
        tree_parameters = objects[1]

        return [
            GetParametersRes(
                status=Status(Code.OK, ""),
                parameters=weights_parameters,
            ),
            tree_parameters,
        ]

    def _get_initial_parameters(
        self, timeout: Optional[float]
    ) -> Tuple[Parameters, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]]:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res_tree = random_client.get_parameters(ins=ins, timeout=timeout)

        get_parameters_res_tree = self.serialized_to_parameters(get_parameters_res_tree)

        parameters = [get_parameters_res_tree[0].parameters, get_parameters_res_tree[1]]

        log(INFO, "Received initial parameters from one random client")

        return parameters


# ## Create server-side evaluation and experiment


def serverside_eval(
    server_round: int,
    parameters: Tuple[
        Parameters,
        Union[
            Tuple[XGBClassifier, int],
            Tuple[XGBRegressor, int],
            List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
        ],
    ],
    config: Dict[str, Scalar],
    task_type: str,
    testloader: DataLoader,
    batch_size: int,
    client_tree_num: int,
    client_num: int,
) -> Tuple[float, Dict[str, float]]:
    """An evaluation function for centralized/serverside evaluation over the entire test set."""
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = CNN(client_num=client_num, client_tree_num=client_tree_num)
    # print_model_layers(model)

    model.set_weights(parameters_to_ndarrays(parameters[0]))
    model.to(device)

    trees_aggregated = parameters[1]

    testloader = tree_encoding_loader(
        testloader, batch_size, trees_aggregated, client_tree_num, client_num
    )
    loss, result, _ = test(
        task_type, model, testloader, device=device, log_progress=False
    )

    if task_type == "BINARY":
        print(
            f"Evaluation on the server: test_loss={loss:.4f}, test_accuracy={result:.4f}"
        )
        return loss, {"accuracy": result}
    elif task_type == "REG":
        print(f"Evaluation on the server: test_loss={loss:.4f}, test_mse={result:.4f}")
        return loss, {"mse": result}


def get_server_and_strategy(
    config, data
) -> Tuple[Optional[fl.server.Server], Strategy]:
    # task_type = config['xgb'][ 'task_type' ]
    # The number of clients participated in the federated learning
    client_num = config["num_clients"]
    # The number of XGBoost trees in the tree ensemble that will be built for each client
    client_tree_num = config["xgb"]["tree_num"] // client_num

    num_rounds = config["num_rounds"]
    client_pool_size = client_num
    num_iterations = config["xgb"]["num_iterations"]
    fraction_fit = 1.0
    min_fit_clients = client_num

    batch_size = config["xgb"]["batch_size"]
    val_ratio = 0.1

    # DATASET = "CVD"
    # # DATASET = "MNIST"
    # # DATASET = "LIBSVM"

    # # Define the type of training task. Binary classification: BINARY; Regression: REG
    # task_types = ["BINARY", "REG"]
    # task_type = task_types[0]

    # PARTITION_DATA = False

    # if DATASET == 'LIBSVM':
    #         (X_train, y_train), (X_test, y_test) = datasets.load_libsvm(task_type)

    # elif DATASET == 'CVD':
    #     (X_train, y_train), (X_test, y_test) = datasets.load_cvd('dataset', 1)

    # elif DATASET == 'MNIST':
    #     (X_train, y_train), (X_test, y_test) = datasets.load_mnist()

    # else:
    #     raise ValueError('Dataset not supported')

    (X_train, y_train), (X_test, y_test) = data

    X_train.flags.writeable = True
    y_train.flags.writeable = True
    X_test.flags.writeable = True
    y_test.flags.writeable = True

    # If the feature dimensions of the trainset and testset do not agree,
    # specify n_features in the load_svmlight_file function in the above cell.
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
    print("Feature dimension of the dataset:", X_train.shape[1])
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

    if task_type == "BINARY":
        result_train = accuracy_score(y_train, preds_train)
        result_test = accuracy_score(y_test, preds_test)
        print("Global XGBoost Training Accuracy: %f" % (result_train))
        print("Global XGBoost Testing Accuracy: %f" % (result_test))
    elif task_type == "REG":
        result_train = mean_squared_error(y_train, preds_train)
        result_test = mean_squared_error(y_test, preds_test)
        print("Global XGBoost Training MSE: %f" % (result_train))
        print("Global XGBoost Testing MSE: %f" % (result_test))

    print(global_tree)

    # ## Simulate local XGBoost trees on clients for comparison

    client_trees_comparison = []

    # if PARTITION_DATA:
    trainloaders, _, testloader = do_fl_partitioning(
        trainset, testset, pool_size=client_num, batch_size="whole", val_ratio=0.0
    )

    # def start_experiment(
    #     task_type: str,
    #     trainset: Dataset,
    #     testset: Dataset,
    #     num_rounds: int = 5,
    #     client_tree_num: int = 50,
    #     client_pool_size: int = 5,
    #     num_iterations: int = 100,
    #     fraction_fit: float = 1.0,
    #     min_fit_clients: int = 2,
    #     batch_size: int = 32,
    #     val_ratio: float = 0.1,
    # ) -> History:
    #     client_resources = {"num_cpus": 0.5}  # 2 clients per CPU

    # Partition the dataset into subsets reserved for each client.
    # - 'val_ratio' controls the proportion of the (local) client reserved as a local test set
    # (good for testing how the final model performs on the client's local unseen data)
    trainloaders, valloaders, testloader = do_fl_partitioning(
        trainset,
        testset,
        batch_size="whole",
        pool_size=client_pool_size,
        val_ratio=val_ratio,
    )
    print(
        f"Data partitioned across {client_pool_size} clients"
        f" and {val_ratio} of local dataset reserved for validation."
    )

    # Configure the strategy
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        print(f"Configuring round {server_round}")
        return {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }

    # FedXgbNnAvg
    strategy = FedXgbNnAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit if val_ratio > 0.0 else 0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=client_pool_size,  # all clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=(lambda r: {"batch_size": batch_size}),
        evaluate_fn=functools.partial(
            serverside_eval,
            task_type=task_type,
            testloader=testloader,
            batch_size=batch_size,
            client_tree_num=client_tree_num,
            client_num=client_num,
        ),
        accept_failures=False,
    )

    print(
        f"FL experiment configured for {num_rounds} rounds with {client_pool_size} client in the pool."
    )
    print(
        f"FL round will proceed with {fraction_fit * 100}% of clients sampled, at least {min_fit_clients}."
    )

    def client_fn(cid: str) -> fl.client.Client:
        """Creates a federated learning client"""
        if val_ratio > 0.0 and val_ratio <= 1.0:
            return FL_Client(
                task_type,
                trainloaders[int(cid)],
                valloaders[int(cid)],
                client_tree_num,
                client_pool_size,
                cid,
                log_progress=False,
            )
        else:
            return FL_Client(
                task_type,
                trainloaders[int(cid)],
                None,
                client_tree_num,
                client_pool_size,
                cid,
                log_progress=False,
            )

    server = FL_Server(client_manager=SimpleClientManager(), strategy=strategy)

    # history = fl.server.start_server(
    #     server_address = "[::]:8080",
    #     server=server,
    #     config = fl.server.ServerConfig(num_rounds=20),
    #     strategy = strategy
    # )
    # Start the simulation
    # history = fl.simulation.start_simulation(
    #     client_fn=client_fn,
    #     server=FL_Server(client_manager=SimpleClientManager(), strategy=strategy),
    #     num_clients=client_pool_size,
    #     client_resources=client_resources,
    #     config=ServerConfig(num_rounds=num_rounds),
    #     strategy=strategy,
    # )
    # print(history)
    # return history

    return server, strategy
