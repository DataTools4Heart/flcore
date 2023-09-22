import warnings
import os
import sys
from pathlib import Path

import flwr as fl
import yaml
import flcore.datasets as datasets
from flcore.server_selector import get_model_server_and_strategy

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"

    # Read the config file

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["production_mode"]:
        data_path = os.getenv("DATA_PATH")
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        certificates = certificates = (
            Path( '.cache/certificates/rootCA_cert.pem' ).read_bytes(),
            Path( '.cache/certificates/server_cert.pem' ).read_bytes(),
            Path( '.cache/certificates/server_key.pem'  ).read_bytes(),
        ),
    else:
        data_path = config["data_path"]
        central_ip = "LOCALHOST"
        central_port = "8080"
        certificates = None

    # Create experiment directory
    experiment_dir = Path("results") / config["experiment"]["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory for saving the model
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # History directory for saving the history
    history_dir = experiment_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    # # Create an instance of the model and get the parameters
    # model = LogisticRegression()
    # utils.set_initial_params( model )

    # # Pass parameters to the Strategy for server-side parameter initialization
    # strategy = fl.server.strategy.FedAvg(
    #     min_available_clients = 2,
    #     evaluate_fn           = get_evaluate_fn( model ),
    #     on_fit_config_fn      = fit_round,
    # )
    # (X_train, y_train), (X_test, y_test) = datasets.load_cvd(DATA_PATH, 'All')
    # (X_train, y_train), (X_test, y_test) = datasets.load_mnist()
    (X_train, y_train), (X_test, y_test) = datasets.load_dataset(config)

    data = (X_train, y_train), (X_test, y_test)

    server, strategy = get_model_server_and_strategy(config, data)

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address=f"{central_ip}:{central_port}",
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"]),
        server=server,
        strategy=strategy,
        certificates = certificates,
    )
    # # Save the model and the history
    # filename = os.path.join( checkpoint_dir, 'final_model.pt' )
    # joblib.dump(model, filename)
    # Save the history as a yaml file
    print(history)
    with open(history_dir / "history.yaml", "w") as f:
        yaml.dump(history, f)
