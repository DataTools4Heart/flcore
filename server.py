import warnings
from pathlib import Path

import flwr as fl
import yaml

import flcore.datasets as datasets
from flcore.server_selector import get_model_server_and_strategy

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Read the config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

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
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"]),
        server=server,
        strategy=strategy,
        # certificates = (
        #     Path( '.cache/certificates/rootCA_cert.pem' ).read_bytes(),
        #     Path( '.cache/certificates/server_cert.pem' ).read_bytes(),
        #     Path( '.cache/certificates/server_key.pem'  ).read_bytes(),
        # ),
    )
    # # Save the model and the history
    # filename = os.path.join( checkpoint_dir, 'final_model.pt' )
    # joblib.dump(model, filename)
    # Save the history as a yaml file
    print(history)
    with open(history_dir / "history.yaml", "w") as f:
        yaml.dump(history, f)
