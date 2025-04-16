import os
import sys
import json
import logging
import warnings
import argparse
from pathlib import Path

import yaml
import numpy
import flwr as fl

import flcore.datasets as datasets
from flcore.compile_results import compile_results
from flwr.common import SecureGRPCBridge, SuperLink
from flcore.server_selector import get_model_server_and_strategy

warnings.filterwarnings("ignore")

def check_config(config):
    assert isinstance(config['num_clients'], int), 'num_clients should be an int'
    assert isinstance(config['num_rounds'], int), 'num_rounds should be an int'
    if(config['smooth_method'] != 'None'):
        assert config['smoothWeights']['smoothing_strenght'] >= 0 and config['smoothWeights']['smoothing_strenght'] <= 1, 'smoothing_strenght should be betwen 0 and 1'
    if(config['dropout_method'] != 'None'):
        assert config['dropout']['percentage_drop'] >= 0 and config['dropout']['percentage_drop'] < 100, 'percentage_drop should be betwen 0 and 100'

    assert (config['smooth_method']== 'EqualVoting' or \
        config['smooth_method']== 'SlowerQuartile' or \
        config['smooth_method']== 'SsupperQuartile' or \
        config['smooth_method']== 'None'), 'the smooth methods are not correct: EqualVoting, SlowerQuartile and SsupperQuartile'

    if(config['model'] == 'weighted_random_forest'):
         assert (config['weighted_random_forest']['levelOfDetail']== 'DecisionTree' or \
            config['weighted_random_forest']['levelOfDetail']== 'RandomForest'), 'the levels of detail for weighted RF are not correct: DecisionTree and RandomForest '


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads parameters from command line.")

    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of federated iterations")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--dataset", type=str, default="dt4h_format", help="Dataloader to use")
    parser.add_argument("--sandbox_path", type=str, default="./", help="Sandbox path to use")
    parser.add_argument("--certs_path", type=str, default="./", help="Certificates path")

    parser.add_argument("--smooth_method", type=str, default="EqualVoting", help="Weight smoothing")
    parser.add_argument("--smoothWeights", type=json.loads, default= {"smoothing_strenght": 0.5}, help="Smoothing parameters")
    parser.add_argument("--dropout_method", type=str, default=None, help="Determines if dropout is used")
    parser.add_argument("--dropout", type=json.loads, default={"percentage_drop":0}, help="Dropout parameters")
    parser.add_argument("--weighted_random_forest", type=json.loads, default={"balanced_rf": "true", "levelOfDetail": "DecisionTree"}, help="Weighted random forest parameters")
    parser.add_argument("--checkpoint_selection_metric", type=str, default="precision", help="Metric used for checkpoints")
    parser.add_argument("--production_mode", type=str, default="True",  help="Production mode")

    parser.add_argument("--data_path", type=str, default=None, help="Data path")
    parser.add_argument("--local_port", type=int, default=8081, help="Local port")
    parser.add_argument("--experiment", type=json.loads, default={"name": "experiment_1", "log_path": "logs", "debug": "true"}, help="experiment logs")
    parser.add_argument("--random_forest", type=json.loads, default={"balanced_rf": "true"}, help="Random forest parameters")

    args = parser.parse_args()

    config = vars(args)

    experiment_dir = Path(os.path.join(config["experiment"]["log_path"], config["experiment"]["name"]))
    config["experiment_dir"] = experiment_dir

    sandbox_log_file = Path(os.path.join(config["sandbox_path"], "log_server.txt"))

    file_handler = logging.FileHandler(sandbox_log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

    logging.debug("This will be logged to both the console and the file.")

    logging.debug("Starting Flower server...")

    check_config(config)
    if config["production_mode"] == "True":
        data_path = os.getenv("DATA_PATH")
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")

        root_certificate = Path(os.path.join(config["certs_path"],"rootCA_cert.pem"))
        server_cert =  Path(os.path.join(config["certs_path"],"server_cert.pem"))
        server_key =  Path(os.path.join(config["certs_path"],"server_key.pem"))

        bridge = SecureGRPCBridge(
            server_address=f"{central_ip}:{central_port}",
            root_certificates=root_certificate,
            private_key=server_key,
            certificate_chain=server_cert,
        )
        superlink = SuperLink(bridge)

    else:
        data_path = config["data_path"]
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        certificates = None

    experiment_dir = Path(os.path.join(config["experiment"]["log_path"], config["experiment"]["name"]))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config["experiment_dir"] = experiment_dir

    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open("config.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    os.system(f"cp config.yaml {experiment_dir}")

    # **************** This part to be removed since data should not be here
    #(X_train, y_train), (X_test, y_test) = datasets.load_dataset(config)
    (X_train, y_train), (X_test, y_test) = ([0],[0]), ([0],[0])
    # valid since only xgb requieres the data and will not be used
    data = (X_train, y_train), (X_test, y_test)

    # ***********************************************************************
    server, strategy = get_model_server_and_strategy(config, data)

    history = fl.server.run_server(
        server=server,
        strategy=strategy,
        server_config=fl.server.ServerConfig(num_rounds=config["num_rounds"], round_timeout=None),
        transport=superlink,
    )
    print(history)