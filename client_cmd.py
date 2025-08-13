import sys
import os

import time
from pathlib import Path
import flwr as fl
import yaml
import argparse
import json
import logging
#import grpc

import flcore.datasets as datasets
from flcore.client_selector import get_model_client

# Start Flower client but after the server or error

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    # # parser.add_argument("--client_id", type=int, default="Client Id", help="Number of client")
    parser.add_argument("--dataset", type=str, default="dt4h_format", help="Dataloader to use")
    parser.add_argument("--metadata_file", type=str, default="metadata.json", help="Json file with metadata")
    parser.add_argument("--data_id", type=str, default="data_id.parquet" , help="Dataset ID")
    parser.add_argument("--normalization_method",type=str, default="IQR", help="Type of normalization: IQR STD MIN_MAX")
    parser.add_argument("--train_labels", type=str, nargs='+', default=None, help="Dataloader to use")
    parser.add_argument("--target_label", type=str, nargs='+', default=None, help="Dataloader to use")
    parser.add_argument("--train_size", type=float, default=0.8, help="Fraction of dataset to use for training. [0,1)")
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of federated iterations")
    parser.add_argument("--checkpoint_selection_metric", type=str, default="precision", help="Metric used for checkpoints")
    parser.add_argument("--dropout_method", type=str, default=None, help="Determines if dropout is used")
    parser.add_argument("--smooth_method", type=str, default=None, help="Weight smoothing")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--local_port", type=int, default=8081, help="Local port")
    parser.add_argument("--production_mode", type=str, default="True",  help="Production mode")
    parser.add_argument("--node_name", type=str, default="./", help="Node name for certificates")

    parser.add_argument("--experiment", type=json.loads, default={"name": "experiment_1", "log_path": "logs", "debug": "true"}, help="experiment logs")
    parser.add_argument("--smoothWeights", type=json.loads, default= {"smoothing_strenght": 0.5}, help="Smoothing parameters")
    parser.add_argument("--linear_models", type=json.loads, default={"n_features": 9}, help="Linear model parameters")
#    parser.add_argument("--n_features", type=int, default=0, help="Number of features")
    parser.add_argument("--random_forest", type=json.loads, default={"balanced_rf": "true"}, help="Random forest parameters")
    parser.add_argument("--weighted_random_forest", type=json.loads, default={"balanced_rf": "true", "levelOfDetail": "DecisionTree"}, help="Weighted random forest parameters")
    parser.add_argument("--xgb", type=json.loads, default={"batch_size": 32,"num_iterations": 100,"task_type": "BINARY","tree_num": 500}, help="XGB parameters")

# Variables hardcoded
    parser.add_argument("--sandbox_path", type=str, default="/sandbox", help="Sandbox path to use")
    parser.add_argument("--certs_path", type=str, default="/certs", help="Certificates path")
    parser.add_argument("--data_path", type=str, default="/data", help="Data path")

    args = parser.parse_args()

    config = vars(args)
    new = []
    for i in config["train_labels"]:
        parsed = i.replace("]", "").replace("[", "").replace(",", "")
        new.append(parsed)
    config["train_labels"] = new

    new = []
    for i in config["target_label"]:
        parsed = i.replace("]", "").replace("[", "").replace(",", "")
        new.append(parsed)
    config["target_labels"] = new

    if config["model"] in ("logistic_regression", "elastic_net", "lsvc"):
        config["linear_models"] = {}
        n_feats = len(config["train_labels"])
        config['linear_models']['n_features'] = n_feats # config["n_features"]
        config["held_out_center_id"] = -1

    # Create sandbox log file path
    sandbox_log_file = Path(os.path.join(config["sandbox_path"], "log_client.txt"))

    # Set up the file handler (writes to file)
    file_handler = logging.FileHandler(sandbox_log_file)
    file_handler.setLevel(logging.DEBUG)
    # Set up the console handler (writes to Docker logs via stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter for consistency
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Get the root logger and configure it
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear any default handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect print() and sys.stdout/sys.stderr into logger
    class StreamToLogger:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level

        def write(self, message):
            for line in message.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

        def flush(self):
            pass

    # Create two sub-loggers
    stdout_logger = logging.getLogger("STDOUT")
    stderr_logger = logging.getLogger("STDERR")

    # Redirect standard output and error to logging
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    # Now you can use logging in both places
    logging.debug("This will be logged to both the console and the file.")

    # Now you can use logging in both places
    logging.debug("This will be logged to both the console and the file.")

    model = config["model"]
    if config["production_mode"] == "True":
        node_name = os.getenv("NODE_NAME")
#        num_client = int(node_name.split("_")[-1])
        data_path = os.getenv("DATA_PATH")
        ca_cert = Path(os.path.join(config["certs_path"],"rootCA_cert.pem"))
        root_certificate = Path(f"{ca_cert}").read_bytes()
#        root_certificate = ca_cert
#        root_certificate =( Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes(),
#            Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes(),
#            Path(os.path.join(config["certs_path"],"rootCA_key.pem")).read_bytes() )

        root_cert = Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes()
        client_cert = Path(os.path.join(config["certs_path"],config["node_name"]+"_client_cert.pem")).read_bytes()
        client_key = Path(os.path.join(config["certs_path"],config["node_name"]+"_client_key.pem")).read_bytes()

        #ssl_credentials = grpc.ssl_channel_credentials(
        #    root_certificates=root_cert,  # Certificado raíz del servidor
        #    private_key=client_key,  # Clave privada del cliente
        #    certificate_chain=client_cert  # Certificado del cliente
        #)

        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        #channel = grpc.secure_channel(f"{central_ip}:{central_port}", ssl_credentials)

    else:
        data_path = config["data_path"]
        root_certificate = None
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
#        if len(sys.argv) == 1:
#            raise ValueError("Please provide the client id when running in simulation mode")
#        num_client = int(sys.argv[1])

num_client = 0 # config["client_id"]
(X_train, y_train), (X_test, y_test) = datasets.load_dataset(config, num_client)

data = (X_train, y_train), (X_test, y_test)
client = get_model_client(config, data, num_client)
"""
if isinstance(client, fl.client.NumPyClient):
    fl.client.start_numpy_client(
        server_address=f"{central_ip}:{central_port}",
#        credentials=ssl_credentials,
        root_certificates=root_certificate,
        client=client,
#        channel = channel,
    )
else:
    fl.client.start_client(
        server_address=f"{central_ip}:{central_port}",
#        credentials=ssl_credentials,
        root_certificates=root_certificate,
        client=client,
#        channel = channel,
    )
#fl.client.start_client(channel=channel, client=client)
"""
for attempt in range(3):
    try:
        if isinstance(client, fl.client.NumPyClient):
            fl.client.start_numpy_client(
                server_address=f"{central_ip}:{central_port}",
                #credentials=ssl_credentials,
                root_certificates=root_certificate,
                client=client,
                #channel=channel,
            )
        else:
            fl.client.start_client(
                server_address=f"{central_ip}:{central_port}",
                # credentials=ssl_credentials,
                root_certificates=root_certificate,
                client=client,
                #channel=channel,
            )
        break  # Si todo salió bien, salimos del bucle
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < 2:
            time.sleep(2)  # Espera un poco antes de reintentar
        else:
            print("All connection attempts failed.")
            raise
