import sys
import os
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

    parser.add_argument("--client_id", type=int, default="Client Id", help="Number of client")
    parser.add_argument("--dataset", type=str, default="dt4h_format", help="Dataloader to use")
    parser.add_argument("--metadata_file", type=str, default="metadata.json", help="Json file with metadata")
    parser.add_argument("--data_file", type=str, default="data.parquet" , help="parquet o csv file with actual data")
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
    parser.add_argument("--data_path", type=str, default=None, help="Data path")
    parser.add_argument("--production_mode", type=str, default="True",  help="Production mode")
    parser.add_argument("--certs_path", type=str, default="./", help="Certificates path")

    parser.add_argument("--sandbox_path", type=str, default="./", help="Sandbox path to use")
    parser.add_argument("--experiment", type=json.loads, default={"name": "experiment_1", "log_path": "logs", "debug": "true"}, help="experiment logs")
    parser.add_argument("--smoothWeights", type=json.loads, default= {"smoothing_strenght": 0.5}, help="Smoothing parameters")
    parser.add_argument("--linear_models", type=json.loads, default={"n_features": 9}, help="Linear model parameters")
    parser.add_argument("--random_forest", type=json.loads, default={"balanced_rf": "true"}, help="Random forest parameters")
    parser.add_argument("--weighted_random_forest", type=json.loads, default={"balanced_rf": "true", "levelOfDetail": "DecisionTree"}, help="Weighted random forest parameters")
    parser.add_argument("--xgb", type=json.loads, default={"batch_size": 32,"num_iterations": 100,"task_type": "BINARY","tree_num": 500}, help="XGB parameters")

    args = parser.parse_args()

    config = vars(args)
    sandbox_log_file = Path(os.path.join(config["sandbox_path"], "log_client.txt"))
    logging.basicConfig(level=logging.INFO, filename=sandbox_log_file)
    file_out = open(sandbox_log_file, "a")
    sys.stdout = file_out
    sys.stderr = file_out

    model = config["model"]
    if config["production_mode"] == "True":
        node_name = os.getenv("NODE_NAME")
#        num_client = int(node_name.split("_")[-1])
        num_client = config["client_id"]
        data_path = os.getenv("DATA_PATH")
        ca_cert = Path(os.path.join(config["certs_path"],"rootCA_cert.pem"))
        root_certificate = Path(f"{ca_cert}").read_bytes()
#        root_certificate = ca_cert
#        root_certificate =( Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes(),
#            Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes(),
#            Path(os.path.join(config["certs_path"],"rootCA_key.pem")).read_bytes() )

#        root_cert = Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes()
#        client_cert = Path(os.path.join(config["certs_path"],"rootCA_cert.pem")).read_bytes()
#        client_key = Path(os.path.join(config["certs_path"],"rootCA_key.pem")).read_bytes()

#        ssl_credentials = grpc.ssl_channel_credentials(
#            root_cert,  # Certificado raíz del servidor
#            client_key,  # Clave privada del cliente
#            client_cert  # Certificado del cliente
#        )

        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
#        channel = grpc.secure_channel(f"{central_ip}:{central_port}", ssl_credentials)

    else:
        data_path = config["data_path"]
        root_certificate = None
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        num_client = config["client_id"]
#        if len(sys.argv) == 1:
#            raise ValueError("Please provide the client id when running in simulation mode")
#        num_client = int(sys.argv[1])


    print("Client id:" + str(num_client))

(X_train, y_train), (X_test, y_test) = datasets.load_dataset(config, num_client)

data = (X_train, y_train), (X_test, y_test)
client = get_model_client(config, data, num_client)
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
file_out.close()