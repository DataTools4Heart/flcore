import sys
import os
import glob

import time
from pathlib import Path
import flwr as fl
import yaml
import argparse
import json
import logging
#import grpc

import flcore.datasets as datasets
from flcore.utils import StreamToLogger, GetModelClient, SanityCheck

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    # Variables node settings
    parser.add_argument("--node_name", type=str, default="./", help="Node name for certificates")
    parser.add_argument("--local_port", type=int, default=8081, help="Local port")
    parser.add_argument("--sandbox_path", type=str, default="/sandbox", help="Sandbox path to use")
    parser.add_argument("--certs_path", type=str, default="/certs", help="Certificates path")
    parser.add_argument("--data_path", type=str, default="/data", help="Data path")
    parser.add_argument("--production_mode", type=str, default="True",  help="Production mode") # ¿Should exist?
    # Variables dataset related
    parser.add_argument("--dataset", type=str, default="dt4h_format", help="Dataloader to use")
    parser.add_argument("--data_id", type=str, default="data_id.parquet" , help="Dataset ID")
    parser.add_argument("--normalization_method",type=str, default="IQR", help="Type of normalization: IQR STD MIN_MAX")
    parser.add_argument("--train_labels", type=str, nargs='+', default=None, help="Dataloader to use")
    parser.add_argument("--target_label", type=str, nargs='+', default=None, help="Dataloader to use")
    parser.add_argument("--train_size", type=float, default=0.8, help="Fraction of dataset to use for training. [0,1)")
    # Variables training related
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of federated iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate when needed")    
    parser.add_argument("--checkpoint_selection_metric", type=str, default="precision", help="Metric used for checkpoints")
    # Esos dropouts puede que sean para el server y yno deberían estar aquí
    parser.add_argument("--dropout_method", type=str, default=None, help="Determines if dropout is used")
    parser.add_argument("--dropout_percentage", type=str, default=None, help="Ratio of dropout nodes")
    parser.add_argument("--smooth_method", type=str, default=None, help="Smoothing method")
    parser.add_argument("--smoothing_strenght", type=str, default=None, help="Smoothing strenght")
    # _____________________________________________________________________________________
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--experiment", type=json.loads, default={"name": "experiment_1", "log_path": "logs", "debug": "true"}, help="experiment logs")
    parser.add_argument("--smoothWeights", type=json.loads, default= {"smoothing_strenght": 0.5}, help="Smoothing parameters")
    # ________________________________________________________________________________
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients") # shouldnt exist here
    # ________________________________________________________________________________

    # General variables model related
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--n_feats", type=int, default=0, help="Number of input features")
    parser.add_argument("--n_out", type=int, default=0, help="Number of output features")
    parser.add_argument("--task", type=int, default=0, help="Task to perform (classification, regression)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for training, CPU, GPU")
    parser.add_argument("--local_epochs", type=int, default=10, help="Number of local epochs to train in each round")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size to train")

    # Specific variables model related
    # # Linear models
    parser.add_argument("--penalty", type=str, default="none", help="Penalties: none, l1, l2, elasticnet")
    parser.add_argument("--solver", type=str, default="saga", help="Numerical solver of optimization method")
    parser.add_argument("--l1_ratio", type=str, default=0.5, help="L1-L2 Ratio, necessary for ElasticNet, 0 -> L1 ; 1 -> L2")
    parser.add_argument("--max_iter", type=int, default=100000, help="Max iterations of optimizer")
    # # Random forest
    parser.add_argument("--balanced", type=str, default="True", help="Balanced Random Forest: True or False")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--max_depth", type=int, default=2, help="Max depth")
    parser.add_argument("--class_weight", type=str, default="balanced", help="Class weight")
    parser.add_argument("--levelOfDetail", type=str, default="DecisionTree", help="Level of detail")
    parser.add_argument("--regression_criterion", type=str, default="squared_error", help="Criterion for training")
    # # Neural networks
    # params : type: "nn", "BNN" Bayesiana, otros
    parser.add_argument("--neural_network", type=json.loads, default={"dropout_p": 0.2, "device": "cpu","local_epochs":10}, help="Neural Network parameters")
    parser.add_argument("--dropout_p", type=int, default=0.2, help="Montecarlo dropout rate")
    parser.add_argument("--T", type=int, default=20, help="Samples of MC dropout")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    # # XGB
    parser.add_argument("--xgb", type=json.loads, default={"batch_size": 32,"num_iterations": 100,"task_type": "BINARY","tree_num": 500}, help="XGB parameters")
    parser.add_argument("--tree_num", type=int, default=100, help="Number of trees")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
# *******************************************************************************************************************

    args = parser.parse_args()

    config = vars(args)

    est = config["data_id"]
    id = est.split("/")[-1]
#    dir_name = os.path.dirname(config["data_id"])
    dir_name_parent = str(Path(config["data_id"]).parent)

#    config["metadata_file"] = os.path.join(dir_name_parent,"metadata.json")
    config["metadata_file"] = os.path.join(est,"metadata.json")

    pattern = "*.parquet"
    parquet_files = glob.glob(os.path.join(est, pattern))
    # Saniy check, empty list
    if len(parquet_files) == 0:
        print("No parquet files found in ",est)
        sys.exit()

    # ¿How to choose one of the list?
    config["data_file"] = parquet_files[-1]

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

    SanityCheck()
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
client = GetModelClient(config, data)
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

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
