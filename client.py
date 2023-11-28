# client.py

import sys
import os
from pathlib import Path
import flwr as fl
import yaml

import flcore.datasets as datasets
from flcore.client_selector import get_model_client
from flcore.smpc_module import SMPClient, SMPClientEvaluator


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage is wrong. Usage: python main.py <number of client>")
        sys.exit
    elif len(sys.argv) == 3:
        config_path = sys.argv[2]
    else:
        config_path = "config.yaml"

    num_client = int(sys.argv[1])

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = config["model"]

    if config["production_mode"]:
        data_path = os.getenv("DATA_PATH")
        flower_ssl_cacert = os.getenv("FLOWER_SSL_CACERT")
        root_certificate = Path(f"{flower_ssl_cacert}").read_bytes()
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
    else:
        data_path = config["data_path"]
        root_certificate = None
        central_ip = "LOCALHOST"
        central_port = "8000"

    print("Client id:" + str(num_client))

    (X_train, y_train), (X_test, y_test) = datasets.load_dataset(config, num_client)

    data = (X_train, y_train), (X_test, y_test)

    client = get_model_client(config, data, num_client)

    # Check if smpc_url is provided in the config, otherwise set it to a default value
    smpc_base_url = config.get("smpc", {}).get("smpc_url", "http://default-smpc-url.com")

    if config.get("use_smpc", True):
        smpc_client = SMPClient(client.model, smpc_base_url)
        smpc_evaluator = SMPClientEvaluator(client.model)
 
    if isinstance(client, fl.client.NumPyClient):
        fl.client.start_numpy_client(
            server_address=f"{central_ip}:{central_port}",
            root_certificates=root_certificate,
            client=client,
        )
    else:
        fl.client.start_client(
            server_address=f"{central_ip}:{central_port}",
            root_certificates=root_certificate,
            client=client,
        )
