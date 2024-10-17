import sys
import os
from pathlib import Path
import flwr as fl
import yaml

import flcore.datasets as datasets
from flcore.client_selector import get_model_client

# Start Flower client but after the server or error

if __name__ == "__main__":
    if len(sys.argv) == 3:
        config_path = sys.argv[2]
    else:
        config_path = "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = config["model"]

    if config["production_mode"]:
        node_name = os.getenv("NODE_NAME")
        num_client = int(node_name.split("_")[-1])
        data_path = os.getenv("DATA_PATH")
        flower_ssl_cacert = os.getenv("FLOWER_SSL_CACERT")
        root_certificate = Path(f"{flower_ssl_cacert}").read_bytes()
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        
    else:
        data_path = config["data_path"]
        root_certificate = None
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        if len(sys.argv) == 1:
            raise ValueError("Please provide the client id when running in simulation mode")
        num_client = int(sys.argv[1])


    print("Client id:" + str(num_client))

(X_train, y_train), (X_test, y_test) = datasets.load_dataset(config, num_client)

data = (X_train, y_train), (X_test, y_test)

client = get_model_client(config, data, num_client)

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
