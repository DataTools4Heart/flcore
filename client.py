import sys

import flwr as fl
import yaml

import flcore.datasets as datasets
from flcore.client_selector import get_model_client

# flower_ssl_cacert = os.getenv("FLOWER_SSL_CACERT")
central_ip = "LOCALHOST"
central_port = "8080"

# Start Flower client but after the server or error

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
model = config["model"]

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage is wrong. Usage: python main.py <number of client>")
        sys.exit

    num_client = int(sys.argv[1])

    print("Client id:" + str(num_client))

(X_train, y_train), (X_test, y_test) = datasets.load_dataset(config, num_client)

data = (X_train, y_train), (X_test, y_test)

client = get_model_client(config, data, num_client)

if isinstance(client, fl.client.NumPyClient):
    fl.client.start_numpy_client(
        server_address=f"{central_ip}:{central_port}",
        client=client,
    )
else:
    fl.client.start_client(
        server_address=f"{central_ip}:{central_port}",
        client=client,
    )
