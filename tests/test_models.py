import logging
import yaml
import subprocess
import time

import pytest

# Set the logging level depending on the level of detail you would like to have in the logs while running the tests.
LOGGING_LEVEL = logging.INFO  # WARNING  # logging.INFO

model_names = [
    "logistic_regression", 
    "elastic_net",
    "lsvc",
    "random_forest",
    "weighted_random_forest",
    "xgb"
    ]

class TestFLCoreModels:
    def setup_class(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.num_clients = 3


    @pytest.mark.parametrize(
        "model_name",
        model_names
    )
    def test_get_model_client(
        self, model_name
    ):
        self.config["model"] = model_name
        
        from flcore.client_selector import get_model_client
        from flcore.datasets import load_dataset
        data = load_dataset(self.config, 2)

        client = get_model_client(self.config, data, 2)

        assert client is not None


    @pytest.mark.parametrize(
        "model_name",
        model_names
    )
    def test_run(self, model_name):

        self.config["model"] = model_name
        
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            config = self.config

        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
    
        print("Starting server")
        server_process = subprocess.Popen("python server.py", shell=True)
        time.sleep(20)

        client_processes = []
        for i in range(1, config["num_clients"] + 1):
            print("Starting client " + str(i))
            client_processes.append(
                subprocess.Popen("python client.py " + str(i), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            )

        for client_process in client_processes:
            client_process.communicate()
            assert client_process.returncode == 0

        server_process.communicate()
        assert server_process.returncode == 0
