import logging
import yaml

import pytest

# Set the logging level depending on the level of detail you would like to have in the logs while running the tests.
LOGGING_LEVEL = logging.INFO  # WARNING  # logging.INFO

# class TestMediganExecutorMethods(unittest.TestCase):
class TestFLCoreModels:
    def setup_class(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.test_output_path = "test_output_path"
        self.num_clients = 3

    @pytest.mark.parametrize(
        "model_name",
        ["logistic_regression", 
        "elastic_net",
        "lsvc",
        "random_forest",
        "xgb"
        ],
    )
    def test_get_model_client(
        self, model_name
    ):
        self.config["model_name"] = model_name
        
        from flcore.client_selector import get_model_client
        from flcore.datasets import load_dataset
        data = load_dataset(self.config, 2)

        client = get_model_client(self.config, data, 2)

        assert client is not None
