import logging
import yaml
import subprocess
import os
import signal
from threading import Timer
import time

import pytest

# Set the logging level depending on the level of detail you would like to have in the logs while running the tests.
LOGGING_LEVEL = logging.INFO  # WARNING  # logging.INFO

model_names = [
   "logistic_regression",
   "elastic_net",
   "lsvc",
    "random_forest",
    "balanced_random_forest",
    # # "weighted_random_forest",
    "xgb"
    ]

datasets = [
    "kaggle_hf",
    "diabetes",
    ]

def free_port(port):
    process = subprocess.Popen(["lsof", "-i", ":{0}".format(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    for process in str(stdout.decode("utf-8")).split("\n")[1:]:       
        data = [x for x in process.split(" ") if x != '']
        if (len(data) <= 1):
            continue
        os.kill(int(data[1]), signal.SIGKILL)

class TestFLCoreModels:
    def setup_class(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.config["num_clients"] = 3
        self.config["num_rounds"] = 2

        # To speed up tests, reduce number of trees in xgboost and random forest
        self.config["random_forest"]["tree_num"] = 5
        self.config["xgb"]["tree_num"] = 5
        self.config["xgb"]["num_iterations"] = 2


    @pytest.mark.parametrize(
        "model_name",
        model_names,
    )
    @pytest.mark.parametrize(
        "dataset_name",
        datasets,
    )
    def test_get_model_client(
        self, model_name, dataset_name
    ):
        self.config["model"] = model_name
        self.config['data_path'] = 'dataset/'
        self.config["dataset"] = dataset_name
        
        from flcore.client_selector import get_model_client
        from flcore.datasets import load_dataset
        data = load_dataset(self.config, 2)

        client = get_model_client(self.config, data, 2)

        assert client is not None


    @pytest.mark.parametrize(
        "model_name",
        model_names,
    )
    @pytest.mark.parametrize(
        "dataset_name",
        datasets,
    )
    def test_run(self, model_name, dataset_name):

        self.config["model"] = model_name
        self.config["dataset"] = dataset_name
        
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            config = self.config

        with open("tmp_test_config.yaml", "w") as f:
            yaml.dump(config, f)

        free_port(config["local_port"])
        run_log = open("run.log", "w")
        run_process = subprocess.Popen("python run.py tmp_test_config.yaml", shell=True, stdout=run_log, stderr=run_log)

        timer = Timer(180, run_process.kill)
        try:
            timer.start()
            run_process.communicate()
        finally:
            timer.cancel()
        
        # Print run_log
        run_log.close()
        run_log = open("run.log", "r")
        print(run_log.read())

        # Delete temporary config file
        os.remove("tmp_test_config.yaml")
        
        assert run_process.returncode == 0
