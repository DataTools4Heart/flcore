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
    # "weighted_random_forest",
    # "xgb"
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

        free_port(config["local_port"])
        run_log = open("run.log", "w")
        run_process = subprocess.Popen("python run.py", shell=True, stdout=run_log, stderr=run_log)

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
        
        assert run_process.returncode == 0

        #     try:
        #         timer.start()
        #         client_process.communicate()
        #     finally:
        #         timer.cancel()

        #     assert client_process.returncode == 0
    
        # print("Starting server")
        # server_process = subprocess.Popen("python server.py", shell=True)
        # time.sleep(20)

        # client_processes = []
        # for i in range(1, config["num_clients"] + 1):
        #     print("Starting client " + str(i))
        #     client_processes.append(
        #         subprocess.Popen("python client.py " + str(i), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #     )

        # for client_process in client_processes:
        #     timer = Timer(30, client_process.kill)
        #     try:
        #         timer.start()
        #         client_process.communicate()
        #     finally:
        #         timer.cancel()

        #     assert client_process.returncode == 0

        # timer = Timer(30, server_process.kill)
        # try:
        #     timer.start()
        #     server_process.communicate()
        # finally:
        #     timer.cancel()
        
        # assert server_process.returncode == 0
