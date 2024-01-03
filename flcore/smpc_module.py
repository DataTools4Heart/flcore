# Weight Sharing Format: The current implementation assumes that the model weights can be flattened and sent as a list of floats.
# This approach works well for simple models but may need adjustments for more complex architectures. 
# Deep learning models with convolutional layers, for example, may have multi-dimensional weight tensors.

# smpc_module.py
import os
import requests
from time import sleep
import numpy as np
from flwr.common import FitIns, Parameters, ndarrays_to_parameters
import flwr as fl

# Prefix for random keys used in the SMPClient
randomprefix = "asdfa"

class SMPClient:
    def __init__(self, model, smpc_base_url, client_id):
        """
        SMPClient class for sharing weights with an external Secure Multi-Party Computation (SMPC) server.

        Args:
        - model: The machine learning model for which weights will be shared.
        - smpc_base_url: The base URL of the SMPC server.
        - client_id: Unique identifier for the client.
        """
        self.model = model
        self.smpc_base_url = smpc_base_url
        self.round = 0  # Initialize the round to 0
        self.client_id = client_id
        self.client_base_url = f"http://{os.getenv('SMPC_SERVER_IP', '167.71.139.232')}:900{client_id}/api/update-dataset/"



    def share_weights(self, weights, config):
        """
        Share the flattened weights of the model with an external SMPC server.

        Args:
        - weights: List of weight arrays from the model.
        - config: Configuration parameters, including the SMPC server URL and round information.
        """
        print(f"SMPC URL: {self.smpc_base_url}")
        smpc_weights = []
        for w in weights:
            print(f"Shape before flattening: {w.shape}")
            flat_weights = w.flatten().tolist()
            print(f"Shape after flattening: {len(flat_weights)}")
        for arr in [w.flatten().tolist() for w in weights]:
            smpc_weights.extend(arr)

        data = {
            "type": "float",
            "data": smpc_weights
        }

        # Increment the round each time share_weights is called
        self.round += 1
        request = self.client_base_url + "testKey" + randomprefix + str(self.round)
        print(request)
        print(data)
        response = requests.post(
                self.client_base_url + "testKey" + randomprefix + str(self.round), json=data)
        print("response", response)
        if response.ok:
            print("SMPC Request was successful!")
            print(response.text)
        else:
            print(f"SMPC Request failed with status code {response.status_code}.")
            print(response.text)

class SMPClientEvaluator:
    def __init__(self, model):
        """
        SMPClientEvaluator class for evaluating model performance based on received parameters.

        Args:
        - model: The machine learning model for which parameters will be set for evaluation.
        """
        self.model = model

    def evaluate(self, parameters, config):
        """
        Evaluate the model based on the received parameters.

        Args:
        - parameters: Model parameters received from the server.
        - config: Configuration parameters, including test data information.
        
        Returns:
        Tuple containing loss, number of test samples, and accuracy.
        """
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(config["x_test"], config["y_test"])
        return loss, len(config["y_test"]), {"accuracy": float(accuracy)}

# SMPServerStrategy class
class SMPServerStrategy(fl.server.strategy.FedAvg):
    def __init__(self, min_available_clients=2, smpc_base_url="http://167.71.139.232:12314/api/"):
        """
        SMPServerStrategy class for defining the server-side strategy in a federated learning scenario.

        Args:
        - min_available_clients: Minimum number of available clients required for aggregation.
        - smpc_base_url: The base URL of the SMPC server.
        """
        super().__init__(min_available_clients=min_available_clients)
        self.smpc_base_url = os.getenv("SMPC_SERVER_IP", "http://167.71.139.232:12314") + "/api/secure-aggregation/job-id/"
        self.result_base_url = os.getenv("SMPC_SERVER_IP", "http://167.71.139.232:12314") + "/api/get-result/job-id/"

        self.triggerBody = {
            "computationType": "fsum",
            "returnUrl": "http://localhost:4100",
            "clients": ["WomenHealthClinica", "ChildrensHospital"]
        }

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate the results from the clients using SMP-specific logic.

        Args:
        - server_round: The current server round.
        - results: Dictionary containing results from clients.
        - failures: List of clients that failed during the round.

        Returns:
        Aggregated model parameters.
        """
        response = requests.post(
            self.smpc_base_url + "testKey" + randomprefix + str(server_round), json=self.triggerBody)
        if response.ok:
            print("Request was successful!")
            print(response.text)
        else:
            print(f"Request failed with status code {response.status_code}.")
            print(response.text)

        while 1:
            response = requests.get(
                self.result_base_url + "testKey" + randomprefix + str(server_round))
            print("Response got ", self.result_base_url, response)
            if response.ok:
                print("Request was successful!")
                json_data = response.json()
                print("Result:", json_data)
                if "computationOutput" in json_data:
                    computation_output = json_data["computationOutput"]
                    try:
                        # Conditionally reshape only if the array size allows it
                        if len(computation_output) > 1:
                            reshaped_output = np.array(computation_output).reshape(-1, 10)
                            res = ndarrays_to_parameters([reshaped_output])
                        else:
                            res = ndarrays_to_parameters([computation_output])
                        print("FINAL RESULT", res)
                        return super().aggregate_fit(server_round, results, failures)
                    except ValueError as e:
                        print(f"Error while reshaping array: {e}")
                        # Handle error, e.g., log it or return default parameters
                        return super().aggregate_fit(server_round, results, failures)
                elif "status" in json_data and json_data["status"] == "FAILED":
                    # Handle failure by logging and continuing with the aggregation
                    print(f"Computation failed: {json_data['message']}")
                    return super().aggregate_fit(server_round, results, failures)
            else:
                print(f"Request failed with status code {response.status_code}.")
                print(response.text)
            sleep(1)
        return super().aggregate_fit(server_round, results, failures)

