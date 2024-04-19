from collections import OrderedDict
from typing import List, Optional, Dict, Tuple
import numpy as np
import flwr as fl


class DLClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader=None):
        """
        Initialize the model and provide the data

        Note: model can be initialized with the shape information of the data,
        however it cannot change it's shape based on data values characteristics.
        Ensure, that the model's architecture stays the same with different subsets of same dataset
        used for initialization.
        """
        self.model = model
        self.net = self.model.model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config=None) -> List[np.ndarray]:
        """
        Return the parameters of the model in an array format
        """
        return self.model.get_parameters()

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set the parameters of the local model
        """
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        """
        Train the model for a specified number of steps/epochs.

        Note: ensure that the model is not reinitialzied in this method, it
        should continue training from the previous state
        """
        self.set_parameters(parameters)
        self.model.train(self.trainloader)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluation method for the model

        It may be called after each round of training
        A dictionary with metrics as keys and values as floats may be returned
        """
        self.set_parameters(parameters)
        if self.valloader is None:
            return float(-1), len(self.trainloader), {}
        else:
            loss, accuracy = self.model.test(self.valloader)
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# Sample loading of the model and data


# if __name__ == "__main__":
#     model = ModelPipeline()
#     trainloader = model.dataloader
#     valloader = model.dataloader
#     client = DLClient(model, trainloader).to_client()
#     fl.client.start_client(server_address="[::]:8080", client=client)
