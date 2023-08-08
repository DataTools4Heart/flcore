import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import flcore.models.logistic_regression.utils as utils


if __name__ == "__main__":
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, data):
        self.model = LogisticRegression(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
        )
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model, data)
    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        print(f"Training finished for round {config['server_round']}")
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


def get_client(data) -> fl.client.Client:
    return MnistClient(data)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
