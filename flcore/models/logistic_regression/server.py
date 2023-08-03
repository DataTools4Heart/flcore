import flwr as fl
import flcore.models.logistic_regression.utils as utils

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression, data):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = data
    # (X_train, y_train), (X_test, y_test) = datasets.load_cvd('dataset', 'All')


    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

def get_server_and_strategy(config, data):
    
    model = LogisticRegression()
    utils.set_initial_params(model, data)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=config['num_clients'],
        evaluate_fn=get_evaluate_fn(model, data),
        on_fit_config_fn=fit_round,
    )
    
    return None, strategy

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
