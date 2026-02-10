#import flcore.models.logistic_regression.server as logistic_regression_server
#import flcore.models.logistic_regression.server as logistic_regression_server
import flcore.models.xgblr.server as xgblr_server
import flcore.models.random_forest.server as random_forest_server
import flcore.models.linear_models.server as linear_models_server
import flcore.models.weighted_random_forest.server as weighted_random_forest_server


def get_model_server_and_strategy(config, data=None):
    model = config["model"]

    if model in ("logistic_regression", "elastic_net", "lsvc"):
        server, strategy = linear_models_server.get_server_and_strategy(
            config
        )
    elif model in ("random_forest", "balanced_random_forest"):
        server, strategy = random_forest_server.get_server_and_strategy(
            config
        )
    elif model == "weighted_random_forest":
        server, strategy = weighted_random_forest_server.get_server_and_strategy(
            config
        )
        
    elif model == "xgblr":
        server, strategy = xgblr_server.get_server_and_strategy(config, data)
    else:
        raise ValueError(f"Unknown model: {model}")

    return server, strategy
