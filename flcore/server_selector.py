# import flcore.models.logistic_regression.server as logistic_regression_server
import flcore.models.logistic_regression.server as logistic_regression_server
import flcore.models.xgb.server as xgb_server


def get_model_server_and_strategy(config, data=None):
    model = config["model"]

    if model == "logistic_regression":
        server, strategy = logistic_regression_server.get_server_and_strategy(
            config, data
        )
    elif model == "rf":
        pass
    elif model == "xgb":
        server, strategy = xgb_server.get_server_and_strategy(config, data)
    else:
        raise ValueError(f"Unknown model: {model}")

    return server, strategy
