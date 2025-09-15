import warnings
import os
import sys
from pathlib import Path
import argparse
import json
import logging

import flwr as fl
import numpy
import yaml
import flcore.datasets as datasets
from flcore.server_selector import get_model_server_and_strategy
from flcore.compile_results import compile_results

warnings.filterwarnings("ignore")

def check_config(config):
    assert isinstance(config['num_clients'], int), 'num_clients should be an int'
    assert isinstance(config['num_rounds'], int), 'num_rounds should be an int'
    if(config['smooth_method'] != 'None'):
        assert config['smoothWeights']['smoothing_strenght'] >= 0 and config['smoothWeights']['smoothing_strenght'] <= 1, 'smoothing_strenght should be betwen 0 and 1'
    if(config['dropout_method'] != 'None'):
        assert config['dropout']['percentage_drop'] >= 0 and config['dropout']['percentage_drop'] < 100, 'percentage_drop should be betwen 0 and 100'

    assert (config['smooth_method']== 'EqualVoting' or \
        config['smooth_method']== 'SlowerQuartile' or \
        config['smooth_method']== 'SsupperQuartile' or \
        config['smooth_method']== 'None'), 'the smooth methods are not correct: EqualVoting, SlowerQuartile and SsupperQuartile'

    if(config['model'] == 'weighted_random_forest'):
         assert (config['weighted_random_forest']['levelOfDetail']== 'DecisionTree' or \
            config['weighted_random_forest']['levelOfDetail']== 'RandomForest'), 'the levels of detail for weighted RF are not correct: DecisionTree and RandomForest '


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads parameters from command line.")

    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of federated iterations")
    parser.add_argument("--model", type=str, default="random_forest", help="Model to train")
    parser.add_argument("--dataset", type=str, default="dt4h_format", help="Dataloader to use")
    parser.add_argument("--train_labels", type=str, nargs='+', default=None, help="Dataloader to use")
    parser.add_argument("--target_label", type=str, nargs='+', default=None, help="Dataloader to use")
    parser.add_argument("--sandbox_path", type=str, default="./sandbox", help="Sandbox path to use")
    #parser.add_argument("--certs_path", type=str, default="./", help="Certificates path")

    parser.add_argument("--smooth_method", type=str, default="EqualVoting", help="Weight smoothing")
    parser.add_argument("--smoothWeights", type=json.loads, default= {"smoothing_strenght": 0.5}, help="Smoothing parameters")
    parser.add_argument("--dropout_method", type=str, default=None, help="Determines if dropout is used")
    parser.add_argument("--dropout", type=json.loads, default={"percentage_drop":0}, help="Dropout parameters")
    parser.add_argument("--weighted_random_forest", type=json.loads, default={"balanced_rf": "true", "levelOfDetail": "DecisionTree"}, help="Weighted random forest parameters")
    parser.add_argument("--checkpoint_selection_metric", type=str, default="precision", help="Metric used for checkpoints")
    parser.add_argument("--production_mode", type=str, default="True",  help="Production mode")
    parser.add_argument("--neural_network", type=json.loads, default={"dropout_p": 0.2, "device": "cpu","local_epochs":100}, help="Neural Network parameters")

    #parser.add_argument("--Wdata_path", type=str, default=None, help="Data path")
    parser.add_argument("--local_port", type=int, default=8081, help="Local port")
    parser.add_argument("--experiment", type=json.loads, default={"name": "experiment_1", "log_path": "logs", "debug": "true"}, help="experiment logs")
    parser.add_argument("--random_forest", type=json.loads, default={"balanced_rf": "true"}, help="Random forest parameters")
    parser.add_argument("--n_features", type=int, default=0, help="Number of features")
    parser.add_argument("--metrics_aggregation", type=str, default="weighted_average",  help="Metrics")
    parser.add_argument("--strategy", type=str, default="FedAvg",  help="Metrics")

    args = parser.parse_args()

    config = vars(args)

    if config["model"] in ("logistic_regression", "elastic_net", "lsvc"):
        print("LINEAR", config["model"], config["n_features"])
        config["linear_models"] = {}
        config['linear_models']['n_features'] = config["n_features"]
        config["held_out_center_id"] = -1
    elif config["model"] == "nn": # in ("nn", "BNN"):
#        config["n_feats"] = config["n_features"]
        config["n_feats"] = len(config["train_labels"])
        config["n_out"] = 1 # Quizás añadir como parámetro también
        config["dropout_p"] = config["neural_network"]["dropout_p"]
        config["device"] = config["neural_network"]["device"]
        config["batch_size"] = 32
        config["lr"] = 1e-3
        config["local_epochs"] = config["neural_network"]["local_epochs"]

    config["min_fit_clients"] = config["num_clients"]
    config["min_evaluate_clients"] = config["num_clients"]
    config["min_available_clients"] = config["num_clients"]

    experiment_dir = Path(os.path.join(config["experiment"]["log_path"], config["experiment"]["name"]))
    config["experiment_dir"] = experiment_dir

    # Create sandbox log file path
# Originalmente estaba asi:
#    sandbox_log_file = Path(os.path.join("./sandbox", "log_server.txt"))
    sandbox_log_file = Path(os.path.join(config["sandbox_path"], "log_server.txt"))

    # Set up the file handler (writes to file)
    file_handler = logging.FileHandler(sandbox_log_file)
    file_handler.setLevel(logging.DEBUG)

    # Set up the console handler (writes to Docker logs via stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter for consistency
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Get the root logger and configure it
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear any default handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect print() and sys.stdout/sys.stderr into logger
    class StreamToLogger:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level

        def write(self, message):
            for line in message.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

        def flush(self):
            pass

    # Create two sub-loggers
    stdout_logger = logging.getLogger("STDOUT")
    stderr_logger = logging.getLogger("STDERR")

    # Redirect standard output and error to logging
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    # Now you can use logging in both places
    logging.debug("This will be logged to both the console and the file.")

    # Your existing code continues here...
    # For example, the following logs will go to both stdout and file:
    logging.debug("Starting Flower server...")

    #Check the config file
    check_config(config)
    if config["production_mode"] == "True":
        print("TRUE")
        #data_path = ""
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")

        ca_cert = Path(os.path.join("/certs","rootCA_cert.pem"))
        server_cert =  Path(os.path.join("/certs","server_cert.pem"))
        server_key =  Path(os.path.join("/certs","server_key.pem"))

        certificates = (
            Path(f"{ca_cert}").read_bytes(),
            Path(f"{server_cert}").read_bytes(),
            Path(f"{server_key}").read_bytes(),
        )
#            Path('.cache/certificates/rootCA_cert.pem').read_bytes(),
#            Path('.cache/certificates/server_cert.pem').read_bytes(),
#            Path('.cache/certificates/server_key.pem').read_bytes(),
    else:
        print("ELSE")
        #data_path = config["data_path"]
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        certificates = None

    # Create experiment directory
    experiment_dir = Path(os.path.join(config["experiment"]["log_path"], config["experiment"]["name"]))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config["experiment_dir"] = experiment_dir

    # Checkpoint directory for saving the model
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # # History directory for saving the history
    # history_dir = experiment_dir / "history"
    # history_dir.mkdir(parents=True, exist_ok=True)

    # Copy the config file to the experiment directory

    with open("config.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    os.system(f"cp config.yaml {experiment_dir}")

    if config["strategy"] == "UncertaintyWeighted":
        if config["model"] == "nn":
            pass
        else:
           print("UncertaintyWeighted is only available for NN")
           print("Changing strategy to FedAvg")
           config["strategy"] = "FedAvg"

    server, strategy = get_model_server_and_strategy(config)

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address=f"{central_ip}:{central_port}",
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"], round_timeout=None ),
        server=server,
        strategy=strategy,
        certificates = certificates,
    )
    # # Save the model and the history
    # filename = os.path.join( checkpoint_dir, 'final_model.pt' )
    # joblib.dump(model, filename)
    # Save the history as a yaml file
    print(history)
    """
    with open(experiment_dir / "metrics.txt", "w") as f:
        f.write(f"Results of the experiment {config['experiment']['name']}\n")
        f.write(f"Model: {config['model']}\n")
        f.write(f"Data: {config['dataset']}\n")
        f.write(f"Number of clients: {config['num_clients']}\n")

        # selection_metric = 'val ' + config['checkpoint_selection_metric']
        selection_metric = config['checkpoint_selection_metric']
        # Get index of tuple of the best round
        best_round = int(numpy.argmax([round[1] for round in history.metrics_distributed[selection_metric]]))
        training_time = history.metrics_distributed_fit['training_time [s]'][-1][1]
        f.write(f"Total training time: {training_time:.2f} [s] \n")
        f.write(f"Best checkpoint based on {selection_metric} after round: {best_round}\n\n")
        print(f"Best checkpoint based on {selection_metric} after round: {best_round}\n\n")

        f.write(f"\nAggregated results:\n\n")

        # best_round = best_round - 1
        per_client_values = {}
        for metric in history.metrics_distributed:
            metric_value = history.metrics_distributed[metric][best_round][1]
            if type(metric_value) in [int, float, numpy.float64]:
                f.write(f"{metric} {metric_value:.4f} \n")
            else:
                for per_client_metric_value in metric_value:
                    metric = metric.replace("per client ", "")
                    if metric not in per_client_values:
                        per_client_values[metric] = []
                    per_client_values[metric].append(round(per_client_metric_value, 3))
        
        f.write(f"\n\nPer client results:\n\n")
        for metric in per_client_values:
            f.write(f"{metric} {per_client_values[metric]} \n")
        
        f.write(f"\n\nHeld out set evaluation:\n\n")
        for metric in history.metrics_centralized:
            # print(f"Len of centralized metric {metric} ", len(history.metrics_centralized[metric]))
            if len(history.metrics_centralized[metric]) == 1:
                metric_value = history.metrics_centralized[metric][0][1]
            else:
                metric_value = history.metrics_centralized[metric][best_round][1]
            if type(metric_value) in [int, float, numpy.float64]:
                f.write(f"{metric} {metric_value:.4f} \n")

dict_history = {}
history = history.__dict__
for logs in history.keys():
    if isinstance(history[logs], list):
        history[logs] = [float(loss) for (round, loss) in history[logs]]
    if isinstance(history[logs], dict):
        for metric in history[logs]:
            extracted_values = [value for (round, value) in history[logs][metric]]
            if isinstance(extracted_values[0], list):
                # Convert list elements to float
                extracted_values = [[float(value) for value in sublist] for sublist in extracted_values]
            else:
                extracted_values = [float(value) for value in extracted_values]
            history[logs][metric] = extracted_values


with open(experiment_dir / "history.yaml", "w") as f:
    yaml.dump(history, f)

# Compile the results
compile_results(experiment_dir)
"""
