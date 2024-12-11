import warnings
import os
import sys
from pathlib import Path

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

    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"

    # Read the config file

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    #Check the config file
    check_config(config)

    if config["production_mode"]:
        data_path = os.getenv("DATA_PATH")
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        ca_cert = os.getenv("FLOWER_SSL_CACERT") #  ca.crt
        server_cert = os.getenv("FLOWER_SSL_SERVER_CERT") # server.pem
        server_key = os.getenv("FLOWER_SSL_SERVER_KEY") # server.key
        certificates = (
            Path(f"{ca_cert}").read_bytes(),
            Path(f"{server_cert}").read_bytes(),
            Path(f"{server_key}").read_bytes(),
        )
         # certificates = (
        #     Path('.cache/certificates/rootCA_cert.pem').read_bytes(),
        #     Path('.cache/certificates/server_cert.pem').read_bytes(),
        #     Path('.cache/certificates/server_key.pem').read_bytes(),
        # )
    else:
        data_path = config["data_path"]
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
    os.system(f"cp {config_path} {experiment_dir}")

    (X_train, y_train), (X_test, y_test) = datasets.load_dataset(config)

    data = (X_train, y_train), (X_test, y_test)

    server, strategy = get_model_server_and_strategy(config, data)

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address=f"{central_ip}:{central_port}",
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"]),
        server=server,
        strategy=strategy,
        certificates = certificates,
    )
    # # Save the model and the history
    # filename = os.path.join( checkpoint_dir, 'final_model.pt' )
    # joblib.dump(model, filename)
    # Save the history as a yaml file
    print(history)
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
