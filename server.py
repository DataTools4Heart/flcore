import warnings
import os
import sys
from pathlib import Path

import flwr as fl
import numpy
import yaml
import flcore.datasets as datasets
from flcore.server_selector import get_model_server_and_strategy
from flcore.smpc_module import SMPServerStrategy

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
        certificates = (
            Path('.cache/certificates/rootCA_cert.pem').read_bytes(),
            Path('.cache/certificates/server_cert.pem').read_bytes(),
            Path('.cache/certificates/server_key.pem').read_bytes(),
        )
    else:
        data_path = config["data_path"]
        central_ip = "LOCALHOST"
        central_port = "8080"
        certificates = None

    # Create experiment directory
    experiment_dir = Path("results") / config["experiment"]["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory for saving the model
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # History directory for saving the history
    history_dir = experiment_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_test, y_test) = datasets.load_dataset(config)

    data = (X_train, y_train), (X_test, y_test)

    # Check if smpc_url is provided in the config, otherwise set it 
    smpc_base_url = config.get("smpc", {}).get("smpc_url")

    if config.get("smpc", {}).get("use_smpc", True):
        smp_strategy = SMPServerStrategy(min_available_clients=2, smpc_base_url=smpc_base_url)
        server, strategy = get_model_server_and_strategy(config, data)
        strategy.configure_fit = smp_strategy.configure_fit
        strategy.aggregate_fit = smp_strategy.aggregate_fit
    else:
        # If use_smpc is False, use a regular strategy
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
# Save the history as a yaml file
    print(history)
    with open(history_dir / "results.txt", "w") as f:
        per_client_values = {}
        for metric in history.metrics_distributed:
            metric_values = history.metrics_distributed[metric][-1][1]
            if isinstance(metric_values, float):
                f.write(f"{metric} {metric_values:.4f} \n")
            else:
                for client_id, value in metric_values:
                    if metric not in per_client_values:
                        per_client_values[metric] = []
                    per_client_values[metric].append(round(value, 3))

        f.write(f"\nPer client results:\n")
        for metric in per_client_values:
            f.write(f"{metric} {per_client_values[metric]} \n")

    with open(history_dir / "history.yaml", "w") as f:
        yaml.dump(history, f)
