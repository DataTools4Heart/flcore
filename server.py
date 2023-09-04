import warnings
from pathlib import Path

import flwr as fl
import yaml

import flcore.datasets as datasets
from flcore.server_selector import get_model_server_and_strategy

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
    
    assert (config['model']== 'linear_models' or \
            config['model']== 'xgb' or \
            config['model']== 'random_forest' or \
            config['model']== 'weighted_random_forest' or \
            config['model']== 'logistic_regression'), 'the ML methods are not correct: linear_models. xgb, random_forest,weighted_random_forest' 
    
    if(config['model']== 'linear_models'):
         assert (config['linear_models']['model_type']== 'LR' or \
            config['linear_models']['model_type']== 'elastic_net' or \
            config['linear_models']['model_type']== 'LSVC'), 'the Linear models are not correct: LR, LSVC, elastic_net '
        

if __name__ == "__main__":
    # Read the config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    #Check the config file
    check_config(config)

    # Create experiment directory
    experiment_dir = Path("results") / config["experiment"]["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint directory for saving the model
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # History directory for saving the history
    history_dir = experiment_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    server, strategy = get_model_server_and_strategy(config)

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"]),
        server=server,
        strategy=strategy,
        # certificates = (
        #     Path( '.cache/certificates/rootCA_cert.pem' ).read_bytes(),
        #     Path( '.cache/certificates/server_cert.pem' ).read_bytes(),
        #     Path( '.cache/certificates/server_key.pem'  ).read_bytes(),
        # ),
    )
    # # Save the model and the history
    # filename = os.path.join( checkpoint_dir, 'final_model.pt' )
    # joblib.dump(model, filename)
    # Save the history as a yaml file
    print(history)
    with open(history_dir / "history.yaml", "w") as f:
        yaml.dump(history, f)
