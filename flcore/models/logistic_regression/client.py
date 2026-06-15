
import json
import joblib
import warnings
import flwr as fl
import numpy as np
from pathlib import Path

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

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
    def __init__(self, data, config=None):
        self.config = config
        self.model = LogisticRegression(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
        )
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        # Setting initial parameters, akin to model.compile for keras models
        self.round = 0
        utils.set_initial_params(self.model, data)

    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        try:
            utils.set_model_params(self.model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(self.X_train, self.y_train)
            print(f"Training finished for round {config['server_round']}")
            
            if self.config and "save_every_n_rounds" in self.config:
                if self.round % self.config["save_every_n_rounds"] == 0:
                    self.save_model()

            self.round += 1
            return utils.get_model_parameters(self.model), len(self.X_train), {}
        except Exception as e:
            from flcore.utils import log_detailed_error
            log_detailed_error("Model Fitting (Local Training)", e, config=self.config or config, X=getattr(self, "X_train", None), y=getattr(self, "y_train", None))
            raise e

    def evaluate(self, parameters, config):  # type: ignore
        try:
            utils.set_model_params(self.model, parameters)
            loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
            accuracy = self.model.score(self.X_test, self.y_test)
            return loss, len(self.X_test), {"accuracy": accuracy}
        except Exception as e:
            from flcore.utils import log_detailed_error
            log_detailed_error("Model Evaluation (Local Validation)", e, config=self.config or config, X=getattr(self, "X_test", None), y=getattr(self, "y_test", None))
            raise e

    def save_model(self):
        save_path = Path(self.config["sandbox_path"])/"model"
        save_path.mkdir(parents=True, exist_ok=True)
        model_name = self.config["model"]+"_"+self.config["task"]+"_round_"+str(self.round)
        model_path = save_path / f"{model_name}_model.joblib"
        joblib.dump(self.model, model_path)

        data_metadata = json.load(open(self.config["metadata_file"], "r"))
        entity = data_metadata.get("entries", {})[0]
        features_list = entity.get("features", [])
        outcomes_list = entity.get("outcomes", [])
        dataset_stats = entity.get("datasetStats", {})
        feature_stats = dataset_stats.get("featureStats", {})
        outcome_stats = dataset_stats.get("outcomeStats", {})

        all_features_meta = {f['name']: f for f in features_list}
        all_outcomes_meta = {o['name']: o for o in outcomes_list}

        for f_name, f_meta in all_features_meta.items():
            stats = feature_stats.get(f_name, {})
            f_meta['stats'] = stats

        for o_name, o_meta in all_outcomes_meta.items():
            stats = outcome_stats.get(o_name, {})
            o_meta['stats'] = stats

        features_meta = {}
        for label in self.config["train_labels"]:
            if label in all_features_meta:
                features_meta[label] = all_features_meta[label]
            elif label in all_outcomes_meta:
                features_meta[label] = all_outcomes_meta[label]

        outcomes_meta = {}
        for label in self.config["target_labels"]:
            if label in all_outcomes_meta:
                outcomes_meta[label] = all_outcomes_meta[label]
            elif label in all_features_meta:
                outcomes_meta[label] = all_features_meta[label]

#>>> features_meta["patient_demographics_age"]["stats"]["min"]
        metadata = {
            "node_name": self.config["node_name"],
            "task": self.config["task"],
            "n_out": self.config["n_out"],
            "n_out": self.config["n_feats"],
            "model_type": self.config["model"],
            "feature_names": self.config["train_labels"],
            "target_names":self.config["target_labels"],
            "metrics": getattr(self, "last_metrics", None),
            "features_meta": features_meta,
            "outcomes_meta": outcomes_meta
        }

        metadata_path = save_path / f"{model_name}_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Model and metadata saved for inference at {save_path}")

def get_client(config, data) -> fl.client.Client:
    return MnistClient(data, config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
