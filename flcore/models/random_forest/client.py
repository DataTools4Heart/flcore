import json
import joblib
import warnings
from pathlib import Path

import flwr as fl
import numpy as np
from sklearn.metrics import log_loss
import flcore.datasets as datasets
from flcore.serialization_funs import serialize_RF, deserialize_RF
import flcore.models.random_forest.utils as utils
from flcore.performance import measurements_metrics
from flcore.metrics import calculate_metrics
from sklearn.metrics import mean_squared_error

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
)
import time


# Define Flower client
class MnistClient(fl.client.Client):
    def __init__(self, data, config):
        self.config = config
        self.node_name = config["node_name"]
        n_folds_out= config['num_rounds']
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        self.splits_nested  = datasets.split_partitions(
                # ¿Qué es esto de folds?
                n_folds_out,
                config["test_size"],
                config["seed"],
                self.X_train,
                self.y_train,
                config["task"])
        self.model = utils.get_model(config)
        # Setting initial parameters, akin to model.compile for keras models
        # AQUI DEBERIA INICIALIZAR CON 0, ya que está en fit, que haga 1 iteración
        self.round = 0
        utils.set_initial_params_client(self.model,self.X_train, self.y_train)

    def get_parameters(self, ins: GetParametersIns):  # , config type: ignore
        params = utils.get_model_parameters(self.model)

        #Serialize to send it to server
        #It is forced to send an bytesIO
        parameters_to_ndarrays_final = serialize_RF(params)

        # Build and return response 
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters_to_ndarrays_final,
        )

    def fit(self, ins: FitIns):  # , parameters, config type: ignore
        parameters = ins.parameters
        #Deserialize to get the real parameters
        parameters = deserialize_RF(parameters)
        utils.set_model_params(self.model, parameters)
        metrics = {}
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_idx, val_idx = next(self.splits_nested)
            X_train_2 = self.X_train.iloc[train_idx, :]
            X_val = self.X_train.iloc[val_idx,:]
            y_train_2 = self.y_train.iloc[train_idx]
            y_val = self.y_train.iloc[val_idx]
            #To implement the center dropout, we need the execution time
            start_time = time.time()
            self.model.fit(X_train_2, y_train_2)
            #accuracy = model.score( X_test, y_test )
            # accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score = \
            # measurements_metrics(self.model,X_val, y_val)
            # ______________________________________________________________________________________
            # ESTO o se cambia para que sea consistente entre clasificación/regresión o se elimina
            #y_pred = self.model.predict(X_val)
            #metrics = calculate_metrics(y_val, y_pred, self.config)
            # ______________________________________________________________________________________

            # print(f"Accuracy client in fit:  {accuracy}")
            # print(f"Sensitivity client in fit:  {sensitivity}")
            # print(f"Specificity client in fit:  {specificity}")
            # print(f"Balanced_accuracy in fit:  {balanced_accuracy}")
            # print(f"precision in fit:  {precision}")
            # print(f"F1_score in fit:  {F1_score}")
            elapsed_time = (time.time() - start_time)
            metrics["running_time"] = elapsed_time

            print(f"num_client {self.node_name} has an elapsed time {elapsed_time}")
            
        print(f"Training finished for round {ins.config['server_round']}")

        # Serialize to send it to the server
        params = utils.get_model_parameters(self.model)
        parameters_updated = serialize_RF(params)

        if self.round % self.config["save_every_n_rounds"] == 0:
            self.save_model()

        self.round += 1

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.X_train),
            metrics=metrics,
        )
        

    def evaluate(self, ins: EvaluateIns):  # , parameters, config type: ignore
        parameters = ins.parameters
        #Deserialize to get the real parameters
        parameters = deserialize_RF(parameters)
        utils.set_model_params(self.model, parameters)
                
        ## AQUI TAMBIEN TENDRIAMOS QUE ADAPTAR PARA REGRESOR/CLASIFICADOR
        if self.config["task"] == "classification":
            if self.config["n_out"] == 1: # Binario
                y_pred_prob = self.model.predict_proba(self.X_test)
                loss = log_loss(self.y_test, y_pred_prob)
                # accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score = \
                # measurements_metrics(self.model,self.X_test, self.y_test)
                y_pred = self.model.predict(self.X_test)
                metrics = calculate_metrics(self.y_test, y_pred, self.config)
                # print(f"Accuracy client in evaluate:  {accuracy}")
                # print(f"Sensitivity client in evaluate:  {sensitivity}")
                # print(f"Specificity client in evaluate:  {specificity}")
                # print(f"Balanced_accuracy in evaluate:  {balanced_accuracy}")
                # print(f"precision in evaluate:  {precision}")
                # print(f"F1_score in evaluate:  {F1_score}")

                # Serialize to send it to the server
                #params = get_model_parameters(model)
                #parameters_updated = serialize_RF(params)
                # Build and return response
                status = Status(code=Code.OK, message="Success")
                return EvaluateRes(
                    status=status,
                    loss=float(loss),
                    num_examples=len(self.X_test),
                    metrics=metrics,
                )
            elif self.config["n_out"] > 1: # Multivariable
                # ************************************************** CORREGIR ADAPTAR
                # ************************************* Por ahora idéntico al binario
                y_pred_prob = self.model.predict_proba(self.X_test)
                loss = log_loss(self.y_test, y_pred_prob,labels=np.arange(self.config["n_out"]))
                # accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score = \
                # measurements_metrics(self.model,self.X_test, self.y_test)
                y_pred = self.model.predict(self.X_test)
                metrics = calculate_metrics(self.y_test, y_pred, self.config)
                # Serialize to send it to the server
                #params = get_model_parameters(model)
                #parameters_updated = serialize_RF(params)
                # Build and return response
                status = Status(code=Code.OK, message="Success")
                return EvaluateRes(
                    status=status,
                    loss=float(loss),
                    num_examples=len(self.X_test),
                    metrics=metrics,
                )

                # ************************************************** CORREGIR ADAPTAR
        elif self.config["task"] == "regression":
                y_pred = self.model.predict(self.X_test)
                loss = mean_squared_error(self.y_test, y_pred)
                metrics = calculate_metrics(self.y_test, y_pred, self.config)
                # Serialize to send it to the server
                #params = get_model_parameters(model)
                #parameters_updated = serialize_RF(params)
                # Build and return response
                status = Status(code=Code.OK, message="Success")
                return EvaluateRes(
                    status=status,
                    loss=float(loss),
                    num_examples=len(self.X_test),
                    metrics=metrics,
                )

    def save_model(self):
        save_path = Path(self.config["sandbox_path"]) / "model"
        save_path.mkdir(parents=True, exist_ok=True)

        model_name = f"{self.config['model']}_{self.config['task']}_round_{self.round}"

        model_path = save_path / f"{model_name}_model.joblib"
        joblib.dump(self.model, model_path)

        with open(self.config["metadata_file"], "r") as f:
            data_metadata = json.load(f)

        entity = data_metadata.get("entries", [])[0]

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

        metadata = {
            "node_name": self.config["node_name"],
            "task": self.config["task"],
            "n_out": self.config["n_out"],
            "n_feats": self.config["n_feats"],  # <-- FIX importante
            "model_type": self.config["model"],
            "feature_names": self.config["train_labels"],
            "target_names": self.config["target_labels"],
            "metrics": getattr(self, "last_metrics", None),
            "features_meta": features_meta,
            "outcomes_meta": outcomes_meta,
        }

        metadata_path = save_path / f"{model_name}_model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[Client {self.node_name}] Model saved at round {self.round} -> {model_path}")

def get_client(config,data) -> fl.client.Client:
    return MnistClient(data, config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
