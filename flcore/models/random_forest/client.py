import warnings

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
                self.y_train)
        self.model = utils.get_model(config)
        # Setting initial parameters, akin to model.compile for keras models
        # AQUI DEBERIA INICIALIZAR CON 0, ya que está en fit, que haga 1 iteración
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
            y_pred = self.model.predict(X_val)
            metrics = calculate_metrics(y_val, y_pred, self.config)
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
            if self.config["n_out"] > 1: # Multivariable
                # ************************************************** CORREGIR ADAPTAR
                # ************************************* Por ahora idéntico al binario
                y_pred_prob = self.model.predict_proba(self.X_test)
                loss = log_loss(self.y_test, y_pred_prob)
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

def get_client(config,data) -> fl.client.Client:
    return MnistClient(data, config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
