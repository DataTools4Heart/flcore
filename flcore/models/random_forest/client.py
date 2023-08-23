import warnings

import flwr as fl
import numpy as np
from sklearn.metrics import log_loss
import flcore.datasets as datasets
from flcore.serialization_funs import serialize_RF, deserialize_RF
import flcore.models.random_forest.utils as utils
from flcore.performance import measurements_metrics
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
    def __init__(self, data,client_id,config):
        self.client_id = client_id
        n_folds_out=3
        seed=42
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        self.splits_nested  = datasets.split_partitions(n_folds_out,0.2, seed, self.X_train, self.y_train)
        self.bal_RF = config['random_forest']['balanced_rf']
        self.model = utils.get_model(self.bal_RF) 
        # Setting initial parameters, akin to model.compile for keras models
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
            accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score = \
            measurements_metrics(self.model,X_val, y_val)
            print(f"Accuracy client in fit:  {accuracy}")
            print(f"Sensitivity client in fit:  {sensitivity}")
            print(f"Specificity client in fit:  {specificity}")
            print(f"Balanced_accuracy in fit:  {balanced_accuracy}")
            print(f"precision in fit:  {precision}")
            print(f"F1_score in fit:  {F1_score}")
    
            ellapsed_time = (time.time() - start_time)
            print(f"num_client has an ellapsed time {ellapsed_time}")
            
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
            metrics= {"running_time":ellapsed_time},
        )
        

    def evaluate(self, ins: EvaluateIns):  # , parameters, config type: ignore
        parameters = ins.parameters
        #Deserialize to get the real parameters
        parameters = deserialize_RF(parameters)
        utils.set_model_params(self.model, parameters)
        y_pred_prob = self.model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, y_pred_prob)
        accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score = \
        measurements_metrics(self.model,self.X_test, self.y_test)
        print(f"Accuracy client in evaluate:  {accuracy}")
        print(f"Sensitivity client in evaluate:  {sensitivity}")
        print(f"Specificity client in evaluate:  {specificity}")
        print(f"Balanced_accuracy in evaluate:  {balanced_accuracy}")
        print(f"precision in evaluate:  {precision}")
        print(f"F1_score in evaluate:  {F1_score}")

        # Serialize to send it to the server
        #params = get_model_parameters(model)
        #parameters_updated = serialize_RF(params)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.X_test),
            metrics={"accuracy": float(accuracy),"sensitivity":float(sensitivity),"specificity":float(specificity)},
        )


def get_client(config,data,client_id) -> fl.client.Client:
    return MnistClient(data,client_id,config)
    # # Start Flower client
    # fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
