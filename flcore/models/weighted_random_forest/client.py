#############################################################################
#RF Agregator Code implemented by Esmeralda Ruiz Pujadas                   ##
#The Federated RF aggregator is implemented with/without drop out center.  ##
#In this version, I implemented the weight of each tree in order to add    ##
#the smoothing weights                                                     ##
#The client will merge all the trees and weights from the server           ##
#and ensamble all the trees weighted if enable                             ##
#Fit does not ensamble. Only create a new tree with a new partition        ##
#Evaluation is where the ensamble is performed and                         ## 
#the result is sent to the server                                          ##
#Feel free to change the method or improve it                              ##
#############################################################################


import operator
import warnings

import flwr as fl
import numpy as np
from sklearn.metrics import log_loss
import flcore.datasets as datasets
from flcore.serialization_funs import serialize_RF, deserialize_RF
import flcore.models.weighted_random_forest.utils as utils
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

from mlxtend.classifier import EnsembleVoteClassifier

#Ensamble in the level of RF trees
def ensambleRFTrees(parameters):
    list_classifiers = [None] * len(parameters)
    weights_classifiers = [None] * len(parameters)
    
    for i in range(len(parameters)):
        model = utils.get_model(True)
        model.estimators_= parameters[i][0][0]
        model.n_classes_ =2 
        model.n_outputs_ = 1 
        model.classes_ = np.array([j for j in range(model.n_classes_)])
        list_classifiers[i] = model
        #If RF is weighted, we will have
        #Three parameters (RF,num_examples, weight of the center)
        if(len(parameters[i])==3):
            weights_classifiers[i] = parameters[i][2]
        else:
            weights_classifiers[i] = 1

    return  list_classifiers,weights_classifiers

#Ensamble in the level of decisions trees
#that compose the RF trees so repeat the
#weight of each center for each 
#decision tree belonging to that center
#if enabled otherwise 1
def ensambleDecisionTrees(parameters):
    list_classifiers = []
    weights_classifiers = [] 

    for i in range(len(parameters)):
        list_classifiers = np.concatenate(((list_classifiers),(parameters[i][0][0])))
        if(len(parameters[i])==3):
            weights_smooth = parameters[i][2]
        else:
            weights_smooth = 1
        weights_classifiers = np.concatenate(((weights_classifiers),([weights_smooth]*len(parameters[i][0][0]))))

    #Represent the data as wanted for the majority voting
    #maybe you can simplify it
    list_classifiers_final = [None] * len(list_classifiers)
    weights_classifiers_final = [None] * len(weights_classifiers)

    for i in range(len(list_classifiers)):
        list_classifiers_final[i] = list_classifiers[i]
        weights_classifiers_final[i] = weights_classifiers[i] 

    return  list_classifiers_final,weights_classifiers_final           

# Define Flower client
class MnistClient(fl.client.Client):
    def __init__(self, data,client_id,config):
        self.client_id = client_id
        n_folds_out=config['num_rounds']
        seed=42
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = data
        self.splits_nested  = datasets.split_partitions(n_folds_out,0.2, seed, self.X_train, self.y_train)
        self.bal_RF = config['weighted_random_forest']['balanced_rf']
        self.model = utils.get_model(self.bal_RF) 
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params_client(self.model,self.X_train, self.y_train)
        self.ensamble_tree = []
        self.weight_ensamble_tree = []
        self.levelOfDetail = config['weighted_random_forest']['levelOfDetail']
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
            #We fit every fitting a new RF
            #with another partition to reduce 
            #variability. We do not consider the
            #accumulated trees of 'evaluate' as here
            #we want to create new RF trees.
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
            print(f"num_client {self.client_id} has an ellapsed time {ellapsed_time}")
            
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

        if(self.levelOfDetail == 'DecisionTree'):
            list_classifiers,weights_classifiers = ensambleDecisionTrees(parameters)
            #self.ensamble_tree = np.concatenate(((self.ensamble_tree),(list_classifiers)))
            #self.weight_ensamble_tree = np.concatenate(((self.weight_ensamble_tree),(weights_classifiers)))
        else:
            list_classifiers,weights_classifiers = ensambleRFTrees(parameters)
            
        #Merge the trees of all clients in each round
        self.ensamble_tree = (self.ensamble_tree)+(list_classifiers)
        self.weight_ensamble_tree = (self.weight_ensamble_tree)+(weights_classifiers)
  
                
       
        #Apply majority voting to the RF trees with the weights
        #fit_base_estimator= false does not fit the classifiers 
        #apply the majority voting from the models
        #weights=self.weight_ensamble_tree
        eclf = EnsembleVoteClassifier(clfs=self.ensamble_tree, fit_base_estimators=False,voting='hard',weights=self.weight_ensamble_tree)
        eclf.fit(self.X_train, self.y_train) 

        #utils.set_model_params(self.model, parameters)
        y_pred_prob = eclf.predict_proba(self.X_test)
        loss = log_loss(self.y_test, y_pred_prob)
        accuracy,specificity,sensitivity,balanced_accuracy, precision, F1_score = \
        measurements_metrics(eclf,self.X_test, self.y_test)
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
