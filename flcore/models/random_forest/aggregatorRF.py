#############################################################################
#RF Agregator Code implemented by Esmeralda Ruiz Pujadas                   ##
#The Federated RF aggregators implemented are:                             ##
#1. The aggregation add all the estimators in the server                   ##
#2. The aggrefation add random trees                                       ##
#3. Aggregation selects according to a probability distribution (default)  ##
#Feel free to extend it                                                    ##
#Another interesting paper is to aggregate via accuracy (not implemented)  ##                                     ##
#https://link.springer.com/chapter/10.1007/978-3-031-08333-4_11#Sec3       ##
#https://ieeexplore.ieee.org/document/9867984                              ##
#############################################################################


from logging import WARNING
from typing import  Dict, List,Callable, Optional,Tuple,Union
#from dropout import Fast_at_odd_rounds

from flwr.common import  FitIns, FitRes,EvaluateRes, MetricsAggregationFn, NDArrays, Parameters,  Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from flcore.serialization_funs import serialize_RF, deserialize_RF

import numpy as np
from flcore.models.random_forest.utils import get_model
import random
import time
import flwr.server.strategy.fedavg as fedav
from flcore.dropout import select_clients
from flcore.smoothWeights import smooth_aggregate,computeSmoothedWeights

#############################
#  AGGREGATOR 1: RANDOM DT  #
#############################

def aggregateRF_random(rfs,config):
    rfa= get_model(config)
    number_Clients = len(rfs)
    numberTreesperclient = int(len(rfs[0][0][0]))
    random_select = int(numberTreesperclient/number_Clients)
    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    rf0 = np.concatenate((random.choices(rfs[0][0][0],k=random_select), random.choices(rfs[1][0][0],k=random_select)))
    for i in range(2,len(rfs)):
        rf0 = np.concatenate((rf0, random.choices(rfs[i][0][0],k=random_select)))
    rfa.estimators_=np.array(rf0)
    rfa.n_estimators = len(rfa.estimators_)

    return [rfa],rfa.estimators_


def aggregateRF_withprevious_random(rfs,previous_estimators,config):
    rfa= get_model(config)
    number_Clients = len(rfs)
    numberTreesperclient = int(len(rfs[0][0][0]))
    random_select =int(numberTreesperclient/number_Clients)
    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    rf0 = np.concatenate((random.choices(rfs[0][0][0],k=random_select), random.choices(rfs[1][0][0],k=random_select)))
    for i in range(2,len(rfs)):
        rf0 = np.concatenate((rf0, random.choices(rfs[i][0][0],k=random_select)))

    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    all_concats = np.concatenate((rf0,previous_estimators))
    rfa.estimators_=np.array(all_concats)
    rfa.n_estimators = len(rfa.estimators_)

    return [rfa],rfa.estimators_


#############################
#  AGGREGATOR 2: ALL DT     #
#############################
#We merge all the trees in one RF
#https://ai.stackexchange.com/questions/34250/random-forests-are-more-estimators-always-better
def aggregateRF(rfs,bal_RF):
    rfa= get_model(bal_RF)
    #number_Clients = len(rfs)
    numberTreesperclient = int(len(rfs[0][0][0]))
    random_select = numberTreesperclient #int(numberTreesperclient/number_Clients)
    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    rf0 = np.concatenate(((rfs[0][0][0]), (rfs[1][0][0])))
    for i in range(2,len(rfs)):
        rf0 = np.concatenate((rf0, (rfs[i][0][0])))
    rfa.estimators_=np.array(rf0)
    rfa.n_estimators = len(rfa.estimators_)

    return [rfa],rfa.estimators_
    

#We merge all the trees in one RF
#https://ai.stackexchange.com/questions/34250/random-forests-are-more-estimators-always-better
def aggregateRF_withprevious(rfs,previous_estimators,config):
    rfa= get_model(config)
    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    rf0 = np.concatenate(((rfs[0][0][0]), (rfs[1][0][0])))
    for i in range(2,len(rfs)):
        rf0 = np.concatenate((rf0, (rfs[i][0][0])))

    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    all_concats = np.concatenate((rf0,previous_estimators))
    rfa.estimators_=np.array(all_concats)
    rfa.n_estimators = len(rfa.estimators_)

    return [rfa],rfa.estimators_


##############################################
#  AGGREGATOR 2: RANDOM WITH PROB DISTRI     #
##############################################
#In this version of aggregation we weight according to smoothing
#weigth, we transform into probability /sum(weights)
#and random choice select according to probability distribution
#def aggregateRFwithSizeCenterProbs(rfs,bal_RF,smoothing_method,smoothing_strenght):
def aggregateRFwithSizeCenterProbs(rfs,config):
    rfa= get_model(config)
    numberTreesperclient = int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    if(config["smooth_method"] != 'None'):
        weights_centers = computeSmoothedWeights(rfs,config["smooth_method"],config["smoothing_strenght"])
    else:
        #If smooth weights is not available all the trees have the
        #same probability
        weights_centers = [1]*(number_Clients)
    for i in range(number_Clients):
        list_classifiers = np.concatenate(((list_classifiers),(rfs[i][0][0])))
        weights_smooth = weights_centers[i]
        weights_classifiers = np.concatenate(((weights_classifiers),([weights_smooth]*len(rfs[i][0][0]))))

    weights_classifiers = weights_classifiers / sum(weights_classifiers )
    client_indices = np.random.choice([j for j in range(len(list_classifiers))], numberTreesperclient, p=weights_classifiers)
    
    selectedTrees = list_classifiers[client_indices]
    weights_selectedTrees = weights_classifiers[client_indices]


    rfa.estimators_=np.array(selectedTrees)
    rfa.n_estimators = len(selectedTrees)

    return [rfa],rfa.estimators_,weights_selectedTrees

#def aggregateRFwithSizeCenterProbs_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght):
def aggregateRFwithSizeCenterProbs_withprevious(rfs,previous_estimators,previous_estimator_weights,config):
#            [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithSizeCenterProbs(rfs,bal_RF,smoothing_method,smoothing_strenght)
            [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithSizeCenterProbs(rfs,config)

            rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
            rfa.estimators_=np.array(rfa.estimators_)
            rfa.n_estimators = len(rfa.estimators_)

            weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

            return [rfa],rfa.estimators_,weights_selectedTrees
