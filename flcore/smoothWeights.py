###################################################################
#Smooth code implemented by Esmeralda Ruiz                       ##
#Following the Akis paper of dropout center                      ##
#https://link.springer.com/chapter/10.1007/978-3-031-09002-8_42  ##
#The Akis'code implements equal volting in:                      ##
#https://github.com/Linardos/BFP/blob/master/src/aggregator.py   ##
#I also added SlowerQuartile and SupperQuartile proposed in      ##
#his paper in smooth_aggregate_quartiles                         ##
#In config, enable or disable the smoothing                      ## 
# None                                                           ##
# EqualVoting                                                    ##
# SlowerQuartile                                                 ##
# SsupperQuartile                                                ##
###################################################################


from typing import List, Tuple
from functools import reduce
import numpy as np


def computeSmoothedWeights(results,smoothing_method,smoothing_strenght):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    num_centers = len(results)
    homogeneous_weights = [1 / num_centers for _ in range(num_centers)]

    # None, or float in [0,1]. 0 equals Federated Averaging.
        
    # if smoothing:
    #if CONFIG['strategy']['smoothing']:
    if smoothing_method!= 'None':
        examples_per_center = [num_examples for _, num_examples in results]
        default_f_weights = [examples_per_center[i] / num_examples_total for i in range(num_centers)]
        # assert round(sum(default_f_weights),3) == 1, "Default weights do not sum to 1, sum: {}".format(sum(default_f_weights))
        smoothing_value = smoothing_strenght #CONFIG['strategy']['smoothing']
        if(smoothing_method=='EqualVoting'): #equal voting 
            # x*smoothing+y*(1-smoothing) where x is the default weight and y is a homogenous weight
            final_weights = [(d*(1-smoothing_value)+h*smoothing_value) for d, h in zip(default_f_weights, homogeneous_weights)]
            # assert round(sum(final_weights),3) == 1, "Final weights after smoothing do not sum to 1, sum: {}".format(sum(final_weights))
        else: #quartile options
            #Savg =(default_f_weight+homogeneus_weight)/2
            Savg = [(d+f)/2 for d,f in zip(default_f_weights, homogeneous_weights)]
            if(smoothing_method=='SlowerQuartile'):
                #SlowerQuartile = (default_f_weight+Svag)/2
                SlowerQuartile = [(d+Smean)/2 for d,Smean in zip(default_f_weights, Savg)]
                final_weights = SlowerQuartile
            else:
                #SupperQuartile = (homogeneous_weights+Svag)/2
                SupperQuartile = [(h+Smean)/2 for h,Smean in zip(homogeneous_weights, Savg)]
                final_weights = SupperQuartile
    else:
        final_weights = homogeneous_weights

    return final_weights

def smooth_aggregate(results,smoothing_method,smoothing_strenght) :
    final_weights = computeSmoothedWeights(results,smoothing_method,smoothing_strenght)

    # Create a list of weights, each multiplied by the related number of examples
    # weighted_weights = [
    #     [layer * num_examples for layer in weights] for weights, num_examples in results
    # ]

    unweighted_weights = [weights for weights, _ in results]

    weighted_weights = [
        [layer * f_weight for layer in weights] for weights, f_weight in zip(unweighted_weights, final_weights)
    ]


    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime