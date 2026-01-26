# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from abc import ABC, abstractmethod
from typing import List, Any

class BaseAggregator(ABC):
    """
    Base class for all federated model aggregators.
    Each model type should implement `aggregate` based on its own parameters structure.
    """

    def __init__(self, models: List[Any], weights: List[int] = None):
        """
        models: list of model parameters from clients (output of get_parameters)
        weights: optional list of integers to weight client contributions
        """
        self.models = models
        self.weights = weights if weights is not None else [1] * len(models)

    @abstractmethod
    def aggregate(self):
        """
        Aggregate the parameters from clients and return the aggregated model parameters.
        Must be implemented by each specific model aggregator.
        """
        pass