# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

# client/models/base_model.py

from abc import ABC, abstractmethod

class BaseSurvivalModel(ABC):
    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, params):
        pass