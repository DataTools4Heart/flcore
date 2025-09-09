# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Uncertainty-Aware Neural Network
# Author: Jorge Fabila Fabian
# Fecha: September 2025
# Project: DT4H
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from typing import Dict, Optional, Tuple, List, Any, Callable
import argparse
import numpy as np
import os
import flwr as fl
from flwr.common import Metrics, Scalar, Parameters
from sklearn.metrics import confusion_matrix
import functools
