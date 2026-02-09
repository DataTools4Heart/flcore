# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from typing import Union, Dict
import numpy as np

def get_numpy(X_train, y_train, X_test, y_test, duration_col, event_col) -> Dict[str, Union[np.ndarray, str, int]]:
    """Return data as numpy/Pandas objects for classical survival models."""
    return {
        "X": X_train,
        "y": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "duration_col": duration_col,
        "event_col": event_col,
        "num_examples": len(X_train),
    }