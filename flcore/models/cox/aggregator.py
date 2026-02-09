# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from typing import List
import numpy as np
from flcore.models.cox.base_aggregator import BaseAggregator

# --- CoxPH Aggregator ---

class CoxAggregator(BaseAggregator):
    """
    Aggregates CoxPH model parameters using Federated Averaging (FedAvg).
    
    The parameters for this model are expected to be a list containing a 
    single numpy array: [beta_coefficients].
    """

    def aggregate(self) -> List[np.ndarray]:
        """
        Performs a weighted average of the beta coefficients from all clients.
        
        Returns:
            List[np.ndarray]: The aggregated parameters in the same format
                              expected by the model's set_parameters method.
        """
        
        # 1. Filter out any clients that might have failed (returned empty params)
        # and extract the beta array (the first element) from each.
        valid_params_and_weights = []
        for params_list, weight in zip(self.models, self.weights):
            if params_list:  # Check if the list is not empty
                valid_params_and_weights.append((params_list[0], weight))
        
        if not valid_params_and_weights:
            print("Warning: No valid model parameters to aggregate. Returning empty list.")
            return []

        # 2. Initialize aggregated parameters and total weight
        # Use the shape of the first client's beta array
        first_beta, first_weight = valid_params_and_weights[0]
        aggregated_beta = np.zeros_like(first_beta, dtype=np.float64)
        total_weight = 0.0

        # 3. Perform the weighted average
        for beta, weight in valid_params_and_weights:
            # Ensure shapes match before aggregating
            if beta.shape != aggregated_beta.shape:
                print(f"Warning: Skipping model with mismatched shape. "
                      f"Expected {aggregated_beta.shape}, got {beta.shape}.")
                continue
                
            aggregated_beta += beta * weight
            total_weight += weight

        # 4. Normalize the aggregated parameters
        if total_weight > 0:
            aggregated_beta /= total_weight
        else:
            print("Warning: Total weight is zero. Aggregation resulted in zeros.")
            # aggregated_beta is already all zeros, which is the best we can do.
            pass

        # 5. Return in the same format: List[np.ndarray]
        return [aggregated_beta]