# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

import pickle
from flcore.models.gbs.base_aggregator import BaseAggregator

class GBSAggregator(BaseAggregator):
    """
    Aggregator for Gradient Boosting Survival models (e.g., FPBoost).
    Each client sends a serialized model (pickled FPBoost model).
    Aggregation concatenates all weak learners (stages) from all clients.
    """

    def aggregate(self):
        """
        Combine boosting stages from all clients into a single model.
        """
        aggregated_stages = []

        for client_params in self.models:
            try:
                # Each client sends [serialized_model]
                serialized_model = client_params[0]
                client_model = pickle.loads(serialized_model)

                # Each FPBoost model has .stages_ (list of weak learners)
                if hasattr(client_model, "stages_"):
                    aggregated_stages.extend(client_model.stages_)
                else:
                    print("[GBSAggregator] Warning: client model has no stages_ attribute")

            except Exception as e:
                print(f"[GBSAggregator] Error while loading client model: {e}")

        # Reconstruct a new model by cloning structure of one client
        # (same base learner, loss, learning rate, etc.)
        base_client = pickle.loads(self.models[0][0])
        aggregated_model = base_client
        aggregated_model.stages_ = aggregated_stages

        # Optionally: adjust n_estimators_
        aggregated_model.n_estimators_ = len(aggregated_stages)

        # Serialize the final aggregated model to return
        try:
            serialized_aggregated = pickle.dumps(aggregated_model)
            return [serialized_aggregated]
        except Exception as e:
            print(f"[GBSAggregator] Serialization error: {e}")
            return []
