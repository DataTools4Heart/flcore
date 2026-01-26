# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *
# Survival model
# Author: Iratxe Moya
# Date: January 2026
# Project: AI4HF
# ********* * * * * *  *  *   *   *    *   *  *  *  * * * * *

from flcore.models.rsf.base_aggregator import BaseAggregator

class RSFAggregator(BaseAggregator):
    """
    Aggregator for RandomSurvivalForest models in federated learning.
    Stores all client trees but does NOT assume shared event_times_.
    """
    def aggregate(self):
        aggregated_trees = []
        metadata = None

        for client_params in self.models:
            if not client_params:
                continue

            # Append trees from this client
            trees = client_params[:-1]
            aggregated_trees.extend(trees)

            # Take metadata from the first client as representative
            if metadata is None:
                metadata = client_params[-1]

        # The aggregated model just stores all trees; event_times_ will be
        # handled on the client side during evaluation using interpolation.
        aggregated = aggregated_trees + ([metadata] if metadata is not None else [])
        print(f"[RSFAggregator] Aggregated {len(aggregated_trees)} trees from {len(self.models)} clients.")
        return aggregated