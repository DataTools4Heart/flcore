"""Drives a real Flower Strategy against real Flower Clients, in one process.

No gRPC, no certs, no subprocesses: this wraps each flcore model client in an
in-process ClientProxy and registers it with a real SimpleClientManager, then
manually replays the same configure_fit -> client.fit -> aggregate_fit ->
configure_evaluate -> client.evaluate -> aggregate_evaluate loop the real Flower
`Server` runs -- so it exercises the *real* Strategy/Client code
(flcore/models/<name>/{server,client,FedCustomAggregator}.py), not a mock.

This is the harness Phase 3 (aggregation-strategy homogenization) will use to diff
before/after outputs per model against tests/golden/.
"""
from typing import List

import flwr as fl
from flwr.common import DisconnectRes, GetParametersIns
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy


class InProcessClientProxy(ClientProxy):
    """Adapts a real fl.client.Client to the ClientProxy interface without a
    network hop, so a real Strategy can drive it via client_manager.sample()."""

    def __init__(self, cid: str, client: "fl.client.Client"):
        super().__init__(cid=cid)
        self._client = client

    def get_properties(self, ins, timeout=None, group_id=None):
        return self._client.get_properties(ins)

    def get_parameters(self, ins, timeout=None, group_id=None):
        return self._client.get_parameters(ins)

    def fit(self, ins, timeout=None, group_id=None):
        return self._client.fit(ins)

    def evaluate(self, ins, timeout=None, group_id=None):
        return self._client.evaluate(ins)

    def reconnect(self, ins, timeout=None, group_id=None):
        return DisconnectRes(reason="")


def to_base_client(client) -> "fl.client.Client":
    """flcore model clients are either fl.client.Client (random_forest,
    weighted_random_forest) or fl.client.NumPyClient (everything else) -- unify
    to the base Client interface the same way client_cmd.py picks between
    start_client/start_numpy_client (isinstance check), instead of guessing."""
    if isinstance(client, fl.client.NumPyClient):
        return client.to_client()
    return client


def run_federated_rounds(strategy, clients: list, num_rounds: int) -> dict:
    """Registers `clients` (already-constructed flcore model clients) into a real
    SimpleClientManager and drives `num_rounds` of the strategy's real
    configure_fit/aggregate_fit/configure_evaluate/aggregate_evaluate.

    Returns {"parameters": final aggregated Parameters, "rounds": [per-round dict]}.
    Raises whatever the model's client/strategy code raises (fail loud -- a smoke
    test's job is to surface exactly that).
    """
    client_manager = SimpleClientManager()
    proxies: List[ClientProxy] = []
    for i, client in enumerate(clients):
        proxy = InProcessClientProxy(cid=str(i), client=to_base_client(client))
        client_manager.register(proxy)
        proxies.append(proxy)

    # Mirror real Flower's Server._get_initial_parameters: ask the strategy first
    # (xgb's FedXgbFullyFederated overrides initialize_parameters to seed an empty
    # model), and only fall back to asking a client when the strategy returns None
    # (true for every other flcore strategy here, since none of them override it).
    parameters = strategy.initialize_parameters(client_manager)
    if parameters is None:
        parameters = proxies[0].get_parameters(GetParametersIns(config={})).parameters

    rounds = []
    for server_round in range(1, num_rounds + 1):
        fit_instructions = strategy.configure_fit(server_round, parameters, client_manager)
        fit_results, fit_failures = [], []
        for proxy, fit_ins in fit_instructions:
            try:
                fit_results.append((proxy, proxy.fit(fit_ins)))
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, re-raised below
                fit_failures.append(exc)
        if fit_failures and not fit_results:
            # Nothing to aggregate and no silent partial-failure masking -- surface it.
            raise fit_failures[0]

        parameters, fit_metrics = strategy.aggregate_fit(server_round, fit_results, fit_failures)

        eval_instructions = strategy.configure_evaluate(server_round, parameters, client_manager)
        eval_results, eval_failures = [], []
        for proxy, eval_ins in eval_instructions:
            try:
                eval_results.append((proxy, proxy.evaluate(eval_ins)))
            except Exception as exc:  # noqa: BLE001
                eval_failures.append(exc)
        if eval_failures and not eval_results:
            raise eval_failures[0]

        loss, eval_metrics = strategy.aggregate_evaluate(server_round, eval_results, eval_failures)

        rounds.append(
            {
                "round": server_round,
                "fit_metrics": fit_metrics,
                "loss": loss,
                "eval_metrics": eval_metrics,
                "fit_failures": [repr(e) for e in fit_failures],
                "eval_failures": [repr(e) for e in eval_failures],
            }
        )

    return {"parameters": parameters, "rounds": rounds}
