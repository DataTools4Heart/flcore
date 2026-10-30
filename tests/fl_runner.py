"""Run a complete local federation (1 server + N clients) as subprocesses.

Mirrors what ``run.py`` does, but takes its parameters as arguments instead
of a YAML file so tests can use isolated sandboxes and free ports.
"""

import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

TRAIN_LABELS = [
    "patient_demographics_gender",
    "encounters_encounterClass",
    "patient_demographics_age",
    "encounters_lengthOfStay",
    "vital_signs_weight_value_p6mo_min",
    "med_admins_history_diuretics_loop_any",
    "patient_flags_smoker",
]


@dataclass
class FederationResult:
    ok: bool
    reason: str
    sandbox: Path
    server_log: str = ""
    client_logs: list = field(default_factory=list)
    history: dict = field(default_factory=dict)
    # True for failures of the local networking (port collision), not of the model
    flaky: bool = False

    def describe(self, tail=40):
        def last(text):
            return "\n".join(text.strip().splitlines()[-tail:])

        parts = [self.reason, "--- server (tail) ---", last(self.server_log)]
        for i, log in enumerate(self.client_logs):
            parts += [f"--- client {i} (tail) ---", last(log)]
        return "\n".join(parts)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _opt(name, value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [f"--{name}", *[str(v) for v in value]] if value else []
    return [f"--{name}", str(value)]


def _wait_for_port(port, server, timeout):
    """True once the server accepts connections on 127.0.0.1 (the clients' route)."""
    start = time.time()
    while time.time() - start < timeout:
        if server.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _read(path):
    return path.read_text(errors="replace") if path.exists() else ""


def _kill(processes):
    for p in processes:
        if p.poll() is None:
            p.kill()
    for p in processes:
        p.wait()


def _attempt(
    *,
    model,
    task,
    data_path,
    sandbox,
    target_labels,
    train_labels=TRAIN_LABELS,
    num_clients=2,
    num_rounds=2,
    extra_args=(),
    client_extra_args=(),
    server_extra_args=(),
    timeout=300,
    seed=42,
):
    """Train ``model`` for ``num_rounds`` with ``num_clients`` clients.

    ``extra_args`` go to both server and clients, ``client_extra_args`` only to
    the clients and ``server_extra_args`` only to the server. Never raises on a failed training: returns a
    :class:`FederationResult` so the caller decides what to assert.
    """
    sandbox = Path(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    # The deployed /sandbox volume ships with these directories, and some
    # servers (e.g. the linear-model aggregator) do not create them on their own.
    for sub in ("checkpoints", "models"):
        (sandbox / sub).mkdir(exist_ok=True)
    port = free_port()

    common = [
        *_opt("model", model),
        *_opt("task", task),
        *_opt("num_rounds", num_rounds),
        *_opt("num_clients", num_clients),
        *_opt("seed", seed),
        *_opt("local_port", port),
        *_opt("production_mode", False),
        *_opt("sandbox_path", sandbox),
        # run.py always forwards these two (random forest requires --balanced)
        *_opt("balanced", True),
        *_opt("levelOfDetail", "DecisionTree"),
        *extra_args,
    ]
    server_cmd = [sys.executable, "server_cmd.py", *common, *server_extra_args]
    client_cmd = [
        sys.executable,
        "client_cmd.py",
        *common,
        *_opt("dataset", "dt4h_format"),
        *_opt("data_path", data_path),
        *_opt("data_id", data_path),
        *_opt("normalization_method", "IQR"),
        *_opt("train_labels", train_labels),
        *_opt("target_labels", target_labels),
        *_opt("train_size", 0.7),
        *client_extra_args,
    ]

    logs = sandbox / "logs"
    logs.mkdir(exist_ok=True)
    server_log = logs / "server.log"
    client_logs = [logs / f"client_{i}.log" for i in range(num_clients)]

    processes = []

    def result(ok, reason, flaky=False):
        _kill(processes)
        history_file = sandbox / "history.yaml"
        history = {}
        if history_file.exists():
            history = yaml.safe_load(history_file.read_text()) or {}
        return FederationResult(
            ok=ok,
            reason=reason,
            sandbox=sandbox,
            server_log=_read(server_log),
            client_logs=[_read(p) for p in client_logs],
            history=history,
            flaky=flaky,
        )

    with open(server_log, "w") as sf:
        server = subprocess.Popen(
            server_cmd, cwd=ROOT, stdout=sf, stderr=subprocess.STDOUT
        )
        processes.append(server)
        if not _wait_for_port(port, server, timeout=90):
            # Still running but unreachable: a socket problem worth one retry.
            # An exited server crashed on its own, which is a real failure.
            return result(
                False, "server did not start listening", flaky=server.poll() is None
            )

        client_files = [open(p, "w") for p in client_logs]
        try:
            clients = []
            for f in client_files:
                clients.append(
                    subprocess.Popen(
                        client_cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
                    )
                )
            processes.extend(clients)

            start = time.time()
            clients_done_at = None
            while server.poll() is None:
                if time.time() - start > timeout:
                    return result(False, f"timed out after {timeout}s")
                if clients_done_at is None and all(c.poll() == 0 for c in clients):
                    clients_done_at = time.time()
                if clients_done_at and time.time() - clients_done_at > 30:
                    # Clients finished cleanly but the server never terminated.
                    return result(False, "server did not exit after training")
                time.sleep(0.5)
            # Clients exit right after a healthy run; if the server crashed they
            # may linger retrying, so only wait for them briefly in that case.
            grace = 60 if server.returncode == 0 else 5
            for c in clients:
                try:
                    c.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    c.kill()
                    c.wait()
        finally:
            for f in client_files:
                f.close()

        codes = {"server": server.returncode}
        codes.update({f"client {i}": c.returncode for i, c in enumerate(clients)})
        failed = {k: v for k, v in codes.items() if v != 0}
        if failed:
            return result(False, f"non-zero exit codes: {failed}")
    return result(True, "ok")


def run_federation(**kwargs):
    """:func:`_attempt` with one retry when the local sockets misbehaved."""
    res = _attempt(**kwargs)
    unreachable = "failed to connect to all addresses" in "".join(res.client_logs)
    if not res.ok and (res.flaky or unreachable):
        res = _attempt(**kwargs)
    return res
