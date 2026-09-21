"""One true end-to-end smoke test: spawns real server_cmd.py + client_cmd.py
subprocesses talking over a real localhost socket (--production_mode False, no
certs), mirroring one command pair from pruebas.md, against the synthetic
fixture. Complements test_model_smoke.py's in-process tests with a real
process/socket check that the CLI entry points themselves still work end to end.
"""
import subprocess
import sys
import time
from pathlib import Path

from fixtures.synthetic_dt4h import make_dt4h_fixture

REPO_ROOT = Path(__file__).parent.parent


def _wait_until_ready_or_dead(proc, timeout=20):
    """No clean readiness signal is exposed by server_cmd.py, so poll briefly and
    bail out early (instead of a blind fixed sleep) if the server already died."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, _ = proc.communicate()
            raise RuntimeError(f"server_cmd.py exited early with code {proc.returncode}:\n{out}")
        time.sleep(0.5)


def test_random_forest_e2e_local(tmp_path):
    data_info = make_dt4h_fixture(tmp_path / "data", task="classification", n_rows=120)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    experiment_name = "smoke_e2e"
    port = "8099"

    server_cmd = [
        sys.executable,
        str(REPO_ROOT / "server_cmd.py"),
        "--model", "random_forest",
        "--task", "classification",
        "--num_rounds", "2",
        "--num_clients", "1",
        "--n_estimators", "5",
        "--max_depth", "2",
        "--sandbox_path", str(sandbox),
        "--experiment_name", experiment_name,
        "--local_port", port,
        "--production_mode", "False",
        "--dropout_method", "None",
        # server_cmd.py's --balanced defaults to None, which fails CheckServerConfig's
        # isinstance(..., str) check -- see CLAUDE.md Sec 4.2 (server/client default
        # mismatch); pass it explicitly until Phase 1 fixes the default.
        "--balanced", "True",
    ]
    client_cmd = [
        sys.executable,
        str(REPO_ROOT / "client_cmd.py"),
        "--model", "random_forest",
        "--task", "classification",
        "--train_labels", *data_info["train_labels"],
        "--target_labels", *data_info["target_labels"],
        "--sandbox_path", str(sandbox),
        "--experiment_name", experiment_name,
        "--local_port", port,
        "--production_mode", "False",
        "--data_id", data_info["data_dir"],
    ]

    server = subprocess.Popen(
        server_cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        _wait_until_ready_or_dead(server)
        client = subprocess.run(client_cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
        server_out, _ = server.communicate(timeout=60)
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)

    assert client.returncode == 0, f"client stdout/err:\n{client.stdout}\n{client.stderr}"
    history_path = sandbox / experiment_name / "history.yaml"
    assert history_path.exists(), f"server output:\n{server_out}"
