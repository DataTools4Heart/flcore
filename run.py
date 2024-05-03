import subprocess
import time
import sys

import yaml

if len(sys.argv) == 2:
    config_path = sys.argv[1]
else:
    config_path = "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

try:
    print("Starting server")
    server_process = subprocess.Popen(f"python server.py {config_path}", shell=True)
    time.sleep(20)

    client_processes = []
    for i in range(1, config["num_clients"] + 1):
        print("Starting client " + str(i))
        client_processes.append(
            subprocess.Popen(f"python client.py {i} {config_path}", shell=True)
        )

    server_process.wait()

except KeyboardInterrupt:
    server_process.terminate()
    server_process.wait()
    for client_process in client_processes:
        client_process.terminate()
        client_process.wait()
        
    print("Server and clients stopped")
