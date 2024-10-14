import subprocess
import time
import sys
import os

import yaml

if len(sys.argv) == 2:
    config_path = sys.argv[1]
else:
    config_path = "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

try:
    print("Starting server")
    server_process = subprocess.Popen(f"python server.py {config_path}", shell=True, stderr=subprocess.PIPE, text=True)
    # server_process = subprocess.Popen(f"python server.py {config_path}", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # print(server_process.stdout)
    # Break when "ready" is printed
    for line in server_process.stderr:
        print(line, end='') # process line here
        if "Requesting initial parameters" in line:
            break
    

    client_processes = []
    for i in range(0, config["num_clients"]):
        # i = i + 10
        print("Starting client " + str(i))
        client_processes.append(
            subprocess.Popen(f"python client.py {i} {config_path}", shell=True)
        )
    
    for line in server_process.stderr:
        print(line, end='')

    server_process.wait()

except KeyboardInterrupt:
    server_process.terminate()
    server_process.wait()
    for client_process in client_processes:
        client_process.terminate()
        client_process.wait()
        
    print("Server and clients stopped")
