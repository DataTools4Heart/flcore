import subprocess
import time
import yaml


with open( 'config.yaml', 'r' ) as f:
    config = yaml.safe_load( f )

try:
    print("Starting server")
    server_process = subprocess.Popen("python server.py", shell=True)
    time.sleep(3)

    client_processes = []
    for i in range(1, config['num_clients']+1):
        print("Starting client " + str(i))
        client_processes.append(subprocess.Popen("python client.py " + str(i), shell=True))

    server_process.wait()

except KeyboardInterrupt:
    server_process.terminate()
    server_process.wait()
    for client_process in client_processes:
        client_process.terminate()
        client_process.wait()
