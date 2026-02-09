import subprocess
import time
import os
import sys

import yaml

if len(sys.argv) == 2:
    config_path = sys.argv[1]
else:
    config_path = "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

repetitions = 5
experiment_name = config['experiment']['name']

config['experiment']['log_path'] = os.path.join(config['experiment']['log_path'], config['experiment']['name'])
os.makedirs(config['experiment']['log_path'], exist_ok=True)

start_time = time.time()
for i in range(repetitions):
    print(f"Experiment run {i + 1}")
    config['experiment']['name'] = 'run_' + str(i + 1)
    config['seed'] = i + 10
    config_path = os.path.join(config['experiment']['log_path'], "config.yaml")
    log_file_path = os.path.join(config['experiment']['log_path'], config['experiment']['name'], "run_log.txt")
    os.makedirs(os.path.join(config['experiment']['log_path'], config['experiment']['name']), exist_ok=True)

    # Kill any existing process using the same port
    if 'local_port' in config:
        kill_command = f"lsof -ti tcp:{config['local_port']} | xargs kill -9"
        subprocess.run(kill_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(config_path, "w") as f:
        yaml.dump(config, f)    
    try:
        run_process = subprocess.Popen(f"python run.py {config_path} | tee {log_file_path}", shell=True)
        run_process.wait()

    except KeyboardInterrupt:
        run_process.terminate()
        run_process.wait()

config['experiment']['name'] = experiment_name
with open(config_path, "w") as f:
    yaml.dump(config, f)


# processes = []
# try:
#     for i in range(repetitions):
#         print(f"Experiment run {i + 1}")
#         config['experiment']['name'] = 'run_' + str(i + 1)
#         config['seed'] = i + 10
#         config['local_port'] = 8081 + i
#         config_path = os.path.join(config['experiment']['log_path'], config['experiment']['name'], "config.yaml")
#         log_file_path = os.path.join(config['experiment']['log_path'], config['experiment']['name'], "run_log.txt")
#         os.makedirs(os.path.join(config['experiment']['log_path'], config['experiment']['name']), exist_ok=True)
#         with open(config_path, "w") as f:
#             yaml.dump(config, f)    
#             run_process = subprocess.Popen(f"python run.py {config_path} | tee {log_file_path}", shell=True)
#             # run_process.wait()
#             processes.append(run_process)
    
#     for run_process in processes:
#         run_process.wait()

# except KeyboardInterrupt:
#     run_process.terminate()
#     run_process.wait()

# config['experiment']['name'] = experiment_name
# config_path = os.path.join(config['experiment']['log_path'], "config.yaml")
# with open(config_path, "w") as f:
#     yaml.dump(config, f)

run_process = subprocess.Popen(f"python flcore/compile_results.py {config['experiment']['log_path']}", shell=True)
run_process.wait()
            
print("Batch experiments finished")
print(f"Total time: {(time.time() - start_time) / 60} minutes")
