import subprocess
import time
import os

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

repetitions = 8
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

run_process = subprocess.Popen(f"python flcore/compile_results.py {config['experiment']['log_path']}", shell=True)
run_process.wait()
            
print("Batch experiments finished")
print(f"Total time: {(time.time() - start_time) / 60} minutes")
