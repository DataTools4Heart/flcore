import subprocess
import time
import os
import yaml
import sys
from itertools import product

experiment_name = "experiment_all_10percent"
benchmark_dir = "benchmark_results"


model_names = [
   "logistic_regression",
   "elastic_net",
   "lsvc",
    "random_forest",
    "balanced_random_forest",
    # # "weighted_random_forest",
    "xgb"
    ]

datasets = [
    # "kaggle_hf",
    "diabetes",
    # "ukbb_cvd",
    # "cvd"
    ]

num_clients = [
    3,
    5,
    10,
    20
]

dirichlet_alpha = [
    None,
    # 1.0,
    # 0.7
]

data_normalization = ["global"]
n_features = [None]

# Normalization experiment
# experiment_name = "normalization"
# benchmark_dir = "benchmark_results_normalization"
# model_names = ["logistic_regression"]
# datasets = ["diabetes", "ukbb_cvd"]
# num_clients = [10]
# dirichlet_alpha = [0.7, None]
# data_normalization = ["global", "local", None]

# Feature selection experiment
experiment_name = "feature_selection"
benchmark_dir = "benchmark_results_feature_selection"
model_names = ["balanced_random_forest"]
datasets = ["ukbb_cvd"]
num_clients = [5,10]
dirichlet_alpha = [0.7, None]
data_normalization = ["global"]
n_features = [10, 20, 35, 40, None]

os.makedirs(benchmark_dir, exist_ok=True)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


config_path = os.path.join(benchmark_dir, "config.yaml")
log_file_path = os.path.join(benchmark_dir, "run_log.txt")

with open(config_path, "w") as f:
    yaml.dump(config, f)

config['data_path'] = 'dataset/'
config['experiment']['log_path'] = benchmark_dir

start_time = time.time()

# Flatten the nested loops into a single iterator
parameters = product(datasets, num_clients, dirichlet_alpha, model_names, data_normalization, n_features)

try:
    for ds_name, n_client, alpha, m_name, norm, n_feat in parameters:
        print(f"Running benchmark: {ds_name}, {m_name}, clients: {n_client}, alpha: {alpha}, normalization: {norm}, features: {n_feat}")
        
        # Update config dictionary
        config.update({
            'model': m_name,
            'dataset': ds_name,
            'num_clients': n_client,
            'dirichlet_alpha': alpha,
            'data_normalization': norm,
            'n_features': n_feat
        })
        if "forest" in m_name:
            config['num_rounds'] = 1  # Set number of jobs for parallel processing

        config['experiment']['name'] = f"{experiment_name}_{ds_name}_{m_name}_c{n_client}_a{alpha}_norm{norm}_feat{n_feat}"

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # subprocess.run is cleaner for synchronous execution
        # Use a list for the command to avoid shell=True security/cleanup issues
        cmd = f"python repeated.py {config_path} | tee {log_file_path}"
        subprocess.run(cmd, shell=True, check=True)

except KeyboardInterrupt:
    print("\nBenchmark interrupted by user. Exiting...")
    sys.exit(1)



# # Run benchmark experiments
# # Iterate over datasets and models
# for dataset_name in datasets:
#     for num_client in num_clients:
#         for alpha in dirichlet_alpha:
#             for model_name in model_names:
#                 print(f"Running benchmark for dataset: {dataset_name}, model: {model_name}")
#                 config['experiment']['name'] = f"{experiment_name}_{dataset_name}_{model_name}_clients_{num_client}_alpha_{alpha}"
#                 config['model'] = model_name
#                 config['dataset'] = dataset_name
#                 config['num_clients'] = num_client
#                 config['dirichlet_alpha'] = alpha

#                 with open(config_path, "w") as f:
#                     yaml.dump(config, f)

#                 try:
#                     run_process = subprocess.Popen(f"python repeated.py {config_path} | tee {log_file_path}", shell=True)
#                     run_process.wait()

#                 except KeyboardInterrupt:
#                     run_process.terminate()
#                     run_process.wait()
#                     break

total_time = time.time() - start_time
print("Benchmark experiments finished in", total_time/60, " minutes")
