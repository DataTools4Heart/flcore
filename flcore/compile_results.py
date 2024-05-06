import sys
import yaml

if len(sys.argv) == 2:
        config_path = sys.argv[1]

import argparse
import os
import pandas as pd
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Compile kfold training results")
parser.add_argument("experiment_dir", type=str, help="Experiment directory")

args = parser.parse_args()
experiment_dir = args.experiment_dir

per_client_metrics = {}
held_out_metrics = {}

for directory in os.listdir(experiment_dir):
    if directory.startswith("fold_") or directory.startswith("run_"):
        fold_dir = os.path.join(experiment_dir, directory)
        # Read history.yaml
        history = yaml.safe_load(open(os.path.join(fold_dir, "history.yaml"), "r"))
        
        client_order = history['metrics_distributed']['per client client_id'][-1]
        for logs in history.keys():
            if isinstance(history[logs], dict):
                for metric in history[logs]:
                    values_history = history[logs][metric]
                    if isinstance(values_history[0], list):
                        values = values_history[-1]
                        # sort by key client_id in the metrics dict
                        ids, values = zip(*sorted(zip(client_order, values), key=lambda x: x[0]))
                        metric = metric.replace("per client ", "")
                        if metric not in per_client_metrics:
                            per_client_metrics[metric] = np.array(values)
                        else:
                            per_client_metrics[metric] = np.vstack((per_client_metrics[metric], values))
                        
                    elif 'centralized' in logs:
                        if metric not in held_out_metrics:
                            held_out_metrics[metric] = [values_history[-1]]
                        else:
                            held_out_metrics[metric].append(values_history[-1])
                    
execution_stats = ['client_id', 'round_time [s]', 'n samples']
# Calculate mean and std for per client metrics
print(f"{'Experiment results':.^100} \n")
for metric in per_client_metrics:
    if metric in execution_stats:
        continue
    # Calculate general mean and std
    mean = np.average(per_client_metrics[metric])
    std = np.std(per_client_metrics[metric])
    per_client_mean = np.around(np.mean(per_client_metrics[metric], axis=0), 3)
    per_client_std = np.around(np.std(per_client_metrics[metric], axis=0), 3)
    print(f"{metric:<20}: {mean:<6.3f}  ±{std:<6.3f}  \t\t\t|| Per client {metric} {per_client_mean}  ({per_client_std})")

# print execution stats
print(f"\n{'Execution stats:'} \n")
for metric in execution_stats:
    mean = np.average(per_client_metrics[metric])
    std = np.std(per_client_metrics[metric])
    per_client_mean = np.around(np.mean(per_client_metrics[metric], axis=0), 5)
    per_client_std = np.around(np.std(per_client_metrics[metric], axis=0), 3)
    print(f"{metric:<20}: {mean:<6.5f}  ±{std:<6.5f}  \t\t\t|| Per client {metric} {per_client_mean}  ({per_client_std})")
    

# Calculate mean and std for held out metrics
print(f"\n{'Held out set evaluation':.^100} \n")
for metric in held_out_metrics:
    mean = np.average(held_out_metrics[metric])
    std = np.std(held_out_metrics[metric])

    print(f"{metric:<20}: {mean:<6.3f}  ±{std:<6.3f}")
