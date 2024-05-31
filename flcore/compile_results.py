import sys
import yaml

if len(sys.argv) == 2:
        config_path = sys.argv[1]

import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Compile kfold training results")
parser.add_argument("experiment_dir", type=str, help="Experiment directory")

args = parser.parse_args()
experiment_dir = args.experiment_dir

per_client_metrics = {}
held_out_metrics = {}

config = yaml.safe_load(open(f"{experiment_dir}/config.yaml", "r"))

csv_dict = {}
center_names = ['Barts', 'Birmingham', 'Bristol', 'Bury', 'Cardiff', 'Croydon', 'Edinburgh', 'Glasgow', 'Hounslow', 'Leeds', 'Liverpool', 'Manchester', 'Middlesborough', 'Newcastle', 'Nottingham', 'Oxford', 'Reading', 'Sheffield', 'Stockport (pilot)', 'Stoke', 'Swansea', 'Wrexham']

writer = open(f"{experiment_dir}/metrics.txt", "w")

writer.write(f"{'Experiment results':.^100} \n\n")
writer.write(f"Name: {config['experiment']['name']}\n")
writer.write(f"Model: {config['model']}\n")
writer.write(f"Data: {config['dataset']}\n")
writer.write(f"Dropout: {config['dropout_method']}\n")


writer.write(f"Number of clients: {config['num_clients']}\n")

for directory in os.listdir(experiment_dir):
    if directory.startswith("fold_") or directory.startswith("run_"):
        fold_dir = os.path.join(experiment_dir, directory)
        # Read history.yaml
        history = yaml.safe_load(open(os.path.join(fold_dir, "history.yaml"), "r"))
        
        selection_metric = 'val '+ config['checkpoint_selection_metric']
        best_round= int(np.argmax(history['metrics_distributed'][selection_metric]))
        # client_order = history['metrics_distributed']['per client client_id'][best_round]
        client_order = history['metrics_distributed']['per client n samples'][best_round]
        for logs in history.keys():
            if isinstance(history[logs], dict):
                for metric in history[logs]:
                    values_history = history[logs][metric]
                    if isinstance(values_history[0], list):
                        if 'fit' in logs and 'local' not in metric:
                            continue
                        if 'local' in metric:
                            values = values_history[0]
                        else:
                            values = values_history[best_round]
                        # sort by key client_id in the metrics dict
                        ids, values = zip(*sorted(zip(client_order, values), key=lambda x: x[0]))
                        metric = metric.replace("per client ", "")
                        
                        if metric not in per_client_metrics:
                            per_client_metrics[metric] = np.array(values)
                        else:
                            per_client_metrics[metric] = np.vstack((per_client_metrics[metric], values))
                        
                    elif 'centralized' in logs:
                        if metric not in held_out_metrics:
                            held_out_metrics[metric] = [values_history[best_round]]
                        else:
                            held_out_metrics[metric].append(values_history[best_round])
                    
execution_stats = ['client_id', 'round_time [s]', 'n samples']
# Calculate mean and std for per client metrics
writer.write(f"{'Evaluation':.^100} \n\n")
val_section = False
local_section = False
for metric in per_client_metrics:
    # if metric in execution_stats:
    #     continue
    if 'val' in metric:
        if not val_section:
            writer.write(f"\n{'Validation set:'} \n")
            val_section = True
   
    if 'local' in metric:
        if not local_section:
            writer.write(f"\n{'Local evaluation:'} \n")
            local_section = True

    # Calculate general mean and std
    mean = np.average(per_client_metrics[metric])
    std = np.std(per_client_metrics[metric])
    per_client_mean = np.around(np.mean(per_client_metrics[metric], axis=0), 3)
    per_client_std = np.around(np.std(per_client_metrics[metric], axis=0), 3)
    if metric not in execution_stats:
        writer.write(f"{metric:<22}: {mean:<6.3f}  ±{std:<6.3f}  \t\t\t|| Per client {metric} {per_client_mean}  ({per_client_std})\n")
    for i, _ in enumerate(per_client_mean):
        center = int(per_client_metrics['client_id'][0, i])
        center = center_names[center]
        if center not in csv_dict:
            csv_dict[center] = {}
        csv_dict[center][metric] = per_client_mean[i]
        csv_dict[center][metric+'_std'] = per_client_std[i]


# print execution stats
writer.write(f"\n{'Execution stats:'} \n")
for metric in execution_stats:
    mean = np.average(per_client_metrics[metric])
    std = np.std(per_client_metrics[metric])
    per_client_mean = np.around(np.mean(per_client_metrics[metric], axis=0), 5)
    per_client_std = np.around(np.std(per_client_metrics[metric], axis=0), 3)
    writer.write(f"{metric:<20}: {mean:<6.5f}  ±{std:<6.5f}  \t\t\t|| Per client {metric} {per_client_mean}  ({per_client_std})\n")
    

# Calculate mean and std for held out metrics
writer.write(f"\n{'Held out set evaluation':.^100} \n\n")
for metric in held_out_metrics:
    center = int(held_out_metrics['client_id'][0])
    center = center_names[center]+' (held out)'
    mean = np.average(held_out_metrics[metric])
    std = np.std(held_out_metrics[metric])

    writer.write(f"{metric:<20}: {mean:<6.3f}  ±{std:<6.3f}\n")
    if center not in csv_dict:
        csv_dict[center] = {}
    csv_dict[center][metric] = mean
    csv_dict[center][metric+'_std'] = std

writer.close()


# Create dataframe from dict
df = pd.DataFrame(csv_dict)
df = df.T
df = df.rename(columns={"index": "center"})

# Write to csv
df.to_csv(f"{experiment_dir}/per_center_results.csv", index=True)
