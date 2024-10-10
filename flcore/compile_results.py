import sys
import yaml
import argparse
import os
import numpy as np
import pandas as pd
from flcore.report.generate_report import generate_report


def compile_results(experiment_dir: str):
    per_client_metrics = {}
    held_out_metrics = {}
    fit_metrics = {}

    config = yaml.safe_load(open(f"{experiment_dir}/config.yaml", "r"))

    csv_dict = {}
    if config['dataset'] == 'ukbb_cvd':
        center_names = ['Barts', 'Birmingham', 'Bristol', 'Bury', 'Cardiff', 'Croydon', 'Edinburgh', 'Glasgow', 'Hounslow', 'Leeds', 'Liverpool', 'Manchester', 'Middlesborough', 'Newcastle', 'Nottingham', 'Oxford', 'Reading', 'Sheffield', 'Stockport (pilot)', 'Stoke', 'Swansea', 'Wrexham']
        center_names[19], center_names[21] = center_names[21], center_names[19]

    elif config['dataset'] == 'kaggle_hf':
        center_names = ['Switzerland',  'Hungary', 'VA', 'Cleveland']

    writer = open(f"{experiment_dir}/metrics.txt", "w")

    writer.write(f"{'Experiment results':.^100} \n\n")
    writer.write(f"Name: {config['experiment']['name']}\n")
    writer.write(f"Model: {config['model']}\n")
    writer.write(f"Data: {config['dataset']}\n")
    writer.write(f"Dropout: {config['dropout_method']}\n")


    writer.write(f"Number of clients: {config['num_clients']}\n")

    # Check if the experiment is a single run or a kfold
    if "history.yaml" in os.listdir(experiment_dir):
        os.makedirs(os.path.join(experiment_dir, "run_0"), exist_ok=True)
        os.system(f"cp {experiment_dir}/* {os.path.join(experiment_dir, 'run_0')} 2>>/dev/null")
        os.makedirs(os.path.join(experiment_dir, "run_00"), exist_ok=True)
        os.system(f"cp {experiment_dir}/* {os.path.join(experiment_dir, 'run_00')} 2>>/dev/null")

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
                            if 'fit' in logs and not ('local' in metric or 'personalized' in metric):
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
                            if len(values_history) == 1:
                                if metric not in held_out_metrics:
                                    held_out_metrics[metric] = [values_history[0]]
                                else:
                                    held_out_metrics[metric].append(values_history[0])
                            else:
                                if metric not in held_out_metrics:
                                    held_out_metrics[metric] = [values_history[best_round]]
                                else:
                                    held_out_metrics[metric].append(values_history[best_round])
                        
                        elif 'fit' in logs:
                            if 'local' in metric or 'running_time' in metric:
                                continue
                            if 'training_time' in metric:
                                if metric not in fit_metrics:
                                    fit_metrics[metric] = np.array(values_history[-1])
                                else:
                                    fit_metrics[metric] = np.vstack((fit_metrics[metric], values_history[-1]))
                            else:
                                if metric not in fit_metrics:
                                    fit_metrics[metric] = np.array(values_history[best_round])
                                else:
                                    fit_metrics[metric] = np.vstack((fit_metrics[metric], values_history[best_round]))
                        
                        
    execution_stats = ['client_id', 'round_time [s]', 'n samples', 'training_time [s]']
    # Calculate mean and std for per client metrics
    writer.write(f"{'Evaluation':.^100} \n\n")
    writer.write(f"\n{'Test set:'} \n")

    val_section = False
    local_section = False
    personalized_section = False
    for metric in per_client_metrics:
        # if metric in execution_stats:
        #     continue
        if 'val' in metric:
            if not val_section:
                writer.write(f"\n{'Validation set:'} \n")
                val_section = True
    
        if 'local' in metric:
            if not local_section:
                writer.write(f"\n{'Non federated:'} \n")
                local_section = True
        
        if 'personalized' in metric:
            if not personalized_section:
                writer.write(f"\n{'Federated finetuned locally:'} \n")
                personalized_section = True

        # Calculate general mean and std
        mean = np.average(per_client_metrics[metric])
        # Calculate std of the average metric between experiment runs
        std = np.std(np.mean(per_client_metrics[metric], axis=1))
        per_client_mean = np.around(np.mean(per_client_metrics[metric], axis=0), 3)
        per_client_std = np.around(np.std(per_client_metrics[metric], axis=0), 3)
        if metric not in execution_stats:
            writer.write(f"{metric:<30}: {mean:<6.3f}  ±{std:<6.3f}  \t\t\t|| Per client {metric} {per_client_mean}  ({per_client_std})\n".replace("\n", "")+"\n")
        for i, _ in enumerate(per_client_mean):
            center = int(per_client_metrics['client_id'][0, i])
            center = center_names[center]
            if center not in csv_dict:
                csv_dict[center] = {}
            csv_dict[center][metric] = per_client_mean[i]
            csv_dict[center][metric+'_std'] = per_client_std[i]


    # print execution stats
    writer.write(f"\n{'Execution stats:'} \n")
    per_client_metrics.update(fit_metrics)
    for metric in execution_stats:
        mean = np.average(per_client_metrics[metric])
        std = np.std(np.mean(per_client_metrics[metric], axis=1))
        per_client_mean = np.around(np.mean(per_client_metrics[metric], axis=0), 3)
        per_client_std = np.around(np.std(per_client_metrics[metric], axis=0), 3)
        writer.write(f"{metric:<30}: {mean:<6.3f}  ±{std:<6.3f}  \t\t\t|| Per client {metric} {per_client_mean}  ({per_client_std})\n".replace("\n", "")+"\n")
        

    # Calculate mean and std for held out metrics
    #Extract centralized metrics from the held out dictionary 
    centralized_metrics = {}
    metrics = held_out_metrics.copy()
    for metric in metrics:
        if 'centralized' in metric:
            centralized_metrics[metric] = held_out_metrics[metric]
            held_out_metrics.pop(metric, None)

    writer.write(f"\n{'Held out set evaluation':.^100} \n\n")
    for metric in held_out_metrics:
        center = int(held_out_metrics['client_id'][0])
        center = center_names[center]+' (held out)'
        mean = np.average(held_out_metrics[metric])
        std = np.std(held_out_metrics[metric])

        writer.write(f"{metric:<30}: {mean:<6.3f}  ±{std:<6.3f}\n")
        if center not in csv_dict:
            csv_dict[center] = {}
        csv_dict[center][metric] = mean
        csv_dict[center][metric+'_std'] = std

    # Calculate mean and std for centralized metrics
    writer.write(f"\n{'Centralized evaluation':.^100} \n\n")
    for metric in centralized_metrics:
        mean = np.average(centralized_metrics[metric])
        std = np.std(centralized_metrics[metric])
        writer.write(f"{metric:<30}: {mean:<6.3f}  ±{std:<6.3f}\n")

    writer.close()


    # Create dataframe from dict
    df = pd.DataFrame(csv_dict)
    df = df.T
    df = df.rename(columns={"index": "center"})

    # Write to csv
    df.to_csv(f"{experiment_dir}/per_center_results.csv", index=True)

    generate_report(experiment_dir)


if __name__ == "__main__":

    if len(sys.argv) == 2:
            config_path = sys.argv[1]

    parser = argparse.ArgumentParser(description="Compile kfold training results")
    parser.add_argument("experiment_dir", type=str, help="Experiment directory")

    args = parser.parse_args()
    experiment_dir = args.experiment_dir

    compile_results(experiment_dir)
