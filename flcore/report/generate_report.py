import os
import pandas as pd
import numpy as np
import jinja2
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import pdfkit


def generate_report(experiment_path: str):

    config = yaml.safe_load(open(f'{experiment_path}/config.yaml', "r"))

    results_file = f'{experiment_path}/per_center_results.csv'


    fig_dir = f'{experiment_path}/images'
    # Create the directory if it does not exist
    os.makedirs(fig_dir, exist_ok=True)

    # Copy images dir to the experiment dir
    os.system(f'cp -r flcore/report/images/ {fig_dir}')

    # Read the CSV file
    df = pd.read_csv(results_file)
    df = df.rename(columns={"Unnamed: 0": "center"})
    # Convert metrics columns to 2 decimal places
    df = df.round(2)
    colors = ['#FF6666', '#FF9999', '#FF3333', '#CC0000', '#990000', '#B22222', '#FF0044', '#960018']


    # print(df.head())

    # Select only index and accuracy column 
    # df = df[['center', 'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']]


    # Get current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    experiment_name = config['experiment']['name']
    experiment_dict = {
        # 'Experiment name': config['experiment']['name'],
                    'Model': config['model'],
                    'Dataset': config['dataset'],
                    'Number of participating institutions': config['num_clients'],
                    'Target metric': config['checkpoint_selection_metric'],
                    'Centre-Dropout method': config['dropout_method'],}

    model_types = ['federated', 'local', 'personalized']
    metrics = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall', 'specificity']

    general_results = {}
    general_results['table'] = df[['center'] + metrics].to_html(classes='table table-striped', index=False, border=0, justify='center')
    general_results['plot'] = {}

    df_general = df[['center', 'balanced_accuracy', 'n samples']]

    ax = df_general.plot.scatter(x='n samples', y='balanced_accuracy', c=colors[:len(df['center'].unique())], title='Center size vs balanced accuracy', figsize=(4, 3))
    # Add legend
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/general_n_samples_vs_balanced_accuracy.png')
    fig.clf()
    general_results['plot']['n_samples_vs_balanced_accuracy'] = 'images/general_n_samples_vs_balanced_accuracy.png'

    df_general = df_general.set_index('center')
    # Display value counts for each center
    def absolute_value(val):
        a  = int(val/100.*df['n samples'].sum())
        return a
    ax = df_general['n samples'].plot.pie(title='Distribution of samples',
                                        colors=colors[:len(df['center'].unique())],
                                        figsize=(4, 4), autopct=absolute_value, ylabel='')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/general_n_samples.png')
    fig.clf()
    general_results['plot']['n_samples'] = 'images/general_n_samples.png'

    # For each center create a table with with metrics containing string from model type
    per_center_results = {}

    # iterate over rows with iterrows()
    for center in df['center'].unique():
        center_results = {}
        for model_type in model_types:
            # print(df[df['center'] == center])
            if model_type == 'federated':
                center_results[model_type] = df[df['center'] == center][metrics]
            else:
                center_results[model_type] = df[df['center'] == center][df.columns[df.columns.str.contains(model_type)]]
                # Remove model type from column names
                center_results[model_type].columns = center_results[model_type].columns.str.replace(f'{model_type} ', '')
            
            # Remove columns with 'std' in the name
            center_results[model_type] = center_results[model_type].loc[:, ~center_results[model_type].columns.str.contains('std')]
            # Add model type as column and index
            center_results[model_type].insert(0, 'model', model_type)
            per_center_results[center] = center_results



    per_center_df = {}
    metrics_to_plot = ['balanced_accuracy', 'specificity', 'recall']
    # light  green, blue and orange
    # colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
    # Create df for each center with model type as row and metrics as columns
    for center in df['center'].unique():
        center_df = pd.DataFrame()
        per_center_df[center] = {}
        for model_type in model_types:
            # print(per_center_results[center][model_type])
            center_df = pd.concat([center_df, per_center_results[center][model_type]], axis=0)
        # per_center_df[center]['table'] = center_df.to_html(classes='table table-striped', index=True)
        per_center_df[center]['table'] = center_df.to_html(classes='table table-striped', index=False, border=0, justify='center')
        center_df = center_df.set_index('model')
        # Plot metric for each center
        per_center_df[center]['plot'] = []
        for metric in metrics_to_plot:
            ax = center_df[metric].plot.bar(color=colors[:len(model_types)], title=f'{metric}', figsize=(4, 3), rot=0,)
            fig = ax.get_figure()
            fig.tight_layout()
            fig.savefig(f'{fig_dir}/{center}_{metric}.png')
            per_center_df[center]['plot'].append(f'images/{center}_{metric}.png')


    # Template handling
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
    template = env.get_template('flcore/report/template.html')

    # Add section with experiment config

    # Render the template
    html = template.render(date = dt_string,
                        experiment_name=experiment_name,
                        experiment=experiment_dict, 
                        general_results=general_results,
                        per_center_results=per_center_df
                        )

    # Plot
    ax = df.plot.bar()
    fig = ax.get_figure()
    fig.savefig(f'{fig_dir}/plot.png')

    # Write the HTML file
    with open(f'{experiment_path}/report.html', 'w') as f:
        f.write(html)

    options = {   
        'disable-smart-shrinking': '',
        'quiet': '',
        'margin-top': '0.1in',
        'margin-right': '0in',
        'margin-bottom': '0in',
        'margin-left': '0in',
        'enable-local-file-access': ''

    }

    pdfkit.from_file(f'{experiment_path}/report.html', f'{experiment_path}/report.pdf', options = options)

# main

if __name__ == '__main__':
    generate_report('logs/CVD mortality prediction')