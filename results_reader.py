import pandas as pd
import pickle
import os

def extract_hyperparameters_final(filename):
    filename = filename.replace('.dict', '')
    parts = filename.split('_')
    dataset = parts[0]
    experiment_setup = '_'.join(parts[1:3])
    return dataset, experiment_setup

def extract_hyperparameters_finetune(filename):
    filename = filename.replace('.dict', '')
    parts = filename.split('_')
    dataset = parts[0]
    experiment_setup = '_'.join(parts[1:-3])
    la = float(parts[-3][2:])
    th = float(parts[-2][2:])
    bs = int(parts[-1][2:].split('.')[0])
    return dataset, experiment_setup, la, th, bs

def load_results(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.dict'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if '_finetune' in filename:
                dataset, experiment_setup, la, th, bs = extract_hyperparameters_finetune(filename)
                result = {
                    'dataset': dataset,
                    'experiment_setup': experiment_setup,
                    'lambda': la,
                    'theta': th,
                    'batch_size': bs,
                    'drift started at': data.get('drift started at', data.get('Drift started at')),
                    'drift detected': data.get('drift detected', data.get('Drift Detected')),
                    'data': data
                }
            else:
                dataset, experiment_setup = extract_hyperparameters_final(filename)
                result = {
                    'dataset': dataset,
                    'experiment_setup': experiment_setup,
                    'drift started at': data.get('drift started at', data.get('Drift started at')),
                    'drift detected': data.get('drift detected', data.get('Drift Detected')),
                    'data': data
                }
            results.append(result)
    return results

def read_results(directory):
    results = load_results(directory)
    df = pd.DataFrame(results)
    return df
