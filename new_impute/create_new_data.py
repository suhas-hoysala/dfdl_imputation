# %%
import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from GENIE3.GENIE3 import *
from sklearn.metrics import roc_auc_score
from utils import gt_benchmark, precision_at_k
import SERGIO.SERGIO.sergio as sergio
import re
import os
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import hashlib

import tensorflow as tf
print("gpu available: ", tf.config.list_physical_devices('GPU'))
# !pwd

# %%
def parse_dataset_name(folder_name):
    pattern1 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_dynamics_(\d+)_DS(\d+)'
    pattern2 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_(\d+)_DS(\d+)'
    match_p1 = re.match(pattern1, folder_name)
    match_p2 = re.match(pattern2, folder_name)
    if match_p1:
        return {
            'number_genes': int(match_p1.group(1)),
            'number_bins': int(match_p1.group(2)),
            'number_sc': int(match_p1.group(3)),
            'dynamics': int(match_p1.group(4)),
            'dataset_id': int(match_p1.group(5)),
            "pattern": "De-noised_{number_genes}G_{number_bins}T_{number_sc}cPerT_dynamics_{dynamics}_DS{dataset_id}"
        }
    if match_p2:
        return {
            'number_genes': int(match_p2.group(1)),
            'number_bins': int(match_p2.group(2)),
            'number_sc': int(match_p2.group(3)),
            'dynamics': int(match_p2.group(4)),
            'dataset_id': int(match_p2.group(5)),
            "pattern": "De-noised_{number_genes}G_{number_bins}T_{number_sc}cPerT_{dynamics}_DS{dataset_id}"
        }
    return

def get_datasets():
    datasets = []
    for folder_name in os.listdir('../SERGIO/data_sets'):
        dataset_info = parse_dataset_name(folder_name)
        if dataset_info:
            datasets.append(dataset_info)
    return sorted(datasets, key=lambda x: x['dataset_id'])

def fstr(template):
    return eval(f'f"""{template}"""')

def experiment(data_info):
    sim = sergio.sergio(
        number_genes=data_info["number_genes"],
        number_bins=data_info["number_bins"], 
        number_sc=data_info["number_sc"],
        noise_params=1,
        decays=0.8, 
        sampling_state=15,
        noise_type='dpd'
    )
    # sim.build_graph(input_file_taregts ='data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt',\
    #                 input_file_regs='data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt', shared_coop_state=2)
    number_genes = data_info["number_genes"]
    number_bins = data_info["number_bins"]
    number_sc = data_info["number_sc"]
    dynamics = data_info["dynamics"]
    dataset_id = data_info["dataset_id"]
    pattern = data_info["pattern"]
    folder_name = pattern.format(number_genes=number_genes, number_bins=number_bins, 
                                 number_sc=number_sc, dynamics=dynamics, dataset_id=dataset_id)
    input_file_targets = f'../SERGIO/data_sets/{folder_name}/Interaction_cID_{data_info["dynamics"]}.txt'
    input_file_regs = f'../SERGIO/data_sets/{folder_name}/Regs_cID_{data_info["dynamics"]}.txt'
    
    sim.build_graph(
        input_file_taregts=input_file_targets,
        input_file_regs=input_file_regs,
        shared_coop_state=2
    )
    sim.simulate()
    expr = sim.getExpressions()
    expr_clean = np.concatenate(expr, axis=1)
    return sim, expr, expr_clean

def save_data(dataset_id, expr_clean, expr, sim, iter=0):
    print(f"DS{dataset_id}: {expr_clean.shape}")
    os.makedirs(f'../SERGIO/imputation_data_2/DS{dataset_id}', exist_ok=True)
    np.save(f'../SERGIO/imputation_data_2/DS{dataset_id}/DS6_clean_iter_{iter}', expr_clean)
    np.save(f'../SERGIO/imputation_data_2/DS{dataset_id}/DS6_expr_iter_{iter}', expr)
    cmat_clean = sim.convert_to_UMIcounts(expr)
    cmat_clean = np.concatenate(cmat_clean, axis=1)
    np.save(f'../SERGIO/imputation_data_2/DS{dataset_id}/DS6_clean_counts_iter_{iter}', cmat_clean)

def sparse_ratio(data):
    # ndarray
    return 1 - np.count_nonzero(data) / data.size

def get_sparsity_of_binary_ind(sim, expr, expr_clean, percentile=45, dataset_id=6, iter=0):
    """
    Add outlier genes
    """
    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 5, scale = 1)

    """
    Add Library Size Effect
    """
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.5, scale = 0.7)

    """
    Add Dropouts
    """
    binary_ind = sim.dropout_indicator(expr_O_L, shape = 8, percentile = percentile)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)

    """
    Convert to UMI count
    """
    count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)

    """
    Make a 2d gene expression matrix
    """
    count_matrix = np.concatenate(count_matrix, axis = 1)
    # count_matrix = np.concatenate(expr_O_L_D, axis = 1)

    if iter == 1:
        row_sums = np.sum(count_matrix, axis=1, keepdims=True)
        count_matrix = count_matrix / row_sums
    elif iter == 2:
        count_matrix = np.log1p(count_matrix)
    elif iter == 3:
        means = np.mean(count_matrix, axis=0)
        stddevs = np.std(count_matrix, axis=0)
        count_matrix = (count_matrix - means) / stddevs
    elif iter == 4:
        from sklearn.preprocessing import quantile_transform
        count_matrix = quantile_transform(count_matrix, axis=0, copy=True)
    elif iter == 5:
        total_counts = np.sum(count_matrix, axis=1, keepdims=True)
        count_matrix = (count_matrix / total_counts) * 1e6
    
    os.makedirs(os.path.dirname(f'../SERGIO/imputation_data_2/DS{dataset_id}'), exist_ok=True)
    np.save(f'../SERGIO/imputation_data_2/DS{dataset_id}/DS6_45_iter_{iter}', count_matrix)
    print(count_matrix.shape)
    return sparse_ratio(binary_ind), expr_O, libFactor, expr_O_L, binary_ind, count_matrix

def compute_checksum(data):
    if isinstance(data, np.ndarray):
        return hashlib.md5(data.tobytes()).hexdigest()
    elif isinstance(data, list):
        return hashlib.md5(str(data).encode()).hexdigest()
    else:
        return hashlib.md5(str(data).encode()).hexdigest()

def compute_stats(data):
    if isinstance(data, np.ndarray):
        return {
            'checksum': compute_checksum(data),
            'mean': np.mean(data),
            'std': np.std(data)
        }
    elif isinstance(data, list):
        return {
            'checksum': compute_checksum(data),
            'mean': np.mean(data),
            'std': np.std(data)
        }
    else:
        return {
            'checksum': compute_checksum(data),
            'value': data
        }

def compare_attempts(attempts):
    comparison = {}
    for key in attempts[0].keys():
        values = [attempt[key]['checksum'] for attempt in attempts]
        comparison[key] = len(set(values)) > 1
    return comparison

# %% [markdown]
# Experiment on clean data, generate clean data multiple times, and see if metrics on clean data have changed significantly.

# %%
datasets = get_datasets()
# TODO: change iter num
num_iters = 6
# TODO: change datasets
for dataset in tqdm([datasets[0]]):
    sparse_ratios = []
    sparse_ratios_noisy = []
    results = []
    results_noisy = []
    # dataset info
    number_genes = dataset["number_genes"]
    number_bins = dataset["number_bins"]
    number_sc = dataset["number_sc"]
    dynamics = dataset["dynamics"]
    dataset_id = dataset["dataset_id"]
    pattern = dataset["pattern"]
    folder_name = pattern.format(number_genes=number_genes, number_bins=number_bins, 
                                number_sc=number_sc, dynamics=dynamics, dataset_id=dataset_id)
    target_file = f'../SERGIO/data_sets/{folder_name}/Interaction_cID_{dataset["dynamics"]}.txt'
    input_file = f'../SERGIO/data_sets/{folder_name}/Regs_cID_{dataset["dynamics"]}.txt'
    for i in tqdm(range(num_iters), desc="Running iterations"):
        sim, expr, expr_clean = experiment(dataset)
        
        # measure results on clean data
        VIM_CLEAN = GENIE3(expr_clean.T, ntrees=100)
        gt, rescaled_vim = gt_benchmark(VIM_CLEAN, target_file)
        roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
        precision_k = precision_at_k(gt, rescaled_vim, range(1, gt.size))
        results.append({
            'iteration': i,
            'roc_auc': roc_score,
            'precision_k': precision_k
        })
        print(f"finished experiment iter {i}")
        save_data(dataset['dataset_id'], expr_clean, expr, sim, iter=i)
        print("saved data")
        sparse_ratios.append(sparse_ratio(expr_clean))
        ratio, expr_O, libFactor, expr_O_L, binary_ind, count_matrix = get_sparsity_of_binary_ind(sim, expr, expr_clean, percentile=0.45, dataset_id=dataset['dataset_id'], iter=i)
        sparse_ratios_noisy.append(ratio)
        # measure results on noisy data
        VIM_NOISY = GENIE3(count_matrix.T, ntrees=100)
        gt, rescaled_vim = gt_benchmark(VIM_NOISY, target_file)
        roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
        precision_k = precision_at_k(gt, rescaled_vim, range(1, gt.size))
        results_noisy.append({
            'iteration': i,
            'roc_auc': roc_score,
            'precision_k': precision_k
        })
    
    roc_scores = [r['roc_auc'] for r in results]
    roc_scores_noisy = [r['roc_auc'] for r in results_noisy]
    print(f"Mean ROC AUC on Clean: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    print(f"Mean ROC AUC on Noisy: {np.mean(roc_scores_noisy):.4f} ± {np.std(roc_scores_noisy):.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(roc_scores)+1), roc_scores, marker='o')
    plt.title(f'GENIE3 Performance Across Different SERGIO Simulations (DS{dataset["dataset_id"]})')
    plt.xlabel('Iteration')
    plt.ylabel('ROC AUC')
    plt.grid(True)
    os.makedirs(f"../SERGIO/experiments/stochasticity", exist_ok=True)
    plt.savefig(f"../SERGIO/experiments/stochasticity/genie3_performance_DS{dataset['dataset_id']}_clean.png")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(roc_scores_noisy)+1), roc_scores_noisy, marker='o')
    plt.title(f'GENIE3 Performance Across Different SERGIO Simulations (DS{dataset["dataset_id"]})')
    plt.xlabel('Iteration')
    plt.ylabel('ROC AUC')
    plt.grid(True)
    os.makedirs(f"../SERGIO/experiments/stochasticity", exist_ok=True)
    plt.savefig(f"../SERGIO/experiments/stochasticity/genie3_performance_DS{dataset['dataset_id']}_noisy.png")
    plt.show()

    for r in results:
        print(f"Iteration {r['iteration']}: ROC AUC = {r['roc_auc']:.4f}")
    
    for r in results_noisy:
        print(f"Iteration {r['iteration']}: ROC AUC = {r['roc_auc']:.4f}")
    print(f"Clean - Results for dataset {dataset_id}: {results}")
    print(f"Noisy - Results for dataset {dataset_id}: {results_noisy}")
    print(f"Clean - Sparse ratios for dataset {dataset_id}: {sparse_ratios}")
    print(f"Noisy - Sparse ratios for dataset {dataset_id}: {sparse_ratios_noisy}")
