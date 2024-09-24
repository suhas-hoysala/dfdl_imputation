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
print("GPU available: ", tf.config.list_physical_devices('GPU'))

nthreads = 12  # Set the number of threads

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

def save_data(dataset_id, expr_clean, expr, sim):
    print(f"DS{dataset_id}: {expr_clean.shape}")
    os.makedirs(f'../SERGIO/imputation_data_2/DS{dataset_id}', exist_ok=True)
    np.save(f'../SERGIO/imputation_data_2/DS{dataset_id}/DS6_clean', expr_clean)
    np.save(f'../SERGIO/imputation_data_2/DS{dataset_id}/DS6_expr', expr)
    cmat_clean = sim.convert_to_UMIcounts(expr)
    cmat_clean = np.concatenate(cmat_clean, axis=1)
    np.save(f'../SERGIO/imputation_data_2/DS{dataset_id}/DS6_clean_counts', cmat_clean)

def sparse_ratio(data):
    # ndarray
    return 1 - np.count_nonzero(data) / data.size

def get_sparsity_of_binary_ind(sim, expr, expr_clean, percentile=45, dataset_id=6, iter=0):
    """
    Add outlier genes
    """
    expr_O = sim.outlier_effect(expr, outlier_prob=0.01, mean=5, scale=1)

    """
    Add Library Size Effect
    """
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean=4.5, scale=0.7)

    """
    Add Dropouts
    """
    binary_ind = sim.dropout_indicator(expr_O_L, shape=8, percentile=percentile)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)

    """
    Convert to UMI count
    """
    count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)

    """
    Make a 2d gene expression matrix
    """
    count_matrix = np.concatenate(count_matrix, axis=1)
    # count_matrix = np.concatenate(expr_O_L_D, axis=1)

    if iter == 1:
        row_sums = np.sum(count_matrix, axis=1, keepdims=True)
        count_matrix = count_matrix / row_sums
    elif iter == 2:
        count_matrix = np.log1p(count_matrix)
    elif iter == 3:
        means = np.mean(count_matrix, axis=0)
        stddevs = np.std(count_matrix, axis=0)
        stddevs[stddevs == 0] = 1e-8  # Avoid division by zero
        count_matrix = (count_matrix - means) / stddevs
    elif iter == 4:
        from sklearn.preprocessing import quantile_transform
        count_matrix = quantile_transform(count_matrix, axis=0, copy=True)
    elif iter == 5:
        total_counts = np.sum(count_matrix, axis=1, keepdims=True)
        total_counts[total_counts == 0] = 1e-8  # Avoid division by zero
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
# Set the number of iterations
num_iters = 6

for dataset in tqdm([datasets[1]]):
    sparse_ratios = []
    sparse_ratios_noisy = []
    results = []
    results_noisy = []
    # Dataset info
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
    sim, expr, expr_clean = experiment(dataset)
    save_data(dataset['dataset_id'], expr_clean, expr, sim)
    
    # Load ground truth for evaluation
    gt = np.zeros((number_genes, number_genes))
    with open(target_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.strip().split(',')
            target_index = int(float(line_list[0]))
            num_regs = int(float(line_list[1]))
            for i in range(num_regs):
                try:
                    reg_index = int(float(line_list[i + 2]))
                    gt[reg_index, target_index] = 1
                except:
                    continue

    for i in tqdm(range(num_iters-1, -1, -1), desc="Running iterations"):
        # Measure results on clean data
        # Check for zero variance genes and remove them
        gene_variances = np.var(expr_clean, axis=1)
        zero_variance_genes = np.where(gene_variances == 0)[0]
        if len(zero_variance_genes) > 0:
            print(f"Iteration {i}: Found {len(zero_variance_genes)} zero variance genes in clean data. Removing them.")
            expr_clean_filtered = np.delete(expr_clean, zero_variance_genes, axis=0)
            gt_filtered = np.delete(gt, zero_variance_genes, axis=0)
            gt_filtered = np.delete(gt_filtered, zero_variance_genes, axis=1)
        else:
            expr_clean_filtered = expr_clean
            gt_filtered = gt

        # Ensure no NaN or Inf values
        if np.any(np.isnan(expr_clean_filtered)) or np.any(np.isinf(expr_clean_filtered)):
            print(f"Iteration {i}: Clean data contains NaN or Inf values. Skipping iteration.")
            continue

        VIM_CLEAN = GENIE3(expr_clean_filtered.T, nthreads=nthreads, ntrees=100)
        gt_resized = gt_filtered

        # Compute ROC AUC
        gt_flat = gt_resized.flatten()
        vim_flat = VIM_CLEAN.flatten()
        roc_score = roc_auc_score(gt_flat, vim_flat)
        precision_k = precision_at_k(gt_resized, VIM_CLEAN, range(1, gt_resized.size))
        results.append({
            'iteration': i,
            'roc_auc': roc_score,
            'precision_k': precision_k
        })
        print(f"Finished experiment iter {i}")

        sparse_ratios.append(sparse_ratio(expr_clean_filtered))
        ratio, expr_O, libFactor, expr_O_L, binary_ind, count_matrix = get_sparsity_of_binary_ind(
            sim, expr, expr_clean, percentile=45, dataset_id=dataset['dataset_id'], iter=i)
        sparse_ratios_noisy.append(ratio)

        gene_variances_noisy = np.var(count_matrix, axis=1)
        zero_variance_genes_noisy = np.where(gene_variances_noisy == 0)[0]
        if len(zero_variance_genes_noisy) > 0:
            print(f"Iteration {i}: Found {len(zero_variance_genes_noisy)} zero variance genes in noisy data. Removing them.")
            count_matrix_filtered = np.delete(count_matrix, zero_variance_genes_noisy, axis=0)
            gt_noisy_filtered = np.delete(gt, zero_variance_genes_noisy, axis=0)
            gt_noisy_filtered = np.delete(gt_noisy_filtered, zero_variance_genes_noisy, axis=1)
        else:
            count_matrix_filtered = count_matrix
            gt_noisy_filtered = gt

        # Ensure no NaN or Inf values
        if np.any(np.isnan(count_matrix_filtered)) or np.any(np.isinf(count_matrix_filtered)):
            print(f"Iteration {i}: Noisy data contains NaN or Inf values. Skipping iteration.")
            continue

        VIM_NOISY = GENIE3(count_matrix_filtered.T, nthreads=nthreads, ntrees=100)
        # Adjust ground truth size if zero variance genes were removed
        gt_noisy_resized = gt_noisy_filtered

        gt_flat_noisy = gt_noisy_resized.flatten()
        vim_flat_noisy = VIM_NOISY.flatten()
        roc_score_noisy = roc_auc_score(gt_flat_noisy, vim_flat_noisy)
        precision_k_noisy = precision_at_k(gt_noisy_resized, VIM_NOISY, range(1, gt_noisy_resized.size))
        results_noisy.append({
            'iteration': i,
            'roc_auc': roc_score_noisy,
            'precision_k': precision_k_noisy
        })

    # Compute mean and standard deviation of ROC AUC scores
    roc_scores = [r['roc_auc'] for r in results]
    roc_scores_noisy = [r['roc_auc'] for r in results_noisy]
    print(f"Mean ROC AUC on Clean: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    print(f"Mean ROC AUC on Noisy: {np.mean(roc_scores_noisy):.4f} ± {np.std(roc_scores_noisy):.4f}")

    # Plot ROC AUC scores for clean data
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(roc_scores)+1), roc_scores, marker='o')
    plt.title(f'GENIE3 Performance Across Different SERGIO Simulations (DS{dataset["dataset_id"]}) - Clean Data')
    plt.xlabel('Iteration')
    plt.ylabel('ROC AUC')
    plt.grid(True)
    os.makedirs(f"../SERGIO/experiments/stochasticity", exist_ok=True)
    plt.savefig(f"../SERGIO/experiments/stochasticity/genie3_performance_DS{dataset['dataset_id']}_clean.png")
    plt.show()

    # Plot ROC AUC scores for noisy data
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(roc_scores_noisy)+1), roc_scores_noisy, marker='o')
    plt.title(f'GENIE3 Performance Across Different SERGIO Simulations (DS{dataset["dataset_id"]}) - Noisy Data')
    plt.xlabel('Iteration')
    plt.ylabel('ROC AUC')
    plt.grid(True)
    plt.savefig(f"../SERGIO/experiments/stochasticity/genie3_performance_DS{dataset['dataset_id']}_noisy.png")
    plt.show()

    # Print individual ROC AUC scores
    for r in results:
        print(f"Iteration {r['iteration']}: ROC AUC (Clean) = {r['roc_auc']:.4f}")

    for r in results_noisy:
        print(f"Iteration {r['iteration']}: ROC AUC (Noisy) = {r['roc_auc']:.4f}")

    print(f"Clean - Results for dataset {dataset_id}: {results}")
    print(f"Noisy - Results for dataset {dataset_id}: {results_noisy}")
    print(f"Clean - Sparse ratios for dataset {dataset_id}: {sparse_ratios}")
    print(f"Noisy - Sparse ratios for dataset {dataset_id}: {sparse_ratios_noisy}")