import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from GENIE3.GENIE3 import GENIE3
import numpy as np
import re
import SERGIO.SERGIO.sergio as sergio
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

nthreads = 12

# Function to parse dataset information
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

# Function to generate noisy expression data
def get_noisy_expr(sim, expr, percentile=45):
    # Add outlier genes
    expr_O = sim.outlier_effect(expr, outlier_prob=0.01, mean=4, scale=1)
    
    # Add Library Size Effect
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean=4.6, scale=0.4)
    
    # Add Dropouts
    binary_ind = sim.dropout_indicator(expr_O_L, shape=6.5, percentile=percentile)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    
    # Convert to UMI count
    expr_O_L_D_C = sim.convert_to_UMIcounts(expr_O_L_D)
    
    return expr_O_L_D_C

# Function to substitute dataset (denoising by substitution)
def substitute_dataset(ds, num_cell_types, cells_per_type):
    ds[ds == 0] = np.nan
    for i in range(num_cell_types):
        start = i * cells_per_type
        end = (i + 1) * cells_per_type
        ds_cell_type = ds[:, start:end]
        mean_array = np.nanmean(ds_cell_type, axis=1)
        var_array = np.nanvar(ds_cell_type, axis=1)
        var_array[var_array == 0] = 0.05  # Avoid division by zero
        std_array = np.sqrt(var_array)
        # Generate random values for each gene across all cells in the cell type
        ds_cell_type[:, :] = np.random.normal(
            loc=mean_array[:, np.newaxis],
            scale=std_array[:, np.newaxis],
            size=(ds_cell_type.shape[0], ds_cell_type.shape[1])
        )
    ds[ds < 0] = 0.0
    np.nan_to_num(ds, copy=False, nan=0.0)
    return ds

# Function to load ground truth network
def load_ground_truth(data_info):
    target_file = f"../SERGIO/data_sets/{data_info['pattern'].format(**data_info)}/Interaction_cID_{data_info['dynamics']}.txt"
    num_genes = data_info['number_genes']
    gt = np.zeros((num_genes, num_genes))
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
    return gt

# Function to compute Pearson correlation matrix
def get_pearson_correlation(ds):
    # ds should be genes x samples
    X = ds
    # Subtract the mean
    X_mean = X - np.mean(X, axis=1, keepdims=True)
    # Compute covariance matrix
    cov_matrix = np.dot(X_mean, X_mean.T)
    # Compute standard deviations
    std_devs = np.linalg.norm(X_mean, axis=1, keepdims=True)
    # Compute denominator
    denominator = np.dot(std_devs, std_devs.T)
    # Avoid division by zero
    denominator[denominator == 0] = 1
    # Compute Pearson correlation matrix
    gene_gene_corr_matrix = cov_matrix / denominator
    return gene_gene_corr_matrix

# Function to save individual VIM results
def save_vim_to_file(vim, dataset_name, iteration, method='GENIE3', dir_path="../new_impute/results/transfer/"):
    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    vim_file_path = os.path.join(dir_path, f'vim_{method}_{dataset_name}_iter_{iteration}.npy')
    np.save(vim_file_path, vim)
    print(f"    Saved VIM ({method}) for iteration {iteration} at {vim_file_path}")

# Function to save aggregated VIM results
def save_aggregated_vim(vim_aggregated, dataset_name, method='GENIE3', dir_path="../new_impute/results/transfer/"):
    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    vim_file_path = os.path.join(dir_path, f'vim_aggregated_{method}_{dataset_name}.npy')
    np.save(vim_file_path, vim_aggregated)
    print(f"    Saved aggregated VIM ({method}) at {vim_file_path}")

# Function to save results to log file
def save_results_to_file(dataset_name, iteration, auc, auc_aggregated=None, method='GENIE3', dir_path="../new_impute/results/transfer/"):
    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log_file_path = os.path.join(dir_path, f'log_{method}_{dataset_name}.txt')
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Dataset: {dataset_name}, Iteration: {iteration}\n")
        log_file.write(f"  AUC ({method}): {auc}\n")
        if auc_aggregated is not None:
            log_file.write(f"  Aggregated AUC ({method}): {auc_aggregated}\n")
        log_file.write("---------------------------------------------------\n")

print("Current working directory:", os.getcwd())
print("Checking for datasets in directory '../SERGIO/data_sets'")
datasets = get_datasets()
print(f"Found {len(datasets)} datasets.")

for data_info in [datasets[2]]:
    dataset_id = data_info['dataset_id']
    dataset_name = f'DS{dataset_id}'
    num_cell_types = data_info['number_bins']
    cells_per_type = data_info['number_sc']
    num_genes = data_info['number_genes']

    print(f"Processing Dataset {dataset_name}")
    
    # Initialize the simulator
    sim = sergio.sergio(
        number_genes=num_genes,
        number_bins=num_cell_types,
        number_sc=cells_per_type,
        noise_params=1,
        decays=0,
        sampling_state=13,
        noise_type='dpd'
    )
    
    # Load ground truth
    gt = load_ground_truth(data_info)
    
    # Initialize lists to store VIMs and AUCs
    VIMs_GENIE3 = []
    VIMs_Pearson = []
    AUCs_GENIE3 = []
    AUCs_Pearson = []
    ds_clean_path = f'../SERGIO/imputation_data_2/{dataset_name}/DS6_clean.npy'
    ds_expr_path = f'../SERGIO/imputation_data_2/{dataset_name}/DS6_expr.npy'

    for iteration in range(6):
        print(f"  Iteration {iteration}")
        # Check if data files exist
        if not os.path.exists(ds_clean_path) or not os.path.exists(ds_expr_path):
            print(os.path.exists(ds_clean_path), ds_clean_path)
            print(os.path.exists(ds_expr_path), ds_expr_path)
            print(f"    Data files for iteration {iteration} not found. Skipping.")
            continue

        # Load datasets
        ds_clean = np.load(ds_clean_path)
        ds_expr = np.load(ds_expr_path)
        
        # Generate noisy data
        expr_noisy = get_noisy_expr(sim, ds_expr)
        ds_noisy = np.concatenate(expr_noisy, axis=1)

        # Substitute dataset (denoising)
        ds_substitute = substitute_dataset(ds_noisy.astype(np.float32), num_cell_types, cells_per_type)
        
        # Compute VIM using GENIE3
        VIM_GENIE3 = GENIE3(
            np.transpose(ds_substitute),
            nthreads=nthreads,
            ntrees=100,
            regulators='all',
            gene_names=[str(i) for i in range(np.transpose(ds_substitute).shape[1])]
        )
        
        # Compute ROC AUC score for GENIE3
        auc_GENIE3 = roc_auc_score(gt.flatten(), VIM_GENIE3.flatten())
        print(f"    AUC (GENIE3): {auc_GENIE3}")
        
        # Store VIM and AUC for GENIE3
        VIMs_GENIE3.append(VIM_GENIE3)
        AUCs_GENIE3.append(auc_GENIE3)
        
        # Save individual VIM and AUC for GENIE3
        save_vim_to_file(VIM_GENIE3, dataset_name, iteration, method='GENIE3')
        save_results_to_file(dataset_name, iteration, auc_GENIE3, method='GENIE3')
        
        # Compute VIM using Pearson correlation
        VIM_Pearson = get_pearson_correlation(ds_substitute)
        
        # Compute ROC AUC score for Pearson
        auc_Pearson = roc_auc_score(gt.flatten(), VIM_Pearson.flatten())
        print(f"    AUC (Pearson): {auc_Pearson}")
        
        # Store VIM and AUC for Pearson
        VIMs_Pearson.append(VIM_Pearson)
        AUCs_Pearson.append(auc_Pearson)
        
        # Save individual VIM and AUC for Pearson
        save_vim_to_file(VIM_Pearson, dataset_name, iteration, method='Pearson')
        save_results_to_file(dataset_name, iteration, auc_Pearson, method='Pearson')

    # Aggregate VIMs over iterations for GENIE3
    if VIMs_GENIE3:
        VIM_GENIE3_aggregated = sum(VIMs_GENIE3)
        auc_GENIE3_aggregated = roc_auc_score(gt.flatten(), VIM_GENIE3_aggregated.flatten())
        print(f"  Aggregated AUC over iterations (GENIE3): {auc_GENIE3_aggregated}")
        
        # Save aggregated VIM and AUC for GENIE3
        save_aggregated_vim(VIM_GENIE3_aggregated, dataset_name, method='GENIE3')
        save_results_to_file(dataset_name, 'Aggregated', auc_GENIE3_aggregated, method='GENIE3')
    else:
        print(f"  No VIMs computed for Dataset {dataset_name} using GENIE3")
    
    # Aggregate VIMs over iterations for Pearson
    if VIMs_Pearson:
        VIM_Pearson_aggregated = sum(VIMs_Pearson)
        auc_Pearson_aggregated = roc_auc_score(gt.flatten(), VIM_Pearson_aggregated.flatten())
        print(f"  Aggregated AUC over iterations (Pearson): {auc_Pearson_aggregated}")
        
        # Save aggregated VIM and AUC for Pearson
        save_aggregated_vim(VIM_Pearson_aggregated, dataset_name, method='Pearson')
        save_results_to_file(dataset_name, 'Aggregated', auc_Pearson_aggregated, method='Pearson')
    else:
        print(f"  No VIMs computed for Dataset {dataset_name} using Pearson")

    print(f"Finished processing Dataset {dataset_name}\n")
