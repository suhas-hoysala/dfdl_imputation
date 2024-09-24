import sys, os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from GENIE3.GENIE3 import GENIE3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from scipy import stats
import SERGIO.SERGIO.sergio as sergio
from sklearn.metrics import roc_auc_score
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

nthreads = 12

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

def zero_impute(ds, num_cell_types, num_genes, cells_per_type):
    ds[ds == 0] = np.nan
    for i in range(num_cell_types):
        start_idx = i * cells_per_type
        end_idx = (i + 1) * cells_per_type
        ds_cell_type = ds[:, start_idx:end_idx]
        mean_array = np.nanmean(ds_cell_type, axis=1)
        var_array = np.nanvar(ds_cell_type, axis=1)
        for j in range(num_genes):
            np.nan_to_num(
                ds_cell_type[j, :],
                copy=False,
                nan=np.random.normal(
                    loc=mean_array[j],
                    scale=np.sqrt(var_array[j]) if var_array[j] > 0 else 0.1  # Avoid sqrt(0)
                )
            )
    ds[ds < 0] = 0.0
    np.nan_to_num(ds, copy=False)
    return ds

def substitute_dataset(ds, num_cell_types, num_genes, cells_per_type):
    ds[ds == 0] = np.nan
    for i in range(num_cell_types):
        start_idx = i * cells_per_type
        end_idx = (i + 1) * cells_per_type
        ds_cell_type = ds[:, start_idx:end_idx]
        mean_array = np.nanmean(ds_cell_type, axis=1)
        var_array = np.nanvar(ds_cell_type, axis=1)
        for j in range(num_genes):
            ds_cell_type[j, :] = np.random.normal(
                loc=mean_array[j],
                scale=np.sqrt(var_array[j]) if var_array[j] > 0 else 0.1,  # Avoid sqrt(0)
                size=cells_per_type
            )
    ds[ds < 0] = 0.0
    np.nan_to_num(ds, copy=False)
    return ds

class CNNMultiCTNet(nn.Module):
    def __init__(self):
        super(CNNMultiCTNet, self).__init__()
        self.num_ct = 9
        
        # Separate 2D convolution layers for each ct
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1) for _ in range(self.num_ct)])
        
        # Separate fully connected layers for mu_(ct,g) and sigma_(ct,g) for each ct
        self.fc_mu_layers = nn.ModuleList([nn.Linear(100, 100) for _ in range(self.num_ct)])
        self.fc_sigma_layers = nn.ModuleList([nn.Linear(100, 100) for _ in range(self.num_ct)])

    def forward(self, x):
        # Input shape: (batch_size, 2, 5, 100)
        batch_size = x.size(0)

        # Prepare lists to store results for each ct
        mu_g_hat_list = []
        sigma_g_hat_list = []

        for ct in range(self.num_ct):
            # Extract the mu_{ct,g} and sigma_{ct,g} for this ct (shape: (batch_size, 2, 100))
            x_ct = x[:, :, ct, :].view(batch_size, 2, 100)

            # Pass through the convolution layer specific to this ct
            x_ct = self.conv_layers[ct](x_ct)  # (batch_size, 2, 100)

            # Separate the mu_{ct,g} and sigma_{ct,g}
            mu_ct_g = x_ct[:, 0, :]  # (batch_size, 100)
            sigma_ct_g = x_ct[:, 1, :]  # (batch_size, 100)

            # Pass through fully connected layers specific to this ct
            mu_ct_g_hat = self.fc_mu_layers[ct](mu_ct_g)  # (batch_size, 100)
            sigma_ct_g_hat = self.fc_sigma_layers[ct](sigma_ct_g)  # (batch_size, 100)

            # Apply activation (ReLU) if necessary
            mu_ct_g_hat = F.relu(mu_ct_g_hat)
            sigma_ct_g_hat = F.relu(sigma_ct_g_hat)

            # Sum the mu_ct_g_hat to normalize it
            sum_mu_ct_g_hat = torch.sum(mu_ct_g_hat, dim=1, keepdim=True)  # (batch_size, 1)

            # Scale the mu_ct_g_hat and sigma_ct_g_hat
            mu_ct_g_hat = mu_ct_g_hat / sum_mu_ct_g_hat  # Normalize mu_ct_g_hat so the sum is 1
            sigma_ct_g_hat = sigma_ct_g_hat / sum_mu_ct_g_hat  # Scale sigma_ct_g_hat with the same factor

            # Collect the results for each ct
            mu_g_hat_list.append(mu_ct_g_hat.unsqueeze(1))
            sigma_g_hat_list.append(sigma_ct_g_hat.unsqueeze(1))

        # Stack the results along the ct dimension
        mu_g_hat = torch.cat(mu_g_hat_list, dim=1)
        sigma_g_hat = torch.cat(sigma_g_hat_list, dim=1)

        # Combine mu and sigma along the 2nd dimension (channels)
        output = torch.stack([mu_g_hat, sigma_g_hat], dim=1)

        return output

def find_x(ds, x_means, x_stds):
    ds[ds == 0] = np.nan
    for i in range(9):
        ds_cell_type = ds[:,i*300:(i+1)*300]
        x_means[i,:] = np.nanmean(ds_cell_type, axis=1)
        x_stds[i,:] = np.nanstd(ds_cell_type, axis=1)
    x = np.stack([x_means, x_stds], axis=0)
    x = np.expand_dims(x, axis=0)
    x = np.nan_to_num(x)
    x = torch.tensor(x,dtype=torch.float32).to(device)
    return(x)

def simulate_dataset(ds, y_means, y_stds):
    ds = np.zeros_like(ds)
    for i in range(9):
        ds_cell_type = ds[:,i*300:(i+1)*300]
        mean_array = y_means[i,:]
        std_array = y_stds[i,:]
        for j in range(100):
            ds_cell_type[j,:] = np.random.normal(loc=mean_array[j],scale=std_array[j],size=300)
    ds[ds<0] = 0.0
    np.nan_to_num(ds,copy=False)
    return(ds)

cleans = {
    1: '../SERGIO/imputation_data_2/DS1/DS6_clean.npy',
    2: '../SERGIO/imputation_data_2/DS2/DS6_clean.npy',
    3: '../SERGIO/imputation_data_2/DS3/DS6_clean.npy'
}
ds_exprs = {
    1: '../SERGIO/imputation_data/DS1/DS6_expr.npy',
    2: '../SERGIO/imputation_data/DS2/DS6_expr.npy',
    3: '../SERGIO/imputation_data/DS3/DS6_expr.npy'
}

datasets = get_datasets()
for data_info in datasets:
    dataset_id = data_info['dataset_id']
    ds_clean = np.load(cleans[dataset_id])
    ds_expr = np.load(ds_exprs[dataset_id])

    number_genes = data_info['number_genes']
    num_cell_types = data_info['number_bins']
    cells_per_type = data_info['number_sc']

    sim = sergio.sergio(
        number_genes=number_genes,
        number_bins=num_cell_types,
        number_sc=cells_per_type,
        noise_params=1,
        decays=0.8,
        sampling_state=15,
        noise_type='dpd'
    )

    target_file = f"../SERGIO/data_sets/{data_info['pattern'].format(**data_info)}/Interaction_cID_{data_info['dynamics']}.txt"
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

    # Part 1: Evaluate the performance on clean and progressively noisier datasets
    # Clean Dataset
    VIM_clean = GENIE3(
        np.transpose(ds_clean),
        nthreads=nthreads,
        ntrees=100,
        regulators='all',
        gene_names=[str(s) for s in range(number_genes)]
    )

    # Add outlier noise
    expr_O = sim.outlier_effect(ds_expr, outlier_prob=0.01, mean=5, scale=1)
    ds_O = np.concatenate(expr_O, axis=1)
    VIM_O = GENIE3(
        np.transpose(ds_O),
        nthreads=nthreads,
        ntrees=100,
        regulators='all',
        gene_names=[str(s) for s in range(number_genes)]
    )

    # Add library noise on top
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean=4.5, scale=0.7)
    ds_O_L = np.concatenate(expr_O_L, axis=1)
    VIM_O_L = GENIE3(
        np.transpose(ds_O_L),
        nthreads=nthreads,
        ntrees=100,
        regulators='all',
        gene_names=[str(s) for s in range(number_genes)]
    )

    # Add dropouts on top
    binary_ind = sim.dropout_indicator(expr_O_L, shape=6, percentile=45)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    ds_O_L_D = np.concatenate(expr_O_L_D, axis=1)
    VIM_O_L_D = GENIE3(
        np.transpose(ds_O_L_D),
        nthreads=nthreads,
        ntrees=100,
        regulators='all',
        gene_names=[str(s) for s in range(number_genes)]
    )

    # Convert to UMI Counts
    expr_O_L_D_C = sim.convert_to_UMIcounts(expr_O_L_D)
    ds_O_L_D_C = np.concatenate(expr_O_L_D_C, axis=1)
    VIM_O_L_D_C = GENIE3(
        np.transpose(ds_O_L_D_C),
        nthreads=nthreads,
        ntrees=100,
        regulators='all',
        gene_names=[str(s) for s in range(number_genes)]
    )

    # Part 2: Apply normalized imputation and assess its effect
    ds_noisy = ds_O_L_D_C
    ds_imputed = zero_impute(ds_noisy.astype(np.float32), num_cell_types, number_genes, cells_per_type)

    # Correct the creation of lib_depth_matrix
    lib_depth_matrix = np.tile(np.sum(ds_imputed, axis=0), (number_genes, 1))
    ds_normalized = ds_imputed / lib_depth_matrix

    VIM_normalized = GENIE3(
        np.transpose(ds_normalized),
        nthreads=nthreads,
        ntrees=100,
        regulators='all',
        gene_names=[str(s) for s in range(number_genes)]
    )

    # Part 3: Use data substitution for denoising and evaluate
    ds_substitute = substitute_dataset(ds_noisy.astype(np.float32), num_cell_types, number_genes, cells_per_type)
    VIM_substitute = GENIE3(
        np.transpose(ds_substitute),
        nthreads=nthreads,
        ntrees=100,
        regulators='all',
        gene_names=[str(s) for s in range(number_genes)]
    )

    auc_clean = roc_auc_score(gt.flatten(), VIM_clean.flatten())
    auc_O = roc_auc_score(gt.flatten(), VIM_O.flatten())
    auc_O_L = roc_auc_score(gt.flatten(), VIM_O_L.flatten())
    auc_O_L_D = roc_auc_score(gt.flatten(), VIM_O_L_D.flatten())
    auc_O_L_D_C = roc_auc_score(gt.flatten(), VIM_O_L_D_C.flatten())
    auc_normalized = roc_auc_score(gt.flatten(), VIM_normalized.flatten())
    auc_substitute = roc_auc_score(gt.flatten(), VIM_substitute.flatten())

    log_dir = './results/denoise/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_DS{data_info["dataset_id"]}.txt')

    with open(log_file, 'w') as f:
        f.write(f'Dataset ID: {data_info["dataset_id"]}\n')
        f.write(f'AUC Clean: {auc_clean}\n')
        f.write(f'AUC O: {auc_O}\n')
        f.write(f'AUC O_L: {auc_O_L}\n')
        f.write(f'AUC O_L_D: {auc_O_L_D}\n')
        f.write(f'AUC O_L_D_C: {auc_O_L_D_C}\n')
        f.write(f'AUC Normalized: {auc_normalized}\n')
        f.write(f'AUC Substitute: {auc_substitute}\n')
