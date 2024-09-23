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

# Load Simulation
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

def zero_impute(ds):
    ds[ds == 0] = np.nan
    for i in range(9):
        ds_cell_type = ds[:, i*300 : (i+1)*300]
        mean_array = np.nanmean(ds_cell_type, axis=1)  # shape (100,)
        var_array = np.nanvar(ds_cell_type, axis=1)    # shape (100,)
        std_array = np.sqrt(var_array)
        # Generate random numbers for each gene across all cells
        ds_cell_type[:, :] = np.random.normal(
            loc=mean_array[:, np.newaxis],
            scale=std_array[:, np.newaxis],
            size=(100, 300)
        )
    ds[np.isnan(ds)] = 0.0
    return ds

def substitute_dataset(ds):
    ds[ds == 0] = np.nan
    for i in range(9):
        ds_cell_type = ds[:, i*300 : (i+1)*300]
        mean_array = np.nanmean(ds_cell_type, axis=1)
        var_array = np.nanvar(ds_cell_type, axis=1)
        std_array = np.sqrt(var_array)
        ds_cell_type[:, :] = np.random.normal(
            loc=mean_array[:, np.newaxis],
            scale=std_array[:, np.newaxis],
            size=(100, 300)
        )
    ds[np.isnan(ds)] = 0.0
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
    x = torch.tensor(x, dtype=torch.float32).to(device)
    return(x)

def simulate_dataset(ds, y_means, y_stds):
    ds = np.zeros_like(ds)
    for i in range(9):
        ds_cell_type = ds[:, i*300 : (i+1)*300]
        mean_array = y_means[i, :]  # shape (100,)
        std_array = y_stds[i, :]    # shape (100,)
        ds_cell_type[:, :] = np.random.normal(
            loc=mean_array[:, np.newaxis],
            scale=std_array[:, np.newaxis],
            size=(100, 300)
        )
    ds[ds < 0] = 0.0
    return ds

cleans = {
    1: '../SERGIO/imputation_data/DS1/DS6_clean_iter_0.npy',
    2: '../SERGIO/imputation_data/DS2/DS6_clean_iter_0.npy',
    3: '../SERGIO/imputation_data/DS3/DS6_clean.npy'
}
ds_exprs = {
    1: '../SERGIO/imputation_data/DS1/DS6_expr_iter_0.npy',
    2: '../SERGIO/imputation_data/DS2/DS6_expr_iter_0.npy',
    3: '../SERGIO/imputation_data/DS3/DS6_expr.npy'
}
datasets = get_datasets()
for data_info in datasets[1:]:
    # Load Data
    ds_clean = np.load(cleans[data_info['dataset_id']])
    ds_expr = np.load(ds_exprs[data_info['dataset_id']])

    sim = sergio.sergio(
        number_genes=data_info['number_genes'],
        number_bins=data_info['number_bins'],
        number_sc=data_info['number_sc'],
        noise_params=1,
        decays=0.8,
        sampling_state=15,
        noise_type='dpd'
    )

    # Load Ground Truth
    target_file = f"../SERGIO/data_sets/{data_info['pattern'].format(**data_info)}/Interaction_cID_{data_info['dynamics']}.txt"
    gt = np.zeros((100, 100))
    f = open(target_file, 'r')
    lines = f.readlines()
    f.close()
    for j in range(len(lines)):
        line = lines[j]
        line_list = line.split(',')
        target_index = int(float(line_list[0]))
        num_regs = int(float(line_list[1]))
        for i in range(num_regs):
            try:
                reg_index = int(float(line_list[i+2]))
                gt[reg_index, target_index] = 1
            except:
                continue

    # Part 1: Evaluate the performance on clean and progressively noisier datasets
    # Clean Dataset
    VIM_clean = GENIE3(np.transpose(ds_clean), ntrees=100, regulators='all', gene_names=[str(s) for s in range(np.transpose(ds_clean).shape[1])])
    # roc_auc_score(gt.flatten(), VIM_clean.flatten())

    # Add outlier noise
    expr_O = sim.outlier_effect(ds_expr, outlier_prob = 0.01, mean = 5, scale = 1)
    ds_O = np.concatenate(expr_O, axis=1)
    VIM_O = GENIE3(np.transpose(ds_O), ntrees=100, regulators='all', gene_names=[str(s) for s in range(np.transpose(ds_O).shape[1])])
    # roc_auc_score(gt.flatten(), VIM_O.flatten())

    # Add library noise on top
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.5, scale = 0.7)
    ds_O_L = np.concatenate(expr_O_L, axis=1)
    VIM_O_L = GENIE3(np.transpose(ds_O_L), ntrees=100, regulators='all', gene_names=[str(s) for s in range(np.transpose(ds_O_L).shape[1])])
    # roc_auc_score(gt.flatten(), VIM_O_L.flatten())

    # Add dropouts on top
    binary_ind = sim.dropout_indicator(expr_O_L, shape = 6, percentile = 45)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    ds_O_L_D = np.concatenate(expr_O_L_D, axis=1)
    VIM_O_L_D = GENIE3(np.transpose(ds_O_L_D), ntrees=100, regulators='all', gene_names=[str(s) for s in range(np.transpose(ds_O_L_D).shape[1])])
    # roc_auc_score(gt.flatten(), VIM_O_L_D.flatten())

    # Convert to UMI Counts
    expr_O_L_D_C = sim.convert_to_UMIcounts(expr_O_L_D)
    ds_O_L_D_C = np.concatenate(expr_O_L_D_C, axis=1)
    VIM_O_L_D_C = GENIE3(np.transpose(ds_O_L_D_C), ntrees=100, regulators='all', gene_names=[str(s) for s in range(np.transpose(ds_O_L_D_C).shape[1])])
    # roc_auc_score(gt.flatten(), VIM_O_L_D_C.flatten())


    # Part 2: Apply normalized imputation and assess its effect
    ds_noisy = ds_O_L_D_C
    ds_imputed = zero_impute(ds_noisy.astype(np.float32))
    lib_depth_matrix = np.tile(np.sum(ds_imputed, axis=0), (100, 1))
    ds_normalized = ds_imputed / lib_depth_matrix

    VIM_normalized = GENIE3(np.transpose(ds_normalized), ntrees=100, regulators='all',
                            gene_names=[str(s) for s in range(np.transpose(ds_normalized).shape[1])])
    # roc_auc_score(gt.flatten(), VIM_normalized.flatten())


    # Part 3: Use data substitution for denoising and evaluate
    ds_noisy = ds_O_L_D_C
    # Dataset substitution
    ds_substitute = substitute_dataset(ds_noisy.astype(np.float32))
    VIM_substitute = GENIE3(np.transpose(ds_substitute), ntrees=100, regulators='all',
                            gene_names=[str(s) for s in range(np.transpose(ds_substitute).shape[1])])
    # roc_auc_score(gt.flatten(), VIM_substitute.flatten())


    # # Part 4: Use a CNN model to predict clean data distributions and simulate a new dataset
    # ds_noisy = ds_O_L_D_C
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # Load the model
    # model = CNNMultiCTNet().to(device)
    # # model.load_state_dict(torch.load('./model_ds.pth'))
    # x_means, x_stds = np.zeros((9,100)), np.zeros((9,100))

    # x = find_x(ds_noisy.astype(np.float32), x_means, x_stds) 
    # y = model(x)
    # y = y.detach().cpu().numpy()
    # y_shape = (1, 2, 9, 100)
    # y_means = y[0,0,:,:]
    # y_stds = y[0,1,:,:]

    # ds_simulated = simulate_dataset(ds_noisy.astype(np.float32), y_means, y_stds)
    # VIM_simulated = GENIE3(np.transpose(ds_simulated), ntrees=100, regulators='all',
    #                     gene_names=[str(s) for s in range(np.transpose(ds_simulated).shape[1])])
    # # roc_auc_score(gt.flatten(), VIM_simulated.flatten())
    
    # Store results
    auc_clean = roc_auc_score(gt.flatten(), VIM_clean.flatten())
    auc_O = roc_auc_score(gt.flatten(), VIM_O.flatten())
    auc_O_L = roc_auc_score(gt.flatten(), VIM_O_L.flatten())
    auc_O_L_D = roc_auc_score(gt.flatten(), VIM_O_L_D.flatten())
    auc_O_L_D_C = roc_auc_score(gt.flatten(), VIM_O_L_D_C.flatten())
    auc_normalized = roc_auc_score(gt.flatten(), VIM_normalized.flatten())
    auc_substitute = roc_auc_score(gt.flatten(), VIM_substitute.flatten())
    # auc_simulated = roc_auc_score(gt.flatten(), VIM_simulated.flatten())

    # Prepare the log directory
    log_dir = './results/denoise/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_DS{data_info["dataset_id"]}.txt')

    # Write the results to the log file
    with open(log_file, 'w') as f:
        f.write(f'Dataset ID: {data_info["dataset_id"]}\n')
        f.write(f'AUC Clean: {auc_clean}\n')
        f.write(f'AUC O: {auc_O}\n')
        f.write(f'AUC O_L: {auc_O_L}\n')
        f.write(f'AUC O_L_D: {auc_O_L_D}\n')
        f.write(f'AUC O_L_D_C: {auc_O_L_D_C}\n')
        f.write(f'AUC Normalized: {auc_normalized}\n')
        f.write(f'AUC Substitute: {auc_substitute}\n')
        # f.write(f'AUC Simulated: {auc_simulated}\n')
