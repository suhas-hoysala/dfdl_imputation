from GENIE3.GENIE3 import *
import sys, os
import numpy as np
import pandas as pd
import scprep
import matplotlib
import json
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

#sys.path.append(os.getcwd())

# path_to_SERGIO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SERGIO'))
# path_to_MAGIC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MAGIC'))
# path_to_SAUCIE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SAUCIE'))
# path_to_scScope = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scScope'))
# path_to_DeepImpute = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deepimpute'))

# if path_to_SERGIO not in sys.path:
#     sys.path.insert(0, path_to_SERGIO)
from SERGIO.SERGIO.sergio import sergio
# if path_to_MAGIC not in sys.path:
#     sys.path.insert(0, path_to_MAGIC)
import MAGIC.magic as magic
# if path_to_SAUCIE not in sys.path:
#     sys.path.insert(0, path_to_SAUCIE)
import SAUCIE.SAUCIE as SAUCIE
# if path_to_scScope not in sys.path:
#     sys.path.insert(0, path_to_scScope)
import scScope.scscope.scscope as DeepImpute
# if path_to_DeepImpute not in sys.path:
#     sys.path.insert(0, path_to_DeepImpute)
import deepimpute.deepimpute as deepimpute

from Pearson.pearson import Pearson

import arboreto as arboreto

from arboreto import algo
import scanpy as sc
import anndata
from arboreto.utils import load_tf_names
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.cli.utils import load_signatures
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize
from pyscenic.export import export2loom
from pyscenic.prune import prune2df
from arboreto.algo import grnboost2
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize
from pyscenic.export import export2loom

from utils import gt_benchmark, reload_modules, delete_modules 
from utils import plot_precisions, precision_at_k

import numpy as np
import subprocess


def run_sergio(input_file, reg_file, ind, n_genes=1200, n_bins=9, n_sc=300, file_extension = ''):
    # Run SERGIO
    if ind == 1:
        n_genes = 100
    if ind == 2:
        n_genes = 400
    sim = sergio(
        number_genes=n_genes, 
        number_bins = n_bins, 
        number_sc = n_sc,
        # In paper
        noise_params = 1,
        # In paper
        decays=0.8, 
        sampling_state=15, 
        noise_type='dpd')
    
    sim.build_graph(input_file_taregts=input_file, input_file_regs=reg_file, shared_coop_state=2)
    sim.simulate()
    
    # Get Expression Data
    expr = sim.getExpressions()
    expr_clean = np.concatenate(expr, axis = 1)
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    
    # Save simulated data variants
    np.save(save_path + '/DS6_clean' + file_extension, expr_clean )
    np.save(save_path + '/DS6_expr' + file_extension, expr)
    cmat_clean = sim.convert_to_UMIcounts(expr)
    cmat_clean = np.concatenate(cmat_clean, axis = 1)
    np.save(save_path + '/DS6_clean_counts' + file_extension, cmat_clean)

    # Add Technical Noise - Steady State Simulations
    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 5, scale = 1)
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.5, scale = 0.7)
    binary_ind = sim.dropout_indicator(expr_O_L, shape = 8, percentile = 45)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
    count_matrix = np.concatenate(count_matrix, axis = 1)
    np.save(save_path + '/DS6_noisy' + file_extension, count_matrix)

def run_saucie(x_path, y_path, ind):
    #reload_modules('tensorflow.compat')
    tf = importlib.import_module('tensorflow.compat.v1')
    #importlib.reload(SAUCIE)
    tf.disable_v2_behavior()
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    print("loading data")
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    print("reset graph")
    tf.reset_default_graph()
    print("Initialize saucie")
    saucie = SAUCIE.SAUCIE(y.shape[1])
    print("Load saucie")
    loadtrain = SAUCIE.Loader(y, shuffle=True)
    print("Train saucie")
    saucie.train(loadtrain, steps=1000)

    loadeval = SAUCIE.Loader(y, shuffle=False)
    # embedding = saucie.get_embedding(loadeval)
    # number_of_clusters, clusters = saucie.get_clusters(loadeval)
    rec_y = saucie.get_reconstruction(loadeval)
    save_str = '/yhat_SAUCIE'
    np.save(save_path + save_str, rec_y)

def run_deepImpute(x_path, y_path, ind):
    #reload_modules('tensorflow.compat')
    importlib.invalidate_caches()
    multinet = importlib.import_module('deepimpute.deepimpute.multinet')
    importlib.reload(multinet)
    tf = importlib.import_module('tensorflow.compat.v1')
    #tf = importlib.import_module('tensorflow')
    tf.init_scope()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    y = np.transpose(np.load(y_path))
    y = pd.DataFrame(y)
    x = np.transpose(np.load(x_path))
    x = pd.DataFrame(x)
    multinet = multinet.MultiNet()
    multinet.fit(y,cell_subset=1,minVMR=0.5)
    imputedData = multinet.predict(y)
    yhat_deepimpute = imputedData.to_numpy()
    save_str = '/yhat_deepImpute'
    np.save(save_path + save_str, yhat_deepimpute)

def run_magic(x_path, y_path, ind):
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    print(x_path, y_path)
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    
    y_hat = scprep.filter.filter_rare_genes(y, min_cells=5)
    y_norm = scprep.normalize.library_size_normalize(y_hat)
    y_norm = scprep.transform.sqrt(y_norm)

    for t_val in [2, 7, 'auto']:
        magic_op = magic.MAGIC(
            # knn=5,
            # knn_max=None,
            # decay=1,
            # Variable changed in paper
            t=t_val,
            n_pca=20,
            # solver="exact",
            # knn_dist="euclidean",
            n_jobs=-1,
            # random_state=None,
            # verbose=1,
        )
        y_hat = magic_op.fit_transform(y_norm, genes='all_genes')
        save_str = '/yhat_MAGIC_t_' + str(t_val)
        np.save(save_path + save_str, y_hat)

def run_scScope(x_path, y_path, ind):
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    DI_model = DeepImpute.train(
          y,
          15,
          use_mask=True,
          batch_size=64,
          max_epoch=1000,
          epoch_per_check=100,
          T=2,
          exp_batch_idx_input=[],
          encoder_layers=[],
          decoder_layers=[],
          learning_rate=0.0001,
          beta1=0.05,
          num_gpus=1)
    latent_code, rec_y, _ = DeepImpute.predict(y, DI_model, batch_effect=[])
    save_str = '/yhat_scScope'
    np.save(save_path + save_str, rec_y)

def run_scenic_py(x_path, y_path, ind):

    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str

    # Load data
    y = np.transpose(np.load(y_path))
    print('y is ', y.shape) #2700 x 1200
    x = np.transpose(np.load(x_path))
    print('x shape ', x.shape) #2700 x 1200
    # Create anndata object
    adata = anndata.AnnData(y)


    gene_names = adata.var_names.tolist()
    print('len of adata varnames', len(adata.var_names))
    print("First few gene names:", gene_names[:5])  # Should be a list of strings
    # Subset adata and tf_names for testing
    subset_genes = adata.var_names[:100]  # Try with the first 100 genes

    tf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  'tfs'))
    tf_fnames = [os.path.join(tf_dir, fname) for fname in os.listdir(tf_dir) if fname.endswith('.txt')]
    tf_names = [tf for fname in tf_fnames for tf in arboreto.utils.load_tf_names(fname)]
    subset_tfs = [tf for tf in tf_names if tf in subset_genes]
    subset_adata = adata[:, subset_genes]
    print('subset adata', subset_adata)
    print('subset genes', subset_genes)
    print('subset tfs', subset_tfs)
    print("Expression data shape:", adata.X.shape)  # Should be (cells, genes), e.g., (2700, 1200)
    print("TF names:", tf_names[:5])  # A list of transcription factors
    print("Gene names:", adata.var_names[:5])  # A list of genes (should match with TFs)
    print(tf_names[:10])  # Should be a list of strings, not arrays or sequences
    print('adata obs names', adata.obs_names)
    print("TF names shape:", np.array(tf_names).shape)
    print("Gene names shape:", np.array(gene_names).shape)

    subset_genes = adata.var_names[:100]  # First 100 genes
    subset_tfs = [tf for tf in tf_names if tf in subset_genes]  # Matching TFs with genes
    subset_adata = adata[:, subset_genes]  # Subset the AnnData object




    adjacencies = grnboost2(expression_data=subset_adata, tf_names=subset_tfs, gene_names=subset_genes, verbose=True)




    adata.var_names = [str(i) for i in range(y.shape[1])]
    print('adata var names', adata.var_names)
    adata.obs_names = [str(i) for i in range(y.shape[0])]
    print('adata obs names', adata.obs_names)
    # Load transcription factors
    tf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  'tfs'))
    tf_fnames = [os.path.join(tf_dir, fname) for fname in os.listdir(tf_dir) if fname.endswith('.txt')]
    tf_names = [tf for fname in tf_fnames for tf in arboreto.utils.load_tf_names(fname)]
    print('tf_names', tf_names)

    # Load ranking databases
    feather_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feather'))
    db_fnames = [os.path.join(feather_dir, fname) for fname in os.listdir(feather_dir) if fname.endswith('.feather')]
    dbs = [RankingDatabase(fname=fname, name=fname.split("/")[-1]) for fname in db_fnames]

    # GRN inference
    print('adj')

    adjacencies = grnboost2(expression_data=adata, tf_names=tf_names, gene_names=gene_names, verbose=True)
    print('modules')
    # Module discovery
    modules = list(modules_from_adjacencies(adjacencies, adata.var_names))

    # Regulon prediction
    df = prune2df(dbs, modules)

    # AUCell
    auc_mtx = aucell(adata, df)

    # Binarize
    print('binarizie') 
    #[name=fname.split("/")[-1]) for fname in db_fnames]

    # GRN inference
    print('adj')

    adjacencies = grnboost2(expression_data=adata, tf_names=tf_names, gene_names=gene_names, verbose=True)
    print('modules')
    # Module discovery
    modules = list(modules_from_adjacencies(adjacencies, adata.var_names))

    # Regulon prediction
    df = prune2df(dbs, modules)

    # AUCell
    auc_mtx = aucell(adata, df)

    # Binarize
    print('binarizie')
    binarized_mtx = binarize(auc_mtx)

    # Save results
    save_str = '/yhat_SCENIC'
    np.save(save_path + save_str, binarized_mtx.X)
    
    
SINCERA_code = """
run_sincera <- function(x_path, y_path, ind) {
    library(SINCERA)
    library(reticulate)
    
    ds_str <- paste0('DS', ind)
    save_path <- file.path('./imputations', ds_str)
    
    y <- t(np$load(y_path))
    x <- t(np$load(x_path))
    
    # Convert numpy arrays to R data frames
    y_df <- as.data.frame(y)
    x_df <- as.data.frame(x)
    
    # Run SINCERA
    rec_y <- sincera_function(y_df)
    
    save_str <- '/yhat_SINCERA.npy'
    np$save(file.path(save_path, save_str), rec_y)
}
"""

SCENIC_CODE = '''
# Load necessary libraries
library(SCENIC)
library(AUCell)
library(GENIE3)
library(RcisTarget)

run_scenic_r <- function(x_path, y_path, ind) {
  # Load data
  y <- t(as.matrix(readRDS(y_path)))  # Load y data
  x <- t(as.matrix(readRDS(x_path)))  # Load x data
  print(dim(y))  # y shape (2700 x 1200)
  print(dim(x))  # x shape (2700 x 1200)
  
  # Subset genes for testing
  subset_genes <- rownames(y)[1:100]  # First 100 genes for testing
  
  # Load transcription factors
  tf_path <- "./tfs/tf_names.txt"  # Path to TF names file
  tf_names <- readLines(tf_path)
  
  # Filter TFs present in subset_genes
  subset_tfs <- intersect(tf_names, subset_genes)
  
  # SCENIC pipeline steps:
  # 1. GRN inference with GENIE3
  print("Running GENIE3 for GRN inference...")
  adjacencies <- GENIE3(as.matrix(y), regulators=subset_tfs)
  
  # 2. Module discovery
  print("Discovering modules...")
  modules <- arboreto::getModules(adjacencies, genes=subset_genes)
  
  # 3. Prune modules using RcisTarget
  print("Running RcisTarget for pruning...")
  db_files <- list.files("./feather", pattern="*.feather", full.names=TRUE)
  df <- pruneModulesUsingRcisTarget(modules, db_files)
  
  # 4. Run AUCell
  print("Running AUCell for regulon activity...")
  auc_mtx <- AUCell::runAUCell(df, as.matrix(y))
  
  # Save results
  saveRDS(auc_mtx, file=paste0("./imputations/DS", ind, "/yhat_SCENIC.rds"))
  
  print("SCENIC pipeline completed!")
}

'''

# Call the SINCERA function using subprocess
def run_sincera(x_path, y_path, ind):
    # Convert .npy files to R-readable format and save them
    x_rds_path = f"./imputations/DS{ind}/x_data.rds"
    y_rds_path = f"./imputations/DS{ind}/y_data.rds"
    
    x_data = np.transpose(np.load(x_path))
    y_data = np.transpose(np.load(y_path))
    
    # Save the data as .rds files
    pd.DataFrame(x_data).to_csv(x_rds_path, index=False)
    pd.DataFrame(y_data).to_csv(y_rds_path, index=False)
    
    # Define the R script content
    r_script_content = SINCERA_code + f"""
    run_sincera("{x_rds_path}", "{y_rds_path}", {ind})
    """
    
    # Write the R script to a temporary file
    r_script_path = f"./run_sincera_{ind}.R"
    with open(r_script_path, "w") as r_script_file:
        r_script_file.write(r_script_content)
    
    # Call the R script using subprocess
    result = subprocess.run(
        ['Rscript', r_script_path],
        capture_output=True,
        text=True
    )
    
    # Print the output and error (if any)
    print("SINCERA run in R is complete.")
    print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    
    # Optionally, remove the temporary R script file
    os.remove(r_script_path)


# Call the R scenic function using subprocess
def run_scenic(x_path, y_path, ind):
    # Convert .npy files to R-readable format and save them
    x_rds_path = f"./imputations/DS{ind}/x_data.rds"
    y_rds_path = f"./imputations/DS{ind}/y_data.rds"
    
    x_data = np.transpose(np.load(x_path))
    y_data = np.transpose(np.load(y_path))
    
    # Save the data as .rds files
    pd.DataFrame(x_data).to_csv(x_rds_path, index=False)
    pd.DataFrame(y_data).to_csv(y_rds_path, index=False)
    
    # Define the R script content
    r_script_content = SCENIC_CODE + f"""
    run_scenic_r("{x_rds_path}", "{y_rds_path}", {ind})
    """
    
    # Write the R script to a temporary file
    r_script_path = f"./run_scenic_{ind}.R"
    with open(r_script_path, "w") as r_script_file:
        r_script_file.write(r_script_content)
    
    # Call the R script using subprocess
    result = subprocess.run(
        ['Rscript', r_script_path],
        capture_output=True,
        text=True
    )
    
    # Print the output and error (if any)
    print("SCENIC run in R is complete.")
    print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    
    # Optionally, remove the temporary R script file
    os.remove(r_script_path)

def run_sincera(x_path, y_path, ind):
    # Convert .npy files to R-readable format and save them
    x_rds_path = f"./imputations/DS{ind}/x_data.rds"
    y_rds_path = f"./imputations/DS{ind}/y_data.rds"
    
    x_data = np.transpose(np.load(x_path))
    y_data = np.transpose(np.load(y_path))
    
    # Save the data as .rds files
    pd.DataFrame(x_data).to_csv(x_rds_path, index=False)
    pd.DataFrame(y_data).to_csv(y_rds_path, index=False)
    
    # Define the R script content
    r_script_content = SINCERA_code + f"""
    run_sincera_r("{x_rds_path}", "{y_rds_path}", {ind})
    """
    
    # Write the R script to a temporary file
    r_script_path = f"./run_sincera_{ind}.R"
    with open(r_script_path, "w") as r_script_file:
        r_script_file.write(r_script_content)
    
    # Call the R script using subprocess
    result = subprocess.run(
        ['Rscript', r_script_path],
        capture_output=True,
        text=True
    )
    
    # Print the output and error (if any)
    print("Sincera run in R is complete.")
    print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    
    # Optionally, remove the temporary R script file
    os.remove(r_script_path)


def run_arboreto(path, roc, precision_recall_k, method_name, target, ind, regs=None):
    if regs is None:
        regs = 'all'
    dataset = np.transpose(np.load(path))
    df = pd.DataFrame(dataset)
    c_names = [str(c) for c in df.columns]
    df.columns = c_names
    #network = algo.grnboost2(expression_data=df, tf_names=regs, verbose=True)
    network = algo.genie3(expression_data=df, tf_names=regs, verbose=True)
    network['TF'] = network['TF'].astype(int)
    network['target'] = network['target'].astype(int)
    num_rows = network['target'].max() + 1
    num_cols = network['TF'].max() + 1
    matrix = np.zeros((num_rows, num_cols))
    ret_dict = {}
    for _, row in network.iterrows():
        matrix[int(row['target']), int(row['TF'])] = row['importance']
    
    gt, rescaled_vim = gt_benchmark(matrix, target)
    if roc:
        roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
        ret_dict['DS' + str(ind) + ' arboreto ' + method_name + ' ROC_AUC'] = float('%.2f'%(roc_score))
    if precision_recall_k:
        k = range(1, gt.size)
        precision_k = precision_at_k(gt, rescaled_vim, k)
        ret_dict['DS' + str(ind) + ' arboreto ' + method_name + ' Precision@k'] = precision_k
    
    return ret_dict

def run_pearson(path, target, roc, precision_recall_k, method_name, ind):
    dataset = np.transpose(np.load(path))
    pearson = Pearson(dataset, '').values
    gt, rescaled_vim = gt_benchmark(pearson, target)
    ret_dict = {}
    if roc:
        roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
        ret_dict['DS' + str(ind) + ' Pearson ' + method_name + ' ROC_AUC'] = float('%.2f'%(roc_score))
    if precision_recall_k:
        k = range(1, gt.size)
        precision_k = precision_at_k(gt, rescaled_vim, k)
        ret_dict['DS' + str(ind) + ' Pearson ' + method_name + ' Precision@K'] = precision_k
    print(ret_dict)
    return ret_dict

def run_simulations(datasets, sergio=True, saucie=True, scScope=True, deepImpute=True, magic=True, genie=True, pearson=False, arboreto=True, roc=True, scenic=False, sincera=False, precision_recall_k=True, run_with_regs=False):
    target_file = ''
    regs_path = ''
    results = {}
    count_methods = 2
    for i in tqdm(datasets):
        individual_results = {}
        if i == 1:   
            target_file = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
            regs_path = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
        elif i == 2:
            target_file = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Interaction_cID_5.txt'
            regs_path = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt'
        else:
            target_file = './SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt'
            regs_path = './SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'
        ds_str = 'DS' + str(i)
        save_path = './imputations/' + ds_str

        if sergio:
            print(f"---> Running SERGIO on DS{i}")
            run_sergio(target_file, regs_path, i)
            count_methods += 1
        
        if saucie:
            print(f"---> Running SAUCIE on DS{i}")
            run_saucie(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 1
        
        if scScope:
            print(f"---> Running scScope on DS{i}")
            run_scScope(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 1

        if deepImpute:
            print(f"---> Running DeepImpute on DS{i}")
            run_deepImpute(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 1

        if scenic:
            print(f"---> Running Scenic on DS{i}")
            run_scenic(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 1

        if sincera:
            print(f"---> Running Sincera on DS{i}")
            run_sincera(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 1

        if magic:
            print(f"---> Running MAGIC on DS{i}")
            run_magic(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 3

        y = np.transpose(np.load(save_path + '/DS6_45.npy'))
        x = np.transpose(np.load(save_path + '/DS6_clean.npy'))

        if arboreto:
            reg_file = None
            if i == 1:
                reg_file = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
            elif i == 2:
                reg_file = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt'
            else:
                reg_file = 'SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'
            master_regs = pd.read_table(reg_file, header=None, sep=',')
            master_regs = master_regs[0].values.astype(int).astype(str).tolist()

            regulators = []
            regulator_file = open(target_file, 'r')
            lines = regulator_file.readlines()
            for line in lines:
                row = line.split(',')
                num_regs_row = int(float(row[1]))
                if num_regs_row != 0:
                    for i in range(2, num_regs_row + 2):
                        regulators.append(str(int(float(row[i]))))
            regs = list(set(regulators))
            regs = [i for i in regs if i not in master_regs]

            # if sergio:
            print(f"---> Running arboreto on Clean Data for DS{i}")
            arboreto_results = run_arboreto(save_path + '/DS6_clean.npy', roc, precision_recall_k, 'Clean', target_file, i, regs)
            individual_results.update(arboreto_results)
            print(f"---> Running arboreto on Noisy Data for DS{i}")
            arboreto_results = run_arboreto(save_path + '/DS6_45.npy', roc, precision_recall_k, 'Noisy', target_file, i, regs)
            individual_results.update(arboreto_results)
            if saucie:
                print(f"---> Running arboreto on SAUCIE Data for DS{i}")
                arboreto_results = run_arboreto(save_path + '/yhat_SAUCIE.npy', roc, precision_recall_k, 'SAUCIE', target_file, i, regs)
                individual_results.update(arboreto_results)
            if scScope:
                print(f"---> Running arboreto on scScope Data for DS{i}")
                arboreto_results = run_arboreto(save_path + '/yhat_scScope.npy', roc, precision_recall_k, 'scScope', target_file, i, regs)
                individual_results.update(arboreto_results)
            if deepImpute:
                print(f"---> Running arboreto on DeepImpute Data for DS{i}")
                arboreto_results = run_arboreto(save_path + '/yhat_deepImpute.npy', roc, precision_recall_k, 'DeepImpute', target_file, i, regs)
                individual_results.update(arboreto_results)
            if magic:
                print(f"---> Running arboreto on MAGIC t=2 Data for DS{i}")
                arboreto_results = run_arboreto(save_path + '/yhat_MAGIC_t_2.npy', roc, precision_recall_k, 'MAGIC t=2', target_file, i, regs)
                individual_results.update(arboreto_results)
                print(f"---> Running arboreto on MAGIC t=7 Data for DS{i}")
                arboreto_results = run_arboreto(save_path + '/yhat_MAGIC_t_7.npy', roc, precision_recall_k, 'MAGIC t=7', target_file, i, regs)
                individual_results.update(arboreto_results)
                print(f"---> Running arboreto on MAGIC t=default Data for DS{i}")
                arboreto_results = run_arboreto(save_path + '/yhat_MAGIC_t_auto.npy', roc, precision_recall_k, 'MAGIC t=default', target_file, i, regs)
                individual_results.update(arboreto_results)
        
        if pearson:
            print(f"---> Running Pearson on Clean Data for DS{i}")
            pearson_results = run_pearson(save_path + '/DS6_clean.npy', target_file, roc, precision_recall_k, 'Clean', i)
            individual_results.update(pearson_results)
            print(f"---> Running Pearson on Noisy Data for DS{i}")
            pearson_results = run_pearson(save_path + '/DS6_45.npy', target_file, roc, precision_recall_k, 'Noisy', i)
            individual_results.update(pearson_results)
            if saucie:
                print(f"---> Running Pearson on SAUCIE Data for DS{i}")
                pearson_results = run_pearson(save_path + '/yhat_SAUCIE.npy', target_file, roc, precision_recall_k, 'SAUCIE', i)
                individual_results.update(pearson_results)
            if scScope:
                print(f"---> Running Pearson on scScope Data for DS{i}")
                pearson_results = run_pearson(save_path + '/yhat_scScope.npy', target_file, roc, precision_recall_k, 'scScope', i)
                individual_results.update(pearson_results)
            if deepImpute:
                print(f"---> Running Pearson on DeepImpute Data for DS{i}")
                pearson_results = run_pearson(save_path + '/yhat_deepImpute.npy', target_file, roc, precision_recall_k, 'DeepImpute', i)
                individual_results.update(pearson_results)
            if magic:
                print(f"---> Running Pearson on MAGIC t=2 Data for DS{i}")
                pearson_results = run_pearson(save_path + '/yhat_MAGIC_t_2.npy', target_file, roc, precision_recall_k, 'MAGIC t=2', i)
                individual_results.update(pearson_results)
                print(f"---> Running Pearson on MAGIC t=7 Data for DS{i}")
                pearson_results = run_pearson(save_path + '/yhat_MAGIC_t_7.npy', target_file, roc, precision_recall_k, 'MAGIC t=7', i)
                individual_results.update(pearson_results)
                print(f"---> Running Pearson on MAGIC t=default Data for DS{i}")
                pearson_results = run_pearson(save_path + '/yhat_MAGIC_t_auto.npy', target_file, roc, precision_recall_k, 'MAGIC t=default', i)
                individual_results.update(pearson_results)

        if genie:
            # get true regulator genes from SERGIO data
            reg_file = None
            if i == 1:
                reg_file = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
            elif i == 2:
                reg_file = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt'
            else:
                reg_file = 'SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'
            master_regs = pd.read_table(reg_file, header=None, sep=',')
            master_regs = master_regs[0].values.astype(int).astype(str).tolist()

            regulators = []
            regulator_file = open(target_file, 'r')
            lines = regulator_file.readlines()
            for line in lines:
                row = line.split(',')
                num_regs_row = int(float(row[1]))
                if num_regs_row != 0:
                    for i in range(2, num_regs_row + 2):
                        regulators.append(str(int(float(row[i]))))
            regs = list(set(regulators))
            regs = [i for i in regs if i not in master_regs]

            # Run GENIE3 on Clean Data
            print(f"---> Running GENIE3 on Clean Data for DS{i}")
            gene_names = [str(i) for i in range(x.shape[1])]
            if not run_with_regs:
                regs = None
                gene_names = None

            VIM_CLEAN = GENIE3(x, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)        
            gt, rescaled_vim = gt_benchmark(VIM_CLEAN, target_file)
            if roc:
                roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                individual_results['DS' + str(i) + ' GENIE3 Clean ROC_AUC'] = float('%.2f'%(roc_score))
            if precision_recall_k:
                k = range(1, gt.size)
                precision_k = precision_at_k(gt, rescaled_vim, k)
                individual_results['DS' + str(i) + ' GENIE3 Clean Precision@k'] = precision_k

            # Run GENIE3 on Noisy Data
            print(f"---> Running GENIE3 on Noisy Data for DS{i}")
            gene_names = [str(i) for i in range(y.shape[1])]
            VIM_NOISY = GENIE3(y, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)       
            gt, rescaled_vim = gt_benchmark(VIM_NOISY, target_file)
            if roc:
                roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                individual_results['DS' + str(i) + ' GENIE3 Noisy ROC_AUC'] = float('%.2f'%(roc_score))
            if precision_recall_k:
                k = range(1, gt.size)
                precision_k = precision_at_k(gt, rescaled_vim, k)
                individual_results['DS' + str(i) + ' GENIE3 Noisy Precision@k'] = precision_k

            # Run GENIE3 on SAUCIE Data
            if saucie:
                y_hat_saucie = np.load(save_path + '/yhat_SAUCIE.npy')

                print(f"---> Running GENIE3 on SAUCIE Data for DS{i}")
                gene_names = [str(i) for i in range(y_hat_saucie.shape[1])]
                VIM_SAUCIE = GENIE3(y_hat_saucie, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
                gt, rescaled_vim = gt_benchmark(VIM_SAUCIE, target_file)
                # np.save(save_path + '/VIM_SAUCIE.npy', rescaled_vim)
                # np.save(save_path + '/gt_SAUCIE.npy', gt)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['DS' + str(i) + ' GENIE3 SAUCIE ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['DS' + str(i) + ' GENIE3 SAUCIE Precision@k'] = precision_k
            
            # Run GENIE3 on scScope Data
            if scScope:
                y_hat_scscope = np.load(save_path + '/yhat_scScope.npy')
                y_hat_scScope  = y_hat_scscope.copy()
                y_hat_scScope[y_hat_scScope == 0] = 1e-5

                print(f"---> Running GENIE3 on scScope Data for DS{i}")
                gene_names = [str(i) for i in range(y_hat_scscope.shape[1])]
                VIM_scScope = GENIE3(y_hat_scScope, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
                gt, rescaled_vim = gt_benchmark(VIM_scScope, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['DS' + str(i) + ' GENIE3 scScope ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['DS' + str(i) + ' GENIE3 scScope Precision@k'] = precision_k

            # Run GENIE3 on DeepImpute Data
            if deepImpute:
                y_hat_deepImpute = np.load(save_path + '/yhat_deepImpute.npy')

                print(f"---> Running GENIE3 on DeepImpute Data for DS{i}")
                gene_names = [str(i) for i in range(y_hat_deepImpute.shape[1])]
                VIM_deepImpute = GENIE3(y_hat_deepImpute, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
                gt, rescaled_vim = gt_benchmark(VIM_deepImpute, target_file)
                np.save(save_path + '/VIM_deepImpute.npy', rescaled_vim)
                np.save(save_path + '/gt_deepImpute.npy', gt)
                print("saved deepimpute files")
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['DS' + str(i) + ' GENIE3 DeepImpute ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['DS' + str(i) + ' GENIE3 DeepImpute Precision@k'] = precision_k

            # Run GENIE3 on MAGIC Data
            if magic:
                y_hat_magic_t2 = np.load(save_path + '/yhat_MAGIC_t_2.npy')
                y_hat_magic_t7 = np.load(save_path + '/yhat_MAGIC_t_7.npy')
                y_hat_magic_t_auto = np.load(save_path + '/yhat_MAGIC_t_auto.npy')

                print(f"---> Running GENIE3 on MAGIC t=2 for DS{i}")
                gene_names = [str(i) for i in range(y_hat_magic_t2.shape[1])]
                VIM_MAGIC = GENIE3(y_hat_magic_t2, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
                gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['DS' + str(i) + ' GENIE3 MAGIC t=2 ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['DS' + str(i) + ' GENIE3 MAGIC t=2 Precision@k'] = precision_k

                print(f"---> Running GENIE3 on MAGIC t=7 for DS{i}")
                gene_names = [str(i) for i in range(y_hat_magic_t7.shape[1])]
                VIM_MAGIC = GENIE3(y_hat_magic_t7, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
                gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['DS' + str(i) + ' GENIE3 MAGIC t=7 ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['DS' + str(i) + ' GENIE3 MAGIC t=7 Precision@k'] = precision_k
                
                print(f"---> Running GENIE3 on MAGIC t=default for DS{i}")
                gene_names = [str(i) for i in range(y_hat_magic_t_auto.shape[1])]
                VIM_MAGIC = GENIE3(y_hat_magic_t_auto, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
                gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['DS' + str(i) + ' GENIE3 MAGIC t=default ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['DS' + str(i) + ' GENIE3 MAGIC t=default Precision@k'] = precision_k
        # write individual results to JSON file
        with open(save_path + '/precision_recall_data.json', 'w') as fp:
            json.dump(individual_results, fp)
    return #, count_methods

def create_correlation_plots(datasets):
    for i in tqdm(datasets):
        print(f"---> Calculating correlations for data from DS{i}")
        ds_str = 'DS' + str(i)
        save_path = './imputations/' + ds_str

        # Load saved data
        y = np.transpose(np.load(save_path + '/DS6_45.npy'))
        x = np.transpose(np.load(save_path + '/DS6_clean.npy'))
        
        # Load MAGIC imputed data
        y_hat_magic_t2 = np.load(save_path + '/yhat_MAGIC_t_2.npy')
        y_hat_magic_t7 = np.load(save_path + '/yhat_MAGIC_t_7.npy')
        y_hat_magic_t_auto = np.load(save_path + '/yhat_MAGIC_t_auto.npy')

        # Create correlation dataframes
        x_corr = pd.DataFrame(x).corr()
        y_corr = pd.DataFrame(y).corr()
        t2_corr = pd.DataFrame(y_hat_magic_t2).corr()
        t7_corr = pd.DataFrame(y_hat_magic_t7).corr()
        t_auto_corr = pd.DataFrame(y_hat_magic_t_auto).corr()
        
        # Create subplots
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle(f'Correlation plots for DS{i}')
        axs[0, 0].stairs(*np.histogram(x_corr, bins = 100, density=True), fill=True, color="green")
        axs[0, 0].set_xlim(-1, 1)
        axs[0, 0].set_title("Clean")

        axs[0, 1].stairs(*np.histogram(y_corr, bins = 100, density=True), fill=True, color="green")
        axs[0, 1].set_xlim(-1, 1)
        axs[0, 1].set_title("Noisy")

        axs[1, 0].stairs(*np.histogram(t2_corr, bins = 100, density=True), fill=True, color="green")
        axs[1, 0].set_xlim(-1, 1)
        axs[1, 0].set_title("Imputed; t=2")

        axs[1, 1].stairs(*np.histogram(t7_corr, bins = 100, density=True), fill=True, color="green")
        axs[1, 1].set_xlim(-1, 1)
        axs[1, 1].set_title("Imputed; t=7")

        axs[2, 0].stairs(*np.histogram(t_auto_corr, bins = 100, density=True), fill=True, color="green")
        axs[2, 0].set_xlim(-1, 1)
        axs[2, 0].set_title("Imputed; t=default")

        for ax in axs.flat:
            ax.set(xlabel = 'Corr. Coeff.', ylabel="Density (%)")
        # for ax in axs.flat:
        #     ax.label_outer()
        fig.tight_layout(pad=2.0)
        plt.show()
    return