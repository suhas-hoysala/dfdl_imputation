import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from new_experiments import run_simulation
from concurrent.futures import ProcessPoolExecutor
from IPython.display import display, clear_output
from consolidated_runs import run_simulations

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'baselines'))
sys.path.append(os.path.join(os.getcwd(), 'metrics'))
sys.path.append(os.path.join(os.getcwd(), 'prev_methods', 'clustering'))
sys.path.append(os.path.join(os.getcwd(), 'prev_methods', 'reconstruct_grn'))

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def run_experiment():
    outputs = []
    ret_df = None
    for dataset in [2]:
        if not os.path.exists(f"./zero_imputation_experiments/DS{dataset}/"):
            os.makedirs(f"./zero_imputation_experiments/DS{dataset}/")
        # Run for first iteration to prevent race condition
        res = run_simulations(
            dataset=dataset,
            sergio=True,
            saucie=True, 
            scScope=True, 
            deepImpute=True, 
            magic=True, 
            genie=True,
            arboreto=False,
            pearson=False,
            roc=True,
            precision_recall_k=False,
            run_with_regs=False,
            iteration=0
        )
        clear_output()
        if ret_df is None:
            ret_df = pd.DataFrame(columns=res.keys())
        new_df = pd.DataFrame([res], columns=res.keys())
        ret_df = pd.concat([ret_df, new_df], ignore_index=True)
        #write to temp file
        ret_df.to_csv("zero_imputation_experiments/imputation_results.csv", index=False)
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(1, 30):
                futures.append(executor.submit(run_simulation, 
                        dataset=dataset,
                        sergio=(i == 0),
                        saucie=True, 
                        scScope=True, 
                        deepImpute=True, 
                        magic=True, 
                        genie=True,
                        arboreto=False,
                        pearson=False,
                        roc=True,
                        precision_recall_k=False,
                        run_with_regs=False,
                        iteration=i
                    ))
                clear_output()
            for future in tqdm(futures):
                res = future.result()
                clear_output(wait=True)
                if ret_df is None:
                    ret_df = pd.DataFrame(columns=res.keys())
                new_df = pd.DataFrame([res], columns=res.keys())
                ret_df = pd.concat([ret_df, new_df], ignore_index=True)
                #write to temp file
                ret_df.to_csv("zero_imputation_experiments/imputation_results.csv", index=False)
    return


run_experiment()

# ### Other Imputation Methods Experimentation

import os
for dataset in [1,2,3]:
    if not os.path.exists(f"./zero_imputation_experiments/DS{dataset}/"):
        os.makedirs(f"./zero_imputation_experiments/DS{dataset}/")
    if not os.path.exists(f"./zero_imputation_experiments/DS{dataset}/DS6_noisy.npy"):
        res = run_simulation(
            dataset=dataset,
            sergio=True,
            saucie=False, 
            scScope=False, 
            deepImpute=False, 
            magic=False, 
            genie=False,
            arboreto=False,
            pearson=False,
            roc=False,
            precision_recall_k=False,
            run_with_regs=False,
            iteration=0
        )
        clear_output()

 
# scVI
from experiment_utils import run_scvi
import numpy as np
import pandas as pd

def fetch_target_regs(dataset):
    if dataset == 1:   
        target_file = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
        regs_path = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
    elif dataset == 2:
        target_file = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Interaction_cID_5.txt'
        regs_path = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt'
    else:
        target_file = './SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt'
        regs_path = './SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'
    return target_file, regs_path    

def scvi_impute():
    ret_df = None
    for dataset in [1, 2, 3]:
        save_path = f"./zero_imputation_experiments/DS{dataset}/"
        y = np.load(save_path + "/DS6_noisy.npy")
        target_file, regs_path = fetch_target_regs(dataset)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(8):
                futures.append(executor.submit(run_scvi, 
                    data=y, 
                    save_path=save_path, 
                    it=i, 
                    file_extension=f"_iter{i}",
                    target_file=target_file
                ))
                clear_output()
            for future in tqdm(futures):
                vim, it = future.result()
                res = {
                    "dataset": dataset,
                    "method": "scvi",
                    "roc": vim,
                    "iteration": it }
                if ret_df is None:
                    ret_df = pd.DataFrame(columns=res.keys())
                new_df = pd.DataFrame([res], columns=res.keys())
                ret_df = pd.concat([ret_df, new_df], ignore_index=True)
                ret_df.to_csv("zero_imputation_experiments/scvi_imputation_results.csv", index=False)  


scvi_impute()

# knn-smoothing

from experiment_utils import run_knn

def run_smoothing():
    ret_df = None
    for dataset in [1,2,3]:
        save_path = f"./zero_imputation_experiments/DS{dataset}/"
        y = np.load(save_path + "/DS6_noisy.npy")
        target_file, regs_path = fetch_target_regs(dataset)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(8):
                futures.append(executor.submit(run_knn, 
                    data=y,
                    k=32,
                    save_path=save_path, 
                    it=i, 
                    file_extension=f"_iter{i}",
                    target_file=target_file
                ))
                clear_output()
            for future in tqdm(futures):
                vim, it = future.result()
                res = {
                    "dataset": dataset,
                    "method": "knn",
                    "roc": vim,
                    "iteration": it }
                if ret_df is None:
                    ret_df = pd.DataFrame(columns=res.keys())
                new_df = pd.DataFrame([res], columns=res.keys())
                ret_df = pd.concat([ret_df, new_df], ignore_index=True)
                ret_df.to_csv("zero_imputation_experiments/knn_imputation_results.csv", index=False)  


run_smoothing()

