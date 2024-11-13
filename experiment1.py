import numpy as np
from GENIE3.GENIE3 import *
from sklearn.metrics import roc_auc_score
from utils import gt_benchmark, precision_at_k

def run_experiment_1(dataset_id, num_iterations=5):
    results = []
    
    for i in range(num_iterations):
        # Generate new clean data each time
        sim, expr, expr_clean = experiment(datasets[dataset_id - 1])
        
        # Run GENIE3 on clean data
        VIM_CLEAN = GENIE3(expr_clean.T, nthreads=12, ntrees=100)
        
        # Evaluate performance
        target_file = f'./SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
        gt, rescaled_vim = gt_benchmark(VIM_CLEAN, target_file)
        
        roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
        precision_k = precision_at_k(gt, rescaled_vim, range(1, gt.size))
        
        results.append({
            'iteration': i,
            'roc_auc': roc_score,
            'precision_k': precision_k
        })
    
    return results

# Run experiment 1
exp1_results = run_experiment_1(dataset_id=1)

# Analyze results
roc_scores = [r['roc_auc'] for r in exp1_results]
print(f"Mean ROC AUC: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
