import numpy as np
from sklearn.metrics import roc_auc_score
from GENIE3 import GENIE3
import matplotlib.pyplot as plt

def apply_sparsity(data, sparsity_level):
    mask = np.random.rand(*data.shape) < sparsity_level
    sparse_data = data.copy()
    sparse_data[mask] = 0
    return sparse_data

# Main execution
if __name__ == "__main__":
    # Load your data and true network here
    # For example:
    # data = np.load('your_expression_data.npy')
    # true_network = np.load('your_true_network.npy')
    
    sparsity_levels = np.linspace(0, 0.9, 10)
    results = []

    for sparsity in sparsity_levels:
        sparse_data = apply_sparsity(x, sparsity)
        VIM = GENIE3(sparse_data, nthreads=12, ntrees=100)
        auc = roc_auc_score(gt.flatten(), VIM.flatten())
        results.append((sparsity, auc))
    
    sparsities, aucs = zip(*results)
    plt.plot(sparsities, aucs)
    plt.xlabel('Sparsity Level')
    plt.ylabel('AUC Score')
    plt.title('GENIE3 Performance vs Sparsity')
    plt.show()


