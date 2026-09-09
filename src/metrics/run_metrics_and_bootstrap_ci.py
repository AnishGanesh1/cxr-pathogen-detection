import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def compute_bootstrap_ci(y_true, y_pred_prob, n_bootstraps=1000, alpha=0.95, seed=42):
    """
    Computes 95% Confidence Intervals for AUC-ROC and F1-Score using non-parametric bootstrapping.
    """
    np.random.seed(seed)
    bootstrapped_aucs = []
    bootstrapped_f1s = []
    
    n_samples = len(y_true)
    y_pred_binary = (y_pred_prob > 0.5).astype(int)
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        auc = roc_auc_score(y_true[indices], y_pred_prob[indices])
        f1 = f1_score(y_true[indices], y_pred_binary[indices], zero_division=0)
        
        bootstrapped_aucs.append(auc)
        bootstrapped_f1s.append(f1)
        
    lower_p = (1.0 - alpha) / 2.0 * 100
    upper_p = (alpha + (1.0 - alpha) / 2.0) * 100
    
    auc_ci = (np.percentile(bootstrapped_aucs, lower_p), np.percentile(bootstrapped_aucs, upper_p))
    f1_ci = (np.percentile(bootstrapped_f1s, lower_p), np.percentile(bootstrapped_f1s, upper_p))
    
    return {
        "mean_auc": np.mean(bootstrapped_aucs),
        "auc_ci_95": auc_ci,
        "mean_f1": np.mean(bootstrapped_f1s),
        "f1_ci_95": f1_ci
    }

if __name__ == "__main__":
    print("Testing 95% Confidence Interval Bootstrapping Module...")
    # Synthetic test sample
    y_true = np.random.randint(0, 2, size=1000)
    y_prob = np.clip(y_true * 0.7 + np.random.normal(0, 0.3, size=1000), 0, 1)
    
    results = compute_bootstrap_ci(y_true, y_prob, n_bootstraps=500)
    print(f"Mean AUC: {results['mean_auc']:.4f} (95% CI: {results['auc_ci_95'][0]:.4f} - {results['auc_ci_95'][1]:.4f})")
    print(f"Mean F1 : {results['mean_f1']:.4f} (95% CI: {results['f1_ci_95'][0]:.4f} - {results['f1_ci_95'][1]:.4f})")
    print("✅ Ready to integrate with full dataset evaluation script!")
