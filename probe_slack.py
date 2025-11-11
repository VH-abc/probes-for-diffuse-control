"""
Probe analysis for MMLU activations.

This script trains linear probes to test if a model's internal representations
contain information about whether it answered correctly or incorrectly.
"""

import os
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from config import CACHED_ACTIVATIONS_DIR, RESULTS_DIR, MODEL_SHORT_NAME

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHED_ACTIVATIONS_DIR, exist_ok=True)

# ============================================================================
# Data Loading
# ============================================================================

def load_mmlu_activations(layer_idx: int, num_examples: int = 1000):
    """Load MMLU cached activations.

    Args:
        layer_idx: Layer index
        num_examples: Number of examples in the cache

    Returns:
        activations: Activation array (n_samples, n_features)
        labels: Binary labels (1=correct, 0=incorrect)
        subjects: MMLU subjects
        prompts: Question prompts
    """
    prefix = f"mmlu_activations_layer_{layer_idx:02d}_n{num_examples}"
    activations_file = os.path.join(CACHED_ACTIVATIONS_DIR, f"{prefix}_activations.npy")
    labels_file = os.path.join(CACHED_ACTIVATIONS_DIR, f"{prefix}_labels.npy")
    subjects_file = os.path.join(CACHED_ACTIVATIONS_DIR, f"{prefix}_subjects.npy")
    prompts_file = os.path.join(CACHED_ACTIVATIONS_DIR, f"{prefix}_prompts.npy")

    print(f"\nLoading activations for {MODEL_SHORT_NAME}")
    print(f"  Layer: {layer_idx}, Examples: {num_examples}")
    print(f"  Directory: {CACHED_ACTIVATIONS_DIR}")

    if not os.path.exists(activations_file):
        raise FileNotFoundError(f"Activations not found: {activations_file}")

    activations = np.load(activations_file)
    labels = np.load(labels_file)
    subjects = np.load(subjects_file, allow_pickle=True) if os.path.exists(subjects_file) else None
    prompts = np.load(prompts_file, allow_pickle=True) if os.path.exists(prompts_file) else None

    print(f"  Shape: {activations.shape}")
    print(f"  Correct: {np.sum(labels)}/{len(labels)} ({100*np.sum(labels)/len(labels):.1f}%)")
    
    return activations, labels, subjects, prompts


# ============================================================================
# Core Probe Functions
# ============================================================================

def train_probe(X_train, y_train, X_test, y_test):
    """Train a logistic regression probe and compute AUROC.
    
    Args:
        X_train: Training activations
        y_train: Training labels
        X_test: Test activations
        y_test: Test labels
        
    Returns:
        auroc: AUROC score
        clf: Trained classifier
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auroc = auc(fpr, tpr)
    
    return auroc, clf


def anomaly_detection(X_train, y_train, X_test, y_test):
    """Gaussian anomaly detection: fit Gaussian to class 0, detect class 1 as anomalies.
    
    Args:
        X_train: Training activations
        y_train: Training labels
        X_test: Test activations
        y_test: Test labels
        
    Returns:
        auroc: AUROC score
        anomaly_scores: Anomaly scores for test set
    """
    # Fit Gaussian to "incorrect" answers (class 0)
    X_normal = X_train[y_train == 0]
    mean = np.mean(X_normal, axis=0)
    cov = np.cov(X_normal, rowvar=False)
    
    # Add regularization for numerical stability
    epsilon = 1e-6
    cov += epsilon * np.eye(cov.shape[0])
    
    try:
        cov_inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        
        if sign <= 0:
            cov += epsilon * 10 * np.eye(cov.shape[0])
            cov_inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
        
        # Compute Mahalanobis distance as anomaly score
        X_centered = X_test - mean
        mahalanobis_sq = np.sum(X_centered @ cov_inv * X_centered, axis=1)
        d = X_test.shape[1]
        log_likelihood = -0.5 * (mahalanobis_sq + logdet + d * np.log(2 * np.pi))
        anomaly_scores = -log_likelihood
        
    except np.linalg.LinAlgError:
        # Fallback to Euclidean distance
        anomaly_scores = np.linalg.norm(X_test - mean, axis=1)
    
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    auroc = auc(fpr, tpr)
    
    return auroc, anomaly_scores


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_roc_curve(fpr, tpr, auroc, fname, title="ROC Curve"):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distribution(scores, labels, fname, title="Score Distribution"):
    """Plot distribution of classifier scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(scores[labels==1], bins=50, alpha=0.6, color='red', 
             label='Correct (Class 1)', density=True, edgecolor='black', linewidth=0.5)
    plt.hist(scores[labels==0], bins=50, alpha=0.6, color='green',
             label='Incorrect (Class 0)', density=True, edgecolor='black', linewidth=0.5)
    plt.xlabel('Classifier Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca(X, y, fname, title="PCA Visualization"):
    """Plot PCA projection of activations."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='green', label='Incorrect (0)', 
                alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', label='Correct (1)',
                alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()


def save_training_analysis(clf, X_train, y_train, subjects, prompts, fname):
    """Analyze and save information about extreme training examples."""
    train_scores = clf.decision_function(X_train)
    
    # Find extreme examples
    lowest_idx = np.argsort(train_scores)[:3]
    highest_idx = np.argsort(train_scores)[-3:][::-1]
    closest_idx = np.argsort(np.abs(train_scores))[:3]
    
    lines = [
        f"\n{'='*80}",
        f"Training Data Analysis: {fname}",
        f"{'='*80}",
        f"\n3 LOWEST SCORING (most confident INCORRECT):",
        f"{'-'*80}"
    ]
    
    for i, idx in enumerate(lowest_idx, 1):
        lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, "
                    f"Label: {int(y_train[idx])} ({'Correct' if y_train[idx] else 'Incorrect'})")
        if subjects is not None:
            lines.append(f"   Subject: {subjects[idx]}")
        if prompts is not None:
            lines.append(f"   Prompt: {prompts[idx][:150]}...")
    
    lines.extend([f"\n3 HIGHEST SCORING (most confident CORRECT):", f"{'-'*80}"])
    
    for i, idx in enumerate(highest_idx, 1):
        lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, "
                    f"Label: {int(y_train[idx])} ({'Correct' if y_train[idx] else 'Incorrect'})")
        if subjects is not None:
            lines.append(f"   Subject: {subjects[idx]}")
        if prompts is not None:
            lines.append(f"   Prompt: {prompts[idx][:150]}...")
    
    lines.extend([f"\n3 CLOSEST TO ZERO (most uncertain):", f"{'-'*80}"])
    
    for i, idx in enumerate(closest_idx, 1):
        lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, "
                    f"Label: {int(y_train[idx])} ({'Correct' if y_train[idx] else 'Incorrect'})")
        if subjects is not None:
            lines.append(f"   Subject: {subjects[idx]}")
        if prompts is not None:
            lines.append(f"   Prompt: {prompts[idx][:150]}...")
    
    lines.append(f"\n{'='*80}\n")
    
    # Save to file
    output_file = f"{RESULTS_DIR}/training_analysis_{fname}.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


# ============================================================================
# Experiment Functions
# ============================================================================

def experiment_basic_probe(X_train, y_train, X_test, y_test, subjects, prompts, fname):
    """Run basic linear probe experiment."""
    print(f"\n{'='*60}")
    print(f"Basic Linear Probe: {fname}")
    print(f"{'='*60}")
    
    auroc, clf = train_probe(X_train, y_train, X_test, y_test)
    print(f"  AUROC: {auroc:.4f}")
    
    # Get predictions
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Visualizations
    plot_roc_curve(fpr, tpr, auroc, f"roc_{fname}", f"ROC Curve - {fname}")
    plot_score_distribution(clf.decision_function(X_test), y_test, 
                          f"density_{fname}", f"Score Distribution - {fname}")
    save_training_analysis(clf, X_train, y_train, subjects, prompts, fname)
    
    return auroc


def experiment_anomaly_detection(X_train, y_train, X_test, y_test, fname):
    """Run Gaussian anomaly detection experiment."""
    print(f"\n{'='*60}")
    print(f"Anomaly Detection: {fname}")
    print(f"{'='*60}")
    
    auroc, anomaly_scores = anomaly_detection(X_train, y_train, X_test, y_test)
    print(f"  AUROC: {auroc:.4f}")
    
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    plot_roc_curve(fpr, tpr, auroc, f"roc_anomaly_{fname}", 
                  f"Anomaly Detection ROC - {fname}")
    
    return auroc


def experiment_label_corruption(X_train, y_train, X_test, y_test, subjects, prompts, fname):
    """Run label corruption robustness experiment."""
    print(f"\n{'='*60}")
    print(f"Label Corruption Experiment: {fname}")
    print(f"{'='*60}")
    
    # Keep first 100 positive examples and all negative examples
    pos_idx = np.where(y_train == 1)[0][:100]
    neg_idx = np.where(y_train == 0)[0]
    
    # Corrupt 10% of negative labels
    np.random.shuffle(neg_idx)
    n_corrupt = int(0.1 * len(neg_idx))
    
    # Build corrupted training set
    keep_idx = np.concatenate([pos_idx, neg_idx])
    X_train_corrupt = X_train[keep_idx]
    y_train_corrupt = y_train[keep_idx].copy()
    
    # Flip first n_corrupt negative examples to positive
    neg_mask = y_train_corrupt == 0
    neg_positions = np.where(neg_mask)[0]
    y_train_corrupt[neg_positions[:n_corrupt]] = 1
    
    # Get corresponding metadata
    subjects_corrupt = subjects[keep_idx] if subjects is not None else None
    prompts_corrupt = prompts[keep_idx] if prompts is not None else None
    
    # Train probe on corrupted data
    auroc = experiment_basic_probe(X_train_corrupt, y_train_corrupt, X_test, y_test,
                                   subjects_corrupt, prompts_corrupt, f"corrupted_{fname}")
    
    return auroc


def experiment_auroc_vs_n(activations, labels, fname, n_trials=10):
    """Measure how AUROC varies with training set size."""
    print(f"\n{'='*60}")
    print(f"AUROC vs Training Set Size: {fname}")
    print(f"{'='*60}")
    
    n_values = [2**i for i in range(4, 8)]  # 16, 32, 64, 128
    results = {n: [] for n in n_values}
    
    for n in n_values:
        print(f"  Testing N={n}...", end=' ')
        for _ in range(n_trials):
            # Random split
            idx = np.random.permutation(len(activations))
            
            X_train = activations[idx[:n]]
            y_train = labels[idx[:n]]
            X_test = activations[idx[n:]]
            y_test = labels[idx[n:]]
            
            # Need both classes in training set
            if len(X_test) > 0 and len(np.unique(y_train)) > 1:
                auroc, _ = train_probe(X_train, y_train, X_test, y_test)
                results[n].append(1 - auroc)  # Plot error rate
        
        if results[n]:
            print(f"Error: {np.mean(results[n]):.4f} ± {np.std(results[n]):.4f}")
    
    # Plot
    means = [np.mean(results[n]) if results[n] else np.nan for n in n_values]
    stds = [np.std(results[n]) if results[n] else 0 for n in n_values]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_values, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xticks(n_values, [str(n) for n in n_values])
    plt.xlabel('Training Set Size (N)', fontsize=12)
    plt.ylabel('Error (1 - AUROC)', fontsize=12)
    plt.title('Probe Error vs Training Set Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/auroc_vs_n_{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()


def experiment_label_corruption_sweep(activations, labels, fname, n_trials=5):
    """Measure robustness to different levels of label corruption."""
    print(f"\n{'='*60}")
    print(f"Label Corruption Sweep: {fname}")
    print(f"{'='*60}")
    
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    n_train = min(200, len(labels) // 2)  # Use half for training
    n_pos = max(10, int(0.1 * n_train))   # Keep 10% positive examples
    
    results = {eps: [] for eps in epsilons}
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    for eps in epsilons:
        print(f"  Testing ε={eps}...", end=' ')
        for _ in range(n_trials):
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            
            # Calculate splits
            n_corrupt = int(eps * n_train)
            n_clean_neg = n_train - n_pos - n_corrupt
            
            if n_clean_neg < 0:
                continue
            
            # Build corrupted training set
            train_pos = pos_idx[:n_pos]
            train_corrupt_neg = neg_idx[:n_corrupt]
            train_clean_neg = neg_idx[n_corrupt:n_corrupt + n_clean_neg]
            
            X_train = np.vstack([
                activations[train_pos],
                activations[train_corrupt_neg],
                activations[train_clean_neg]
            ])
            y_train = np.concatenate([
                np.ones(n_pos),
                np.ones(n_corrupt),  # Corrupted: labeled as 1
                np.zeros(n_clean_neg)
            ])
            
            # Test set: remaining examples
            test_idx = np.concatenate([
                pos_idx[n_pos:],
                neg_idx[n_corrupt + n_clean_neg:]
            ])
            X_test = activations[test_idx]
            y_test = labels[test_idx]
            
            if len(X_test) > 0:
                auroc, _ = train_probe(X_train, y_train, X_test, y_test)
                results[eps].append(1 - auroc)
        
        if results[eps]:
            print(f"Error: {np.mean(results[eps]):.4f} ± {np.std(results[eps]):.4f}")
    
    # Plot
    means = [np.mean(results[eps]) if results[eps] else np.nan for eps in epsilons]
    stds = [np.std(results[eps]) if results[eps] else 0 for eps in epsilons]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(epsilons, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Label Corruption Rate (ε)', fontsize=12)
    plt.ylabel('Error (1 - AUROC)', fontsize=12)
    plt.title('Probe Robustness to Label Corruption', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/label_corruption_{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_full_analysis(layer=13, num_examples=200):
    """Run complete analysis pipeline on MMLU activations.
    
    Args:
        layer: Layer index to analyze
        num_examples: Number of examples to use
    """
    print(f"\n{'#'*80}")
    print(f"# PROBE SLACK ANALYSIS")
    print(f"# Model: {MODEL_SHORT_NAME}")
    print(f"# Layer: {layer}, Examples: {num_examples}")
    print(f"{'#'*80}")
    
    # Load data
    activations, labels, subjects, prompts = load_mmlu_activations(layer, num_examples)
    
    # Shuffle for random split
    idx = np.random.permutation(len(activations))
    activations = activations[idx]
    labels = labels[idx]
    subjects = subjects[idx] if subjects is not None else None
    prompts = prompts[idx] if prompts is not None else None
    
    # Split train/test
    split = len(activations) // 2
    X_train, X_test = activations[:split], activations[split:]
    y_train, y_test = labels[:split], labels[split:]
    train_subjects = subjects[:split] if subjects is not None else None
    train_prompts = prompts[:split] if prompts is not None else None
    
    fname = f"mmlu_layer{layer}_n{num_examples}"
    
    # Run experiments
    experiment_basic_probe(X_train, y_train, X_test, y_test, 
                          train_subjects, train_prompts, fname)
    
    plot_pca(X_test, y_test, f"pca_{fname}", f"PCA - {fname}")
    
    experiment_anomaly_detection(X_train, y_train, X_test, y_test, fname)
    
    experiment_label_corruption(X_train, y_train, X_test, y_test,
                               train_subjects, train_prompts, fname)
    
    experiment_auroc_vs_n(activations, labels, fname)
    
    experiment_label_corruption_sweep(activations, labels, fname)
    
    print(f"\n{'#'*80}")
    print(f"# Analysis complete! Results saved to: {RESULTS_DIR}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    run_full_analysis(layer=13, num_examples=200)
