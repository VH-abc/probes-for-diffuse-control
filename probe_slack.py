import os; import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

CACHE_DIR = "cached_activations"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def reduce_dimension(activations, out_dim):
    pca = PCA(n_components=out_dim)
    return pca.fit_transform(activations)

def load_cached_activations(layer_idx: int, prompt_name: str = "prompt1"):
    output_prefix = f"activations_layer_{layer_idx:02d}_{prompt_name}"
    activations_file = os.path.join(CACHE_DIR, f"{output_prefix}_activations.npy")
    labels_file = os.path.join(CACHE_DIR, f"{output_prefix}_labels.npy")
    topics_file = os.path.join(CACHE_DIR, f"{output_prefix}_topics.npy")
    prompts_file = os.path.join(CACHE_DIR, f"{output_prefix}_prompts.npy")
    print(f"Loading cached activations from {CACHE_DIR}...")
    print(f"  Activations: {activations_file}")
    print(f"  Labels: {labels_file}")
    if not os.path.exists(activations_file):
        raise FileNotFoundError(f"Cached activations not found: {activations_file}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Cached labels not found: {labels_file}")
    activations = np.load(activations_file)
    labels = np.load(labels_file)
    topics = np.load(topics_file, allow_pickle=True) if os.path.exists(topics_file) else None
    prompts = np.load(prompts_file, allow_pickle=True) if os.path.exists(prompts_file) else None
    print(f"  Loaded {activations.shape[0]} samples with {activations.shape[1]} features")
    print(f"  Math samples: {np.sum(labels)}")
    print(f"  Non-math samples: {len(labels) - np.sum(labels)}")
    return activations, labels, topics, prompts

def plot_roc_and_density(Xtrain, Ytrain, Xtest, Ytest, fname, train_topics, train_prompts, base_point=None):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtrain, Ytrain)
    y_pred_logits = clf.decision_function(Xtest)
    fpr, tpr, _ = roc_curve(Ytest, y_pred_logits)
    auroc = auc(fpr, tpr)

    print(f"AUROC for {fname}: {auroc:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve for {fname}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if base_point is not None:
        plt.plot([0,base_point[0],1], [0,base_point[1],1], 'r--', label='labels')
    filename = f"{OUT_DIR}/roc_{fname}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_logits[Ytest==1], bins=50, alpha=0.6, color='red', label='Math', density=True, edgecolor='black', linewidth=0.5)
    plt.hist(y_pred_logits[Ytest==0], bins=50, alpha=0.6, color='green', label='Non-Math', density=True, edgecolor='black', linewidth=0.5)
    plt.xlabel('Classifier Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Score distribution for {fname}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/density_{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()

    """
    Please find the 3 lowest, highest and closest to 0 scoring training data points, and print the training data corresponding to them.
    """
    # Get scores on training data
    train_scores = clf.decision_function(Xtrain)
    
    # Find indices of 3 lowest scores
    lowest_indices = np.argsort(train_scores)[:3]
    
    # Find indices of 3 highest scores
    highest_indices = np.argsort(train_scores)[-3:][::-1]
    
    # Find indices of 3 closest to 0 scores
    closest_to_zero_indices = np.argsort(np.abs(train_scores))[:3]
    
    # Prepare output text
    output_lines = []
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"Training Data Analysis for {fname}")
    output_lines.append(f"{'='*80}")
    
    output_lines.append(f"\n3 LOWEST SCORING TRAINING POINTS:")
    output_lines.append(f"{'-'*80}")
    for i, idx in enumerate(lowest_indices, 1):
        output_lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, Label: {int(Ytrain[idx])} ({'Math' if Ytrain[idx] == 1 else 'Non-Math'})")
        output_lines.append(f"   Topic: {train_topics[idx]}")
        output_lines.append(f"   Prompt: {train_prompts[idx][:100]}...")
        output_lines.append(f"   Prompt: {train_prompts[idx][:100]}...")
    
    output_lines.append(f"\n3 HIGHEST SCORING TRAINING POINTS:")
    output_lines.append(f"{'-'*80}")
    for i, idx in enumerate(highest_indices, 1):
        output_lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, Label: {int(Ytrain[idx])} ({'Math' if Ytrain[idx] == 1 else 'Non-Math'})")
        output_lines.append(f"   Topic: {train_topics[idx]}")
        output_lines.append(f"   Prompt: {train_prompts[idx][:100]}...")
    
    output_lines.append(f"\n3 CLOSEST TO ZERO SCORING TRAINING POINTS:")
    output_lines.append(f"{'-'*80}")
    for i, idx in enumerate(closest_to_zero_indices, 1):
        output_lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, Label: {int(Ytrain[idx])} ({'Math' if Ytrain[idx] == 1 else 'Non-Math'})")
        output_lines.append(f"   Topic: {train_topics[idx]}")
        output_lines.append(f"   Prompt: {train_prompts[idx][:100]}...")
    
    output_lines.append(f"\n{'='*80}\n")
    
    # Print to console
    output_text = '\n'.join(output_lines)
    print(output_text)
    
    # Save to file
    output_file = f"{OUT_DIR}/training_analysis_{fname}.txt"
    with open(output_file, 'w') as f:
        f.write(output_text)
    print(f"Saved training data analysis to {output_file}")
    
    return auroc

def plot_pca_activations(X, Y, fname):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[Y==1, 0], X_pca[Y==1, 1], c='red', label='Math', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.scatter(X_pca[Y==0, 0], X_pca[Y==0, 1], c='green', label='Non-Math', alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
    plt.title(f'PCA of {fname}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pca_{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()

def anomaly_slack(X_train: np.ndarray, train_labels: np.ndarray, X_test: np.ndarray, test_labels: np.ndarray, fname) -> float:
    """
    Gaussian anomaly detection model. Fits a Gaussian to training data with label 0 (non-math),
    then uses negative log-likelihood as anomaly scores for classification.
    
    Args:
        X_train: Training activations (n_samples, n_features)
        train_labels: Training labels (n_samples,) - 0 for non-math, 1 for math
        X_test: Test activations (n_samples, n_features)
        test_labels: Test labels (n_samples,)
        output_dir: Directory to save plots
        layer_idx: Optional layer index for the title
        prompt_name: Name of the prompt template used
        
    Returns:
        AUROC score
    """
    print(f"\nPerforming Gaussian anomaly detection...")
    print(f"Building Gaussian model on non-math (label 0) training samples...")
    
    # Filter training data to only label 0 (non-math terms)
    X_train_normal = X_train[train_labels == 0]
    
    print(f"  Using {X_train_normal.shape[0]} non-math training samples")
    print(f"  Feature dimension: {X_train_normal.shape[1]}")
    
    # Fit a multivariate Gaussian: estimate mean and covariance
    mean = np.mean(X_train_normal, axis=0)
    
    # Compute covariance matrix with regularization for numerical stability
    cov = np.cov(X_train_normal, rowvar=False)
    
    # Add regularization to diagonal (helps with numerical stability)
    epsilon = 1e-6
    cov += epsilon * np.eye(cov.shape[0])
    
    print(f"  Computed mean vector and covariance matrix")
    
    # Compute log probability density for test samples
    # For multivariate Gaussian: log p(x) = -0.5 * [(x-mu)^T * Sigma^-1 * (x-mu) + log|Sigma| + d*log(2*pi)]
    # We'll use the negative log-likelihood as the anomaly score (higher = more anomalous)
    
    try:
        # Compute inverse covariance (precision matrix)
        cov_inv = np.linalg.inv(cov)
        
        # Compute log determinant
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            print("  Warning: Covariance matrix is not positive definite, using regularization")
            cov += epsilon * 10 * np.eye(cov.shape[0])
            cov_inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
        
        # Compute anomaly scores for test data
        # anomaly_score = -log p(x) (higher = more anomalous = more likely math)
        X_centered = X_test - mean
        mahalanobis_sq = np.sum(X_centered @ cov_inv * X_centered, axis=1)
        
        d = X_test.shape[1]
        log_likelihood = -0.5 * (mahalanobis_sq + logdet + d * np.log(2 * np.pi))
        anomaly_scores = -log_likelihood  # Negative log-likelihood as anomaly score
        
        print(f"  Computed anomaly scores for {X_test.shape[0]} test samples")
        
    except np.linalg.LinAlgError as e:
        print(f"  Error inverting covariance matrix: {e}")
        print("  Using simplified Euclidean distance to mean as anomaly score")
        anomaly_scores = np.linalg.norm(X_test - mean, axis=1)
    
    # Compute ROC curve and AUROC
    # test_labels: 0 = non-math (normal), 1 = math (anomaly)
    # anomaly_scores: higher values should correspond to math (label 1)
    fpr, tpr, thresholds = roc_curve(test_labels, anomaly_scores)
    auroc = auc(fpr, tpr)
    
    print(f"Gaussian Anomaly Detection AUROC: {auroc:.4f}")
    
    # Save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Gaussian Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{OUT_DIR}/roc_anomaly_{fname}.png", dpi=150, bbox_inches='tight')
    print(f"Saved Gaussian anomaly detection ROC curve to {OUT_DIR}/roc_{fname}.png")
    plt.close()
    return auroc

def compute_auroc(Xtrain, Ytrain, Xtest, Ytest):
    clf = LogisticRegression(max_iter=100)
    clf.fit(Xtrain, Ytrain)
    y_pred_proba = clf.predict_proba(Xtest)[:, 1]
    fpr, tpr, _ = roc_curve(Ytest, y_pred_proba)
    return auc(fpr, tpr)

def plot_auroc_vs_n(activations, labels, fname, n_trials=10):
    """Plot 1-AUROC vs N where N is number of training samples."""
    n_values = [2**i for i in range(2, 11)]  # 16 to 1024
    results = {n: [] for n in n_values}
    
    for n in n_values:
        for trial in range(n_trials):
            idx = np.random.permutation(len(activations))
            activations_shuffled = activations[idx]
            labels_shuffled = labels[idx]
            
            Xtrain = activations_shuffled[:n]
            Ytrain = labels_shuffled[:n]
            Xtest = activations_shuffled[n:]
            Ytest = labels_shuffled[n:]
            
            if len(Xtest) > 0 and len(np.unique(Ytrain)) > 1:
                auroc = compute_auroc(Xtrain, Ytrain, Xtest, Ytest)
                results[n].append(1 - auroc)
    
    means = [np.mean(results[n]) for n in n_values]
    stds = [np.std(results[n]) for n in n_values]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_values, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xticks(n_values, [str(n) for n in n_values])
    plt.xlabel('N (training samples)', fontsize=12)
    plt.ylabel('1 - AUROC', fontsize=12)
    plt.title('Model Error vs Training Set Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/auroc_vs_n_{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved AUROC vs N plot to {OUT_DIR}/auroc_vs_n_{fname}.png")

def plot_label_corruption(Xtrain, Ytrain, Xtest, Ytest, fname, n_trials=5):
    """Plot 1-AUROC vs epsilon for label corruption experiment."""
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    n_pos = 10
    results = {eps: [] for eps in epsilons}
    
    pos_idx = np.where(Ytrain == 1)[0]
    neg_idx = np.where(Ytrain == 0)[0]
    
    for eps in epsilons:
        for trial in range(n_trials):
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            n_corrupt = int(eps * len(neg_idx))
            n_clean_neg = len(neg_idx) - n_corrupt
            train_pos = pos_idx[:n_pos]
            train_corrupt_neg = neg_idx[:n_corrupt]
            train_clean_neg = neg_idx[n_corrupt:n_corrupt + n_clean_neg]
            Xtrain_corrupted = np.vstack([
                Xtrain[train_pos],
                Xtrain[train_corrupt_neg],
                Xtrain[train_clean_neg]
            ])
            Ytrain_corrupted = np.concatenate([
                np.ones(n_pos),
                np.ones(n_corrupt),  # Mislabeled as positive
                np.zeros(n_clean_neg)
            ])
            
            auroc = compute_auroc(Xtrain_corrupted, Ytrain_corrupted, Xtest, Ytest)
            results[eps].append(1 - auroc)
    
    means = [np.mean(results[eps]) for eps in epsilons]
    stds = [np.std(results[eps]) for eps in epsilons]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(epsilons, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('ε - Label Corruption Rate', fontsize=12)
    plt.ylabel('1 - AUROC', fontsize=12)
    plt.title('Model Error vs Label Corruption', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/label_corruption_{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved label corruption plot to {OUT_DIR}/label_corruption_{fname}.png")

def plot_corruption_heatmap(activations, labels, fname, n_trials=5):
    """Plot lines of 1-AUROC vs epsilon for different N values."""
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    n_values = [2**i for i in range(5, 11)]
    pos_frac = .1
    results_by_n = {n: {eps: [] for eps in epsilons} for n in n_values}
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    for n_train in n_values:
        for eps in epsilons:
            errors = []
            for _ in range(n_trials):
                np.random.shuffle(pos_idx)
                np.random.shuffle(neg_idx)

                n_pos = int(pos_frac * n_train)
                n_corrupt = int(eps * n_train)
                n_clean_neg = n_train - n_corrupt
                
                train_pos = pos_idx[:n_pos]
                train_corrupt_neg = neg_idx[:n_corrupt]
                train_clean_neg = neg_idx[n_corrupt:n_corrupt + n_clean_neg]
                
                Xtrain = np.vstack([ activations[train_pos], activations[train_corrupt_neg], activations[train_clean_neg] ])
                Ytrain = np.concatenate([ np.ones(n_pos), np.ones(n_corrupt), np.zeros(n_clean_neg) ])
                
                test_idx = np.concatenate([pos_idx[n_pos:], neg_idx[n_corrupt + n_clean_neg:]])
                Xtest = activations[test_idx]
                Ytest = labels[test_idx]
                auroc = compute_auroc(Xtrain, Ytrain, Xtest, Ytest)
                errors.append(1 - auroc)
            if errors:
                results_by_n[n_train][eps] = errors
    
    # Create line plot
    plt.figure(figsize=(12, 8))
    for n_train in n_values:
        means = [np.mean(results_by_n[n_train][eps]) if results_by_n[n_train][eps] else np.nan for eps in epsilons]
        stds = [np.std(results_by_n[n_train][eps]) if results_by_n[n_train][eps] else 0 for eps in epsilons]
        plt.errorbar(epsilons, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8, label=f'N={n_train}')
    
    plt.xscale('log')
    plt.xlabel('ε (corruption rate)', fontsize=12)
    plt.ylabel('1 - AUROC', fontsize=12)
    plt.title('Model Error vs Label Corruption for Different Training Set Sizes', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/corruption_heatmap_{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved corruption line plot to {OUT_DIR}/corruption_heatmap_{fname}.png")

def makeplots(prompt, layer, reduce_dim=None):
    activations, labels, topics, prompts = load_cached_activations(layer, prompt)
    idx = np.random.permutation(len(activations))
    activations = activations[idx]
    labels = labels[idx]
    topics = topics[idx]
    prompts = prompts[idx]
    
    Xtrain = activations[:len(activations)//2]
    Xtest = activations[len(activations)//2:]

    if reduce_dim is not None:
        Xtrain = reduce_dimension(Xtrain, reduce_dim)
        Xtest = reduce_dimension(Xtest, reduce_dim)

    Ytrain = labels[:len(labels)//2]
    Ytest = labels[len(labels)//2:]
    
    train_topics = topics[:len(topics)//2] if topics is not None else None
    train_prompts = prompts[:len(prompts)//2] if prompts is not None else None
    
    fname = f"basic_{prompt}_layer{layer}"
    if reduce_dim is not None:
        fname += f"_reducedim{reduce_dim}"
    plot_roc_and_density(Xtrain, Ytrain, Xtest, Ytest, fname, train_topics, train_prompts)
    plot_pca_activations(Xtest, Ytest, fname)
    anomaly_slack(Xtrain, Ytrain, Xtest, Ytest, fname)

    """
    Remove all but 100 examples of class 1
    and flip 10% of the 0 labels to 1s.
    """
    # Keep only 100 examples of class 1
    class1_indices = np.where(Ytrain == 1)[0]
    class0_indices = np.where(Ytrain == 0)[0]
    
    # If there are more than 100 class 1 examples, randomly select 100
    if len(class1_indices) > 100:
        np.random.shuffle(class1_indices)
        class1_indices = class1_indices[:100]
    
    # Keep all class 0 examples
    kept_indices = np.concatenate([class1_indices, class0_indices])
    Xtrain_corrupted = Xtrain[kept_indices]
    Ytrain_corrupted = Ytrain[kept_indices].copy()
    train_topics_corrupted = train_topics[kept_indices] if train_topics is not None else None
    train_prompts_corrupted = train_prompts[kept_indices] if train_prompts is not None else None
    
    # Flip 10% of the 0 labels to 1s
    class0_mask = Ytrain_corrupted == 0
    class0_count = np.sum(class0_mask)
    flip_count = int(0.1 * class0_count)
    class0_positions = np.where(class0_mask)[0]
    np.random.shuffle(class0_positions)
    flip_positions = class0_positions[:flip_count]
    Ytrain_corrupted[flip_positions] = 1
    
    plot_roc_and_density(Xtrain_corrupted, Ytrain_corrupted, Xtest, Ytest, "corrupted_" + fname, train_topics_corrupted, train_prompts_corrupted, base_point=(.1,1))

    print("\nGenerating AUROC vs N plot...")
    plot_auroc_vs_n(activations, labels, fname)
    print("\nGenerating label corruption plot...")
    plot_label_corruption(Xtrain, Ytrain, Xtest, Ytest, fname)
    print("\nGenerating corruption heatmap...")
    plot_corruption_heatmap(activations, labels, fname)


def make_many_plots(prompts, layers, reduce_dims=None):
    """
    Create combined plots for multiple prompts, layers, and reduce_dims.
    
    Args:
        prompts: List of prompt names
        layers: List of layer indices
        reduce_dims: Optional list of dimension reduction values. If None, no reduction is applied.
    """
    if reduce_dims is None:
        reduce_dims = [None]
    
    # Prepare data structures to collect results
    all_configs = []
    
    # Iterate over all combinations
    for prompt in prompts:
        for layer in layers:
            for reduce_dim in reduce_dims:
                config = {
                    'prompt': prompt,
                    'layer': layer,
                    'reduce_dim': reduce_dim
                }
                
                # Load and prepare data
                activations, labels, topics, prompts_data = load_cached_activations(layer, prompt)
                idx = np.random.permutation(len(activations))
                activations = activations[idx]
                labels = labels[idx]
                topics = topics[idx]
                prompts_data = prompts_data[idx]
                
                Xtrain = activations[:len(activations)//2]
                Xtest = activations[len(activations)//2:]
                
                if reduce_dim is not None:
                    Xtrain = reduce_dimension(Xtrain, reduce_dim)
                    Xtest = reduce_dimension(Xtest, reduce_dim)
                
                Ytrain = labels[:len(labels)//2]
                Ytest = labels[len(labels)//2:]
                
                train_topics = topics[:len(topics)//2] if topics is not None else None
                train_prompts = prompts_data[:len(prompts_data)//2] if prompts_data is not None else None
                
                # Generate label for this configuration
                label = f"{prompt}_L{layer}"
                if reduce_dim is not None:
                    label += f"_D{reduce_dim}"
                
                # Store configuration with all needed data
                config['activations'] = activations
                config['labels'] = labels
                config['Xtrain'] = Xtrain
                config['Xtest'] = Xtest
                config['Ytrain'] = Ytrain
                config['Ytest'] = Ytest
                config['train_topics'] = train_topics
                config['train_prompts'] = train_prompts
                config['label'] = label
                
                all_configs.append(config)
    
    # Create combined filename
    prompt_str = "_".join(prompts)
    layer_str = "_".join([str(l) for l in layers])
    if reduce_dims != [None]:
        reduce_str = "_".join([str(d) for d in reduce_dims if d is not None])
        combined_fname = f"combined_{prompt_str}_layers{layer_str}_dims{reduce_str}"
    else:
        combined_fname = f"combined_{prompt_str}_layers{layer_str}"
    
    # 1. Combined Basic ROC Curves
    print("\n" + "="*80)
    print("Creating combined basic ROC curves...")
    print("="*80)
    plt.figure(figsize=(10, 8))
    for config in all_configs:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(config['Xtrain'], config['Ytrain'])
        y_pred_logits = clf.decision_function(config['Xtest'])
        fpr, tpr, _ = roc_curve(config['Ytest'], y_pred_logits)
        auroc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{config['label']} (AUROC={auroc:.3f})", linewidth=2)
        print(f"  {config['label']}: AUROC = {auroc:.4f}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Combined ROC Curves', fontsize=14)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_{combined_fname}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined ROC curves to {OUT_DIR}/roc_{combined_fname}.png")
    
    # 2. Combined Corrupted ROC Curves
    print("\n" + "="*80)
    print("Creating combined corrupted ROC curves...")
    print("="*80)
    plt.figure(figsize=(10, 8))
    for config in all_configs:
        Xtrain = config['Xtrain']
        Ytrain = config['Ytrain']
        Xtest = config['Xtest']
        Ytest = config['Ytest']
        
        # Apply same corruption as in makeplots
        class1_indices = np.where(Ytrain == 1)[0]
        class0_indices = np.where(Ytrain == 0)[0]
        
        if len(class1_indices) > 100:
            np.random.shuffle(class1_indices)
            class1_indices = class1_indices[:100]
        
        kept_indices = np.concatenate([class1_indices, class0_indices])
        Xtrain_corrupted = Xtrain[kept_indices]
        Ytrain_corrupted = Ytrain[kept_indices].copy()
        
        # Flip 10% of the 0 labels to 1s
        class0_mask = Ytrain_corrupted == 0
        class0_count = np.sum(class0_mask)
        flip_count = int(0.1 * class0_count)
        class0_positions = np.where(class0_mask)[0]
        np.random.shuffle(class0_positions)
        flip_positions = class0_positions[:flip_count]
        Ytrain_corrupted[flip_positions] = 1
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtrain_corrupted, Ytrain_corrupted)
        y_pred_logits = clf.decision_function(Xtest)
        fpr, tpr, _ = roc_curve(Ytest, y_pred_logits)
        auroc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{config['label']} (AUROC={auroc:.3f})", linewidth=2)
        print(f"  {config['label']}: AUROC = {auroc:.4f}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.plot([0, 0.1, 1], [0, 1, 1], 'r--', label='Corruption baseline', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Combined Corrupted ROC Curves (100 pos, 10% neg flipped)', fontsize=14)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_corrupted_{combined_fname}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined corrupted ROC curves to {OUT_DIR}/roc_corrupted_{combined_fname}.png")
    
    # 3. Combined AUROC vs N plot
    print("\n" + "="*80)
    print("Creating combined AUROC vs N plot...")
    print("="*80)
    n_values = [2**i for i in range(2, 11)]  # 4 to 1024
    n_trials = 10
    
    plt.figure(figsize=(12, 8))
    for config in all_configs:
        activations = config['activations']
        labels = config['labels']
        results = {n: [] for n in n_values}
        
        for n in n_values:
            for trial in range(n_trials):
                idx = np.random.permutation(len(activations))
                activations_shuffled = activations[idx]
                labels_shuffled = labels[idx]
                
                Xtrain = activations_shuffled[:n]
                Ytrain = labels_shuffled[:n]
                Xtest = activations_shuffled[n:]
                Ytest = labels_shuffled[n:]
                
                if len(Xtest) > 0 and len(np.unique(Ytrain)) > 1:
                    auroc = compute_auroc(Xtrain, Ytrain, Xtest, Ytest)
                    results[n].append(1 - auroc)
        
        means = [np.mean(results[n]) for n in n_values]
        stds = [np.std(results[n]) for n in n_values]
        
        plt.errorbar(n_values, means, yerr=stds, marker='o', capsize=5, 
                    linewidth=2, markersize=8, label=config['label'])
        print(f"  {config['label']}: Completed AUROC vs N")
    
    plt.xscale('log')
    plt.xticks(n_values, [str(n) for n in n_values])
    plt.xlabel('N (training samples)', fontsize=12)
    plt.ylabel('1 - AUROC', fontsize=12)
    plt.title('Combined Model Error vs Training Set Size', fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/auroc_vs_n_{combined_fname}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined AUROC vs N plot to {OUT_DIR}/auroc_vs_n_{combined_fname}.png")
    
    # 4. Combined Label Corruption plot
    print("\n" + "="*80)
    print("Creating combined label corruption plot...")
    print("="*80)
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    n_pos = 10
    n_trials = 5
    
    plt.figure(figsize=(12, 8))
    for config in all_configs:
        Xtrain = config['Xtrain']
        Ytrain = config['Ytrain']
        Xtest = config['Xtest']
        Ytest = config['Ytest']
        
        results = {eps: [] for eps in epsilons}
        pos_idx = np.where(Ytrain == 1)[0]
        neg_idx = np.where(Ytrain == 0)[0]
        
        for eps in epsilons:
            for trial in range(n_trials):
                np.random.shuffle(pos_idx)
                np.random.shuffle(neg_idx)
                n_corrupt = int(eps * len(neg_idx))
                n_clean_neg = len(neg_idx) - n_corrupt
                train_pos = pos_idx[:n_pos]
                train_corrupt_neg = neg_idx[:n_corrupt]
                train_clean_neg = neg_idx[n_corrupt:n_corrupt + n_clean_neg]
                Xtrain_corrupted = np.vstack([
                    Xtrain[train_pos],
                    Xtrain[train_corrupt_neg],
                    Xtrain[train_clean_neg]
                ])
                Ytrain_corrupted = np.concatenate([
                    np.ones(n_pos),
                    np.ones(n_corrupt),
                    np.zeros(n_clean_neg)
                ])
                
                auroc = compute_auroc(Xtrain_corrupted, Ytrain_corrupted, Xtest, Ytest)
                results[eps].append(1 - auroc)
        
        means = [np.mean(results[eps]) for eps in epsilons]
        stds = [np.std(results[eps]) for eps in epsilons]
        
        plt.errorbar(epsilons, means, yerr=stds, marker='o', capsize=5,
                    linewidth=2, markersize=8, label=config['label'])
        print(f"  {config['label']}: Completed label corruption")
    
    plt.xscale('log')
    plt.xlabel('ε - Label Corruption Rate', fontsize=12)
    plt.ylabel('1 - AUROC', fontsize=12)
    plt.title('Combined Model Error vs Label Corruption', fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/label_corruption_{combined_fname}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined label corruption plot to {OUT_DIR}/label_corruption_{combined_fname}.png")
    
    print("\n" + "="*80)
    print("All combined plots completed!")
    print("="*80)

if __name__ == "__main__":
    for prompt in ["prompt1", "prompt2"]:
        makeplots(prompt, 13)
    for prompt in ["prompt1", "prompt2"]:
        for reduce_dim in [20, 100, 1000]:
            makeplots(prompt, 13, reduce_dim=reduce_dim)
    for prompt in ["prompt1", "prompt2"]:
        for layer in [13, 20, 5]:
            makeplots(prompt, layer)

    make_many_plots(["prompt1", "prompt2"], [13], reduce_dims=[20, 100, 1000])
    make_many_plots(["prompt1", "prompt2"], [13])
    make_many_plots(["prompt1", "prompt2"], [13, 20, 5])
