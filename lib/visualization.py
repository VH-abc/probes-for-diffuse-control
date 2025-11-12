"""
Visualization utilities for probe analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    output_path: str,
    title: str = "ROC Curve"
):
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auroc: AUROC score
        output_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "Score Distribution"
):
    """
    Plot distribution of classifier scores.

    Args:
        scores: Classifier scores
        labels: Binary labels (1=correct, 0=incorrect)
        output_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, color='red',
             label='Correct (Class 1)', density=True, edgecolor='black', linewidth=0.5)
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, color='green',
             label='Incorrect (Class 0)', density=True, edgecolor='black', linewidth=0.5)
    plt.xlabel('Classifier Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    output_path: str,
    title: str = "PCA Visualization",
    names: dict = {0: "Incorrect", 1: "Correct"}
):
    """
    Plot PCA projection of activations.

    Args:
        X: Activations (n_samples, n_features)
        y: Labels (n_samples,)
        output_path: Path to save plot
        title: Plot title
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='green', label=f'{names[0]} (0)',
                alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red', label=f'{names[1]} (1)',
                alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_auroc_vs_training_size(
    results: dict,
    output_path: str,
    title: str = "Probe Error vs Training Set Size"
):
    """
    Plot how probe error varies with training set size.

    Args:
        results: Dictionary mapping training sizes to lists of error rates
        output_path: Path to save plot
        title: Plot title
    """
    n_values = sorted(results.keys())
    means = [np.mean(results[n]) if results[n] else np.nan for n in n_values]
    stds = [np.std(results[n]) if results[n] else 0 for n in n_values]

    plt.figure(figsize=(10, 6))
    plt.errorbar(n_values, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xticks(n_values, [str(n) for n in n_values])
    plt.xlabel('Training Set Size (N)', fontsize=12)
    plt.ylabel('Error (1 - AUROC)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_label_corruption_robustness(
    results: dict,
    output_path: str,
    title: str = "Probe Robustness to Label Corruption"
):
    """
    Plot probe robustness to label corruption.

    Args:
        results: Dictionary mapping corruption rates to lists of error rates
        output_path: Path to save plot
        title: Plot title
    """
    epsilons = sorted(results.keys())
    means = [np.mean(results[eps]) if results[eps] else np.nan for eps in epsilons]
    stds = [np.std(results[eps]) if results[eps] else 0 for eps in epsilons]

    plt.figure(figsize=(10, 6))
    plt.errorbar(epsilons, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Label Corruption Rate (ε)', fontsize=12)
    plt.ylabel('Error (1 - AUROC)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_analysis(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects: Optional[np.ndarray],
    prompts: Optional[np.ndarray],
    output_path: str,
    analysis_name: str
):
    """
    Analyze and save information about extreme training examples.

    Args:
        clf: Trained classifier with decision_function method
        X_train: Training activations
        y_train: Training labels
        subjects: MMLU subjects (optional)
        prompts: Question prompts (optional)
        output_path: Path to save analysis
        analysis_name: Name for this analysis
    """
    train_scores = clf.decision_function(X_train)

    # Find extreme examples
    lowest_idx = np.argsort(train_scores)[:3]
    highest_idx = np.argsort(train_scores)[-3:][::-1]
    closest_idx = np.argsort(np.abs(train_scores))[:3]

    lines = [
        f"\n{'=' * 80}",
        f"Training Data Analysis: {analysis_name}",
        f"{'=' * 80}",
        f"\n3 LOWEST SCORING (most confident INCORRECT):",
        f"{'-' * 80}"
    ]

    for i, idx in enumerate(lowest_idx, 1):
        lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, "
                     f"Label: {int(y_train[idx])} ({'Correct' if y_train[idx] else 'Incorrect'})")
        if subjects is not None:
            lines.append(f"   Subject: {subjects[idx]}")
        if prompts is not None:
            lines.append(f"   Prompt: {prompts[idx][:150]}...")

    lines.extend([f"\n3 HIGHEST SCORING (most confident CORRECT):", f"{'-' * 80}"])

    for i, idx in enumerate(highest_idx, 1):
        lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, "
                     f"Label: {int(y_train[idx])} ({'Correct' if y_train[idx] else 'Incorrect'})")
        if subjects is not None:
            lines.append(f"   Subject: {subjects[idx]}")
        if prompts is not None:
            lines.append(f"   Prompt: {prompts[idx][:150]}...")

    lines.extend([f"\n3 CLOSEST TO ZERO (most uncertain):", f"{'-' * 80}"])

    for i, idx in enumerate(closest_idx, 1):
        lines.append(f"\n#{i} - Score: {train_scores[idx]:.4f}, "
                     f"Label: {int(y_train[idx])} ({'Correct' if y_train[idx] else 'Incorrect'})")
        if subjects is not None:
            lines.append(f"   Subject: {subjects[idx]}")
        if prompts is not None:
            lines.append(f"   Prompt: {prompts[idx][:150]}...")

    lines.append(f"\n{'=' * 80}\n")

    # Save to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

