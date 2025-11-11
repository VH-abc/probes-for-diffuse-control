#!/usr/bin/env python3
"""
Probe analysis for MMLU activations.

This script trains linear probes to test if a model's internal representations
contain information about whether it answered correctly or incorrectly.

Experiments:
1. Linear probe with AUROC evaluation
2. PCA visualization
3. Anomaly detection (correct as normal, incorrect as anomaly)
4. Label corruption robustness
5. AUROC vs training set size
6. Label corruption sweep
"""

import os
import json
import argparse
import numpy as np

import config
from lib.probes import (
    train_linear_probe,
    anomaly_detection,
    measure_auroc_vs_training_size,
    measure_label_corruption_robustness
)
from lib.visualization import (
    plot_roc_curve,
    plot_score_distribution,
    plot_pca,
    plot_auroc_vs_training_size,
    plot_label_corruption_robustness,
    save_training_analysis
)


def load_cached_activations(layer_idx: int, token_position: str, num_examples: int):
    """
    Load cached MMLU activations.

    Args:
        layer_idx: Layer index
        token_position: Token position
        num_examples: Number of examples

    Returns:
        activations, labels, subjects, prompts
    """
    prefix = f"mmlu_layer{layer_idx:02d}_pos-{token_position}_n{num_examples}"
    
    activations_file = os.path.join(config.CACHED_ACTIVATIONS_DIR, f"{prefix}_activations.npy")
    labels_file = os.path.join(config.CACHED_ACTIVATIONS_DIR, f"{prefix}_labels.npy")
    subjects_file = os.path.join(config.CACHED_ACTIVATIONS_DIR, f"{prefix}_subjects.npy")
    prompts_file = os.path.join(config.CACHED_ACTIVATIONS_DIR, f"{prefix}_prompts.npy")

    print(f"\nLoading cached activations:")
    print(f"  Model: {config.MODEL_SHORT_NAME}")
    print(f"  Layer: {layer_idx}, Position: {token_position}, Examples: {num_examples}")
    print(f"  Directory: {config.CACHED_ACTIVATIONS_DIR}")

    if not os.path.exists(activations_file):
        raise FileNotFoundError(f"Activations not found: {activations_file}")

    activations = np.load(activations_file)
    labels = np.load(labels_file)
    subjects = np.load(subjects_file, allow_pickle=True) if os.path.exists(subjects_file) else None
    prompts = np.load(prompts_file, allow_pickle=True) if os.path.exists(prompts_file) else None

    print(f"  Shape: {activations.shape}")
    print(f"  Correct: {np.sum(labels)}/{len(labels)} ({100 * np.sum(labels) / len(labels):.1f}%)")

    return activations, labels, subjects, prompts


def run_probe_analysis(
    layer: int = None,
    token_position: str = None,
    num_examples: int = None,
    skip_experiments: list = None
):
    """
    Run complete probe analysis pipeline.

    Args:
        layer: Layer index
        token_position: Token position
        num_examples: Number of examples
        skip_experiments: List of experiment names to skip
    """
    # Use config defaults
    layer = layer if layer is not None else config.DEFAULT_LAYER
    token_position = token_position or config.DEFAULT_TOKEN_POSITION
    num_examples = num_examples or config.DEFAULT_NUM_EXAMPLES
    skip_experiments = skip_experiments or []

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("\n" + "#" * 80)
    print("# PROBE ANALYSIS")
    print(f"# Model: {config.MODEL_SHORT_NAME}")
    print(f"# Layer: {layer}, Position: {token_position}, Examples: {num_examples}")
    print("#" * 80)

    # Load data
    activations, labels, subjects, prompts = load_cached_activations(
        layer, token_position, num_examples
    )

    # Shuffle for random split
    np.random.seed(config.PROBE_RANDOM_STATE)
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

    fname = f"layer{layer}_pos-{token_position}_n{num_examples}"

    # Experiment 1: Linear Probe
    if "linear_probe" not in skip_experiments:
        print(f"\n{'=' * 60}")
        print(f"Experiment 1: Linear Probe")
        print(f"{'=' * 60}")

        auroc, clf, y_pred_proba, fpr, tpr = train_linear_probe(
            X_train, y_train, X_test, y_test,
            max_iter=config.PROBE_MAX_ITER,
            random_state=config.PROBE_RANDOM_STATE
        )
        print(f"  AUROC: {auroc:.4f}")

        # Save AUROC
        auroc_file = os.path.join(config.RESULTS_DIR, f"auroc_{fname}.json")
        with open(auroc_file, 'w') as f:
            json.dump({'auroc': float(auroc), 'fname': fname}, f, indent=2)

        # Visualizations
        plot_roc_curve(
            fpr, tpr, auroc,
            os.path.join(config.RESULTS_DIR, f"roc_{fname}.png"),
            f"ROC Curve - {fname}"
        )
        plot_score_distribution(
            clf.decision_function(X_test), y_test,
            os.path.join(config.RESULTS_DIR, f"scores_{fname}.png"),
            f"Score Distribution - {fname}"
        )
        save_training_analysis(
            clf, X_train, y_train, train_subjects, train_prompts,
            os.path.join(config.RESULTS_DIR, f"training_analysis_{fname}.txt"),
            fname
        )

    # Experiment 2: PCA Visualization
    if "pca" not in skip_experiments:
        print(f"\n{'=' * 60}")
        print(f"Experiment 2: PCA Visualization")
        print(f"{'=' * 60}")

        plot_pca(
            X_test, y_test,
            os.path.join(config.RESULTS_DIR, f"pca_{fname}.png"),
            f"PCA - {fname}"
        )

    # Experiment 3: Anomaly Detection
    if "anomaly_detection" not in skip_experiments:
        print(f"\n{'=' * 60}")
        print(f"Experiment 3: Anomaly Detection")
        print(f"  (Correct as normal, Incorrect as anomaly)")
        print(f"{'=' * 60}")

        auroc_anomaly, anomaly_scores, fpr_anomaly, tpr_anomaly = anomaly_detection(
            X_train, y_train, X_test, y_test
        )
        print(f"  AUROC: {auroc_anomaly:.4f}")

        plot_roc_curve(
            fpr_anomaly, tpr_anomaly, auroc_anomaly,
            os.path.join(config.RESULTS_DIR, f"roc_anomaly_{fname}.png"),
            f"Anomaly Detection ROC - {fname}"
        )

    # Experiment 4: AUROC vs Training Set Size
    if "auroc_vs_n" not in skip_experiments:
        print(f"\n{'=' * 60}")
        print(f"Experiment 4: AUROC vs Training Set Size")
        print(f"{'=' * 60}")

        results = measure_auroc_vs_training_size(
            activations, labels,
            n_values=[16, 32, 64, 128],
            n_trials=10,
            max_iter=config.PROBE_MAX_ITER,
            random_state=config.PROBE_RANDOM_STATE
        )

        for n, errors in results.items():
            if errors:
                print(f"  N={n}: Error = {np.mean(errors):.4f} ± {np.std(errors):.4f}")

        plot_auroc_vs_training_size(
            results,
            os.path.join(config.RESULTS_DIR, f"auroc_vs_n_{fname}.png")
        )

    # Experiment 5: Label Corruption Robustness
    if "corruption_sweep" not in skip_experiments:
        print(f"\n{'=' * 60}")
        print(f"Experiment 5: Label Corruption Robustness")
        print(f"{'=' * 60}")

        results = measure_label_corruption_robustness(
            activations, labels,
            epsilons=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
            n_trials=5,
            max_iter=config.PROBE_MAX_ITER,
            random_state=config.PROBE_RANDOM_STATE
        )

        for eps, errors in results.items():
            if errors:
                print(f"  ε={eps}: Error = {np.mean(errors):.4f} ± {np.std(errors):.4f}")

        plot_label_corruption_robustness(
            results,
            os.path.join(config.RESULTS_DIR, f"corruption_{fname}.png")
        )

    print(f"\n{'#' * 80}")
    print(f"# Analysis complete!")
    print(f"# Results saved to: {config.RESULTS_DIR}")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe analysis for cached activations")
    parser.add_argument("--layer", type=int, help=f"Layer index (default: {config.DEFAULT_LAYER})")
    parser.add_argument("--position", type=str, choices=["last", "first", "middle", "all"],
                        help=f"Token position (default: {config.DEFAULT_TOKEN_POSITION})")
    parser.add_argument("--num-examples", type=int, help=f"Number of examples (default: {config.DEFAULT_NUM_EXAMPLES})")
    parser.add_argument("--skip", type=str, nargs="+", 
                        choices=["linear_probe", "pca", "anomaly_detection", "auroc_vs_n", "corruption_sweep"],
                        help="Experiments to skip")

    args = parser.parse_args()

    run_probe_analysis(
        layer=args.layer,
        token_position=args.position,
        num_examples=args.num_examples,
        skip_experiments=args.skip
    )

