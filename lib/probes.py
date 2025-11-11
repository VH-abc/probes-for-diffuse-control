"""
Probe training and evaluation utilities.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from typing import Tuple

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning)


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42
) -> Tuple[float, LogisticRegression, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train a logistic regression probe and compute AUROC.

    Args:
        X_train: Training activations (n_samples, n_features)
        y_train: Training labels (n_samples,) - binary (1=correct, 0=incorrect)
        X_test: Test activations
        y_test: Test labels
        max_iter: Maximum iterations for logistic regression
        random_state: Random seed

    Returns:
        auroc: AUROC score
        clf: Trained classifier
        y_pred_proba: Predicted probabilities for test set
        fpr: False positive rates
        tpr: True positive rates
    """
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auroc = auc(fpr, tpr)

    return auroc, clf, y_pred_proba, fpr, tpr


def anomaly_detection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gaussian anomaly detection: fit Gaussian to correct answers (class 1),
    detect incorrect answers (class 0) as anomalies.

    This is the conceptually correct formulation: we model "correct" as normal
    and detect "incorrect" as anomalies.

    Args:
        X_train: Training activations
        y_train: Training labels (1=correct, 0=incorrect)
        X_test: Test activations
        y_test: Test labels

    Returns:
        auroc: AUROC score
        anomaly_scores: Anomaly scores for test set
        fpr: False positive rates
        tpr: True positive rates
    """
    # Fit Gaussian to "correct" answers (class 1) - these are the "normal" examples
    X_normal = X_train[y_train == 1]
    
    if len(X_normal) < 2:
        print("Warning: Not enough correct examples for anomaly detection")
        # Return random scores
        anomaly_scores = np.random.random(len(X_test))
        fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
        auroc = auc(fpr, tpr)
        return auroc, anomaly_scores, fpr, tpr
    
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
        print("Warning: Using Euclidean distance fallback for anomaly detection")
        anomaly_scores = np.linalg.norm(X_test - mean, axis=1)

    # For AUROC calculation: higher anomaly score should predict class 0 (incorrect)
    # So we want to predict y_test correctly where 0 is the positive class (anomaly)
    # We need to invert: use anomaly_scores to predict "incorrect" (0)
    fpr, tpr, _ = roc_curve(y_test == 0, anomaly_scores)
    auroc = auc(fpr, tpr)

    return auroc, anomaly_scores, fpr, tpr


def evaluate_probe_with_corrupted_labels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    corruption_rate: float = 0.1,
    max_positive_examples: int = 100,
    max_iter: int = 1000,
    random_state: int = 42
) -> float:
    """
    Evaluate probe robustness with corrupted training labels.

    Args:
        X_train: Training activations
        y_train: Training labels
        X_test: Test activations
        y_test: Test labels
        corruption_rate: Fraction of negative labels to flip to positive
        max_positive_examples: Maximum positive examples to keep
        max_iter: Maximum iterations for logistic regression
        random_state: Random seed

    Returns:
        auroc: AUROC score on clean test set
    """
    # Keep limited positive examples and all negative examples
    pos_idx = np.where(y_train == 1)[0][:max_positive_examples]
    neg_idx = np.where(y_train == 0)[0]

    # Corrupt some negative labels
    np.random.shuffle(neg_idx)
    n_corrupt = int(corruption_rate * len(neg_idx))

    # Build corrupted training set
    keep_idx = np.concatenate([pos_idx, neg_idx])
    X_train_corrupt = X_train[keep_idx]
    y_train_corrupt = y_train[keep_idx].copy()

    # Flip first n_corrupt negative examples to positive
    neg_mask = y_train_corrupt == 0
    neg_positions = np.where(neg_mask)[0]
    y_train_corrupt[neg_positions[:n_corrupt]] = 1

    # Train probe on corrupted data
    auroc, _, _, _, _ = train_linear_probe(
        X_train_corrupt, y_train_corrupt, X_test, y_test, max_iter, random_state
    )

    return auroc


def measure_auroc_vs_training_size(
    activations: np.ndarray,
    labels: np.ndarray,
    n_values: list = None,
    n_trials: int = 10,
    max_iter: int = 1000,
    random_state: int = 42
) -> dict:
    """
    Measure how AUROC varies with training set size.

    Args:
        activations: All activations
        labels: All labels
        n_values: List of training set sizes to test
        n_trials: Number of trials per size
        max_iter: Maximum iterations for logistic regression
        random_state: Base random seed

    Returns:
        Dictionary mapping training sizes to lists of error rates (1 - AUROC)
    """
    if n_values is None:
        n_values = [2**i for i in range(4, 8)]  # 16, 32, 64, 128

    results = {n: [] for n in n_values}

    for n in n_values:
        for trial in range(n_trials):
            # Random split with different seed for each trial
            np.random.seed(random_state + trial)
            idx = np.random.permutation(len(activations))

            X_train = activations[idx[:n]]
            y_train = labels[idx[:n]]
            X_test = activations[idx[n:]]
            y_test = labels[idx[n:]]

            # Need both classes in training set
            if len(X_test) > 0 and len(np.unique(y_train)) > 1:
                auroc, _, _, _, _ = train_linear_probe(
                    X_train, y_train, X_test, y_test, max_iter, random_state
                )
                results[n].append(1 - auroc)  # Store error rate

    return results


def measure_label_corruption_robustness(
    activations: np.ndarray,
    labels: np.ndarray,
    epsilons: list = None,
    n_trials: int = 5,
    n_train: int = 200,
    n_pos_frac: float = 0.1,
    max_iter: int = 1000,
    random_state: int = 42
) -> dict:
    """
    Measure robustness to different levels of label corruption.

    Args:
        activations: All activations
        labels: All labels
        epsilons: List of corruption rates to test
        n_trials: Number of trials per epsilon
        n_train: Training set size
        n_pos_frac: Fraction of training set to be positive examples
        max_iter: Maximum iterations for logistic regression
        random_state: Base random seed

    Returns:
        Dictionary mapping corruption rates to lists of error rates
    """
    if epsilons is None:
        epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    n_train = min(n_train, len(labels) // 2)
    n_pos = max(10, int(n_pos_frac * n_train))

    results = {eps: [] for eps in epsilons}

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    for eps in epsilons:
        for trial in range(n_trials):
            np.random.seed(random_state + trial)
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
                auroc, _, _, _, _ = train_linear_probe(
                    X_train, y_train, X_test, y_test, max_iter, random_state
                )
                results[eps].append(1 - auroc)

    return results

