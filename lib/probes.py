"""
Probe training and evaluation utilities.
"""

import warnings
import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

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
) -> Tuple[float, "LogisticRegression", np.ndarray, np.ndarray, np.ndarray]:
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
    # Lazy import (only load sklearn when actually training probes)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auroc = auc(fpr, tpr)

    return auroc, clf, y_pred_proba, fpr, tpr


class ResidualMLPClassifier:
    """
    MLP with residual (skip) connections for binary classification.
    Uses PyTorch for flexibility with skip connections.
    Supports multiple hidden layers with skip connection for each layer.
    """
    def __init__(self, hidden_layer_sizes, max_iter=1000, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes if isinstance(hidden_layer_sizes, (tuple, list)) else (hidden_layer_sizes,)
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.input_size = None
        
    def fit(self, X, y):
        """Train the model."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Detect GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"    Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
        
        self.input_size = X.shape[1]
        
        # Define ResNet-style MLP with skip connections for each layer
        class ResNetMLP(nn.Module):
            def __init__(self, input_size, hidden_layer_sizes):
                super().__init__()
                self.hidden_layer_sizes = hidden_layer_sizes
                
                # Build layers
                self.layers = nn.ModuleList()
                self.batch_norms = nn.ModuleList()
                self.skip_projections = nn.ModuleList()
                
                prev_size = input_size
                for hidden_size in hidden_layer_sizes:
                    # Main layer
                    self.layers.append(nn.Linear(prev_size, hidden_size))
                    self.batch_norms.append(nn.BatchNorm1d(hidden_size))
                    
                    # Skip connection projection (if dimensions differ)
                    if prev_size != hidden_size:
                        self.skip_projections.append(nn.Linear(prev_size, hidden_size))
                    else:
                        self.skip_projections.append(None)
                    
                    prev_size = hidden_size
                
                # Output layer
                self.output_layer = nn.Linear(prev_size, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                out = x
                
                # Process each hidden layer with skip connection
                for i, (layer, bn, skip_proj) in enumerate(zip(self.layers, self.batch_norms, self.skip_projections)):
                    # Store input for skip connection
                    residual = out
                    
                    # Main path
                    out = layer(out)
                    out = bn(out)
                    
                    # Skip connection with projection if needed
                    if skip_proj is not None:
                        residual = skip_proj(residual)
                    
                    # Add skip connection
                    out = out + residual
                    
                    # Activation and dropout
                    out = self.relu(out)
                    out = self.dropout(out)
                
                # Output layer
                out = self.output_layer(out)
                return out
        
        self.model = ResNetMLP(self.input_size, self.hidden_layer_sizes).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        arch_str = " → ".join([f"Input({self.input_size})"] + [f"Hidden({h})" for h in self.hidden_layer_sizes] + ["Output(1)"])
        print(f"    Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        print(f"    Architecture: {arch_str}")
        print(f"    Hidden layers: {len(self.hidden_layer_sizes)}, Skip connections: {len(self.hidden_layer_sizes)}")
        print(f"    Training samples: {len(X)}, Validation: {int(0.1 * len(X))}")
        
        # Convert to PyTorch tensors and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        
        # Split into train and validation
        n_val = int(0.1 * len(X))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train_t = X_tensor[train_idx]
        y_train_t = y_tensor[train_idx]
        X_val_t = X_tensor[val_idx]
        y_val_t = y_tensor[val_idx]
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_epoch = 0
        
        print(f"\n    Starting training (max {self.max_iter} epochs, early stopping patience={patience})...")
        print(f"    {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Status':<20}")
        print(f"    {'-'*8} {'-'*12} {'-'*12} {'-'*20}")
        
        for epoch in range(self.max_iter):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            
            # Validation every 10 epochs or last epoch
            if epoch % 10 == 0 or epoch == self.max_iter - 1:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                
                # Status message
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    status = "✓ New best!"
                else:
                    patience_counter += 1
                    status = f"No improve ({patience_counter}/{patience})"
                
                print(f"    {epoch:<8} {train_loss:<12.6f} {val_loss:<12.6f} {status:<20}")
                
                if patience_counter >= patience:
                    print(f"\n    Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                    break
        else:
            print(f"\n    Reached max iterations ({self.max_iter} epochs)")
        
        print(f"    Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"    Training complete!\n")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        import torch
        import torch.nn.functional as F
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            # Return as [P(class=0), P(class=1)] for each sample
            return np.column_stack([1 - probs, probs])


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_layer_sizes: tuple,
    max_iter: int = 1000,
    random_state: int = 42
) -> Tuple[float, object, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train an MLP (Multi-Layer Perceptron) probe with skip connections and compute AUROC.
    
    Uses a ResNet-style architecture with:
    - Skip connection for each hidden layer
    - Supports multiple hidden layers (e.g., (12000,), (12000, 6000), (12000, 12000, 6000))
    - Batch normalization and dropout for regularization

    Args:
        X_train: Training activations (n_samples, n_features)
        y_train: Training labels (n_samples,) - binary (1=correct, 0=incorrect)
        X_test: Test activations
        y_test: Test labels
        hidden_layer_sizes: Tuple of hidden layer sizes 
                          - (12000,) for single layer (default)
                          - (12000, 12000) for two layers
                          - (12000, 6000, 3000) for three layers with decreasing size
        max_iter: Maximum iterations for training
        random_state: Random seed

    Returns:
        auroc: AUROC score
        clf: Trained MLP classifier
        y_pred_proba: Predicted probabilities for test set
        fpr: False positive rates
        tpr: True positive rates
    """
    # Lazy import
    from sklearn.metrics import roc_curve, auc
    
    clf = ResidualMLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state
    )
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
    # Lazy import
    from sklearn.metrics import roc_curve, auc
    
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
        high = int(np.floor(np.log2(len(activations)))) - 1
        n_values = [2**i for i in range(4, high+1)] + [int(len(activations)*0.9)]
        n_values = list(set(n_values))
        n_values.sort()

    results = {n: [] for n in n_values}

    for n in n_values:
        print(f"Training probes with {n} examples")
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
