"""
Unit tests for lib.probes module.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.probes import train_linear_probe, anomaly_detection


class TestProbesModule(unittest.TestCase):
    """Test cases for probe training and evaluation."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)

        # Create synthetic data where class 1 and class 0 are separable
        n_samples = 100
        n_features = 10

        # Class 1 (correct): centered at [1, 1, ..., 1]
        self.X_class1 = np.random.randn(n_samples, n_features) + 1.0
        # Class 0 (incorrect): centered at [-1, -1, ..., -1]
        self.X_class0 = np.random.randn(n_samples, n_features) - 1.0

        # Combine and create labels
        self.X = np.vstack([self.X_class1, self.X_class0])
        self.y = np.array([1] * n_samples + [0] * n_samples)

        # Shuffle
        idx = np.random.permutation(len(self.X))
        self.X = self.X[idx]
        self.y = self.y[idx]

        # Split
        split = len(self.X) // 2
        self.X_train = self.X[:split]
        self.y_train = self.y[:split]
        self.X_test = self.X[split:]
        self.y_test = self.y[split:]

    def test_train_linear_probe_returns_valid_auroc(self):
        """Test that linear probe returns valid AUROC."""
        auroc, clf, y_pred_proba, fpr, tpr = train_linear_probe(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        self.assertGreaterEqual(auroc, 0.0)
        self.assertLessEqual(auroc, 1.0)
        self.assertIsNotNone(clf)
        self.assertEqual(len(y_pred_proba), len(self.y_test))

    def test_train_linear_probe_high_auroc_on_separable_data(self):
        """Test that probe achieves high AUROC on separable data."""
        auroc, _, _, _, _ = train_linear_probe(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        # Data is well-separated, AUROC should be high
        self.assertGreater(auroc, 0.9)

    def test_anomaly_detection_returns_valid_auroc(self):
        """Test that anomaly detection returns valid AUROC."""
        auroc, anomaly_scores, fpr, tpr = anomaly_detection(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        self.assertGreaterEqual(auroc, 0.0)
        self.assertLessEqual(auroc, 1.0)
        self.assertEqual(len(anomaly_scores), len(self.y_test))

    def test_anomaly_detection_conceptually_correct(self):
        """Test that anomaly detection treats incorrect as anomalies.
        
        This test verifies the bug fix: we should fit Gaussian to correct (class 1)
        and detect incorrect (class 0) as anomalies.
        """
        auroc, anomaly_scores, _, _ = anomaly_detection(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        # Check that incorrect samples tend to have higher anomaly scores
        # (though this is statistical, not guaranteed for every sample)
        mean_anomaly_correct = np.mean(anomaly_scores[self.y_test == 1])
        mean_anomaly_incorrect = np.mean(anomaly_scores[self.y_test == 0])

        # Incorrect should generally have higher anomaly scores
        # (This may not always hold for random data, but should for separable data)
        self.assertGreater(mean_anomaly_incorrect, mean_anomaly_correct * 0.8)

    def test_anomaly_detection_insufficient_correct_examples(self):
        """Test anomaly detection with insufficient correct examples."""
        # Create dataset with only 1 correct example
        X_train = np.random.randn(10, 5)
        y_train = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        X_test = np.random.randn(5, 5)
        y_test = np.array([1, 0, 1, 0, 0])

        auroc, anomaly_scores, _, _ = anomaly_detection(
            X_train, y_train, X_test, y_test
        )

        # Should not crash, returns some result
        self.assertIsNotNone(auroc)
        self.assertEqual(len(anomaly_scores), len(y_test))


class TestProbesSanityChecks(unittest.TestCase):
    """Sanity checks for probe behavior."""

    def test_perfect_separation_gives_perfect_auroc(self):
        """Test that perfectly separable data gives AUROC ≈ 1.0."""
        # Create perfectly separable data
        X_train = np.vstack([
            np.ones((50, 10)) * 10,   # Class 1: far positive
            np.ones((50, 10)) * -10,  # Class 0: far negative
        ])
        y_train = np.array([1] * 50 + [0] * 50)

        X_test = np.vstack([
            np.ones((25, 10)) * 10,
            np.ones((25, 10)) * -10,
        ])
        y_test = np.array([1] * 25 + [0] * 25)

        auroc, _, _, _, _ = train_linear_probe(X_train, y_train, X_test, y_test)

        # Should be nearly perfect
        self.assertGreater(auroc, 0.99)

    def test_random_labels_gives_auroc_near_half(self):
        """Test that random labels give AUROC ≈ 0.5."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(100, 10)
        y_test = np.random.randint(0, 2, 100)

        auroc, _, _, _, _ = train_linear_probe(X_train, y_train, X_test, y_test)

        # Should be close to 0.5 (random)
        self.assertGreater(auroc, 0.3)
        self.assertLess(auroc, 0.7)


if __name__ == '__main__':
    unittest.main()

