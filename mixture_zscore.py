"""Mixture z-scoring utilities.

Notes:
- `origin_distribution` is expected to be an array/list of labels representing the
  desired class distribution (as in your original code), from which prevalence
  weights are computed.
- For within-subject and cross-subject settings, you can compute `mu/sigma2`
  from a "folded" subset and apply to another subset via `apply_mixture_zscore`.
"""

from __future__ import annotations
import numpy as np


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Class prevalence weights (ratio) from label array."""
    labels = np.asarray(labels)
    n_samples = labels.size
    weights = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        weights[c] = np.sum(labels == c) / n_samples
    return weights


def compute_class_statistics(X: np.ndarray, labels: np.ndarray, n_classes: int):
    """Class-conditional mean/variance for each feature."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    n_features = X.shape[1]
    mu = np.zeros((n_classes, n_features), dtype=float)
    sigma2 = np.zeros((n_classes, n_features), dtype=float)

    for c in range(n_classes):
        class_data = X[labels == c]
        if class_data.size == 0:
            print(f"Warning: No samples found for class {c}.")
            continue
        mu[c] = np.mean(class_data, axis=0)
        sigma2[c] = np.var(class_data, axis=0)

    return mu, sigma2


def mixture_zscore(origin_distribution, train_X: np.ndarray, train_labels: np.ndarray):
    """Compute mixture z-scored `train_X` and return (z_train, weights).

    Parameters
    ----------
    origin_distribution : array-like
        Labels representing the desired class distribution for weighting.
    train_X : (N, F) array
    train_labels : (N,) array
    """
    train_X = np.asarray(train_X)
    train_labels = np.asarray(train_labels)

    n_classes = len(np.unique(train_labels))
    weights = compute_class_weights(np.asarray(origin_distribution), n_classes)

    mu, sigma2 = compute_class_statistics(train_X, train_labels, n_classes)

    weighted_mean = np.dot(weights, mu)
    weighted_var = np.dot(weights, sigma2 + (mu - weighted_mean) ** 2)

    z_train = (train_X - weighted_mean) / np.sqrt(weighted_var)
    return z_train, weights


def cal_mean_std(folded_data: np.ndarray, folded_labels: np.ndarray):
    """Compute (mu, sigma2) on a folded subset."""
    folded_data = np.asarray(folded_data)
    folded_labels = np.asarray(folded_labels)
    n_classes = len(np.unique(folded_labels))
    mu, sigma2 = compute_class_statistics(folded_data, folded_labels, n_classes)
    return mu, sigma2


def apply_mixture_zscore(test_X: np.ndarray, weights: np.ndarray, mu: np.ndarray, sigma2: np.ndarray):
    """Apply mixture z-scoring to `test_X` using provided (weights, mu, sigma2)."""
    test_X = np.asarray(test_X)
    weights = np.asarray(weights)
    mu = np.asarray(mu)
    sigma2 = np.asarray(sigma2)

    weighted_mean = np.dot(weights, mu)
    weighted_var = np.dot(weights, sigma2 + (mu - weighted_mean) ** 2)

    z_test = (test_X - weighted_mean) / np.sqrt(weighted_var)
    return z_test
