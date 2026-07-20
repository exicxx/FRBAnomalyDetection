"""Unsupervised outlier-detection scorers for Catalog 2 morphological features.

Each scorer takes a feature matrix and returns a per-row anomaly score under the shared
convention that a higher score means more anomalous. This is the single canonical,
unit-tested implementation of all six Phase 1 methods; 02_methods_cat2.ipynb,
03_injection_recovery_cat2.ipynb and 04_candidate_validation_cat2.ipynb all score through
these functions rather than keeping their own copies.
"""

import numpy as np
import pandas as pd
from isotree import IsolationForest as ExtendedIsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def lof_scores(X, k=75):
    """Local Outlier Factor score (higher = more locally isolated).

    sklearn returns a negated factor (inliers near -1, outliers more negative); the sign
    is flipped so the score follows the higher-is-more-anomalous convention.
    """
    lof = LocalOutlierFactor(n_neighbors=k).fit(X)
    return -lof.negative_outlier_factor_


def eif_scores(X, ntrees=100, sample_size=256, random_seed=42, ndim=None):
    """Extended Isolation Forest score (higher = more isolated => more anomalous).

    ndim defaults to the full feature count: every split is a random hyperplane through
    all features (full extension), not an axis-aligned cut. predict() already returns the
    standardised [0, 1] anomaly score under the higher-is-more-anomalous convention.
    """
    if ndim is None:
        ndim = X.shape[1]
    eif = ExtendedIsolationForest(
        ndim=ndim, ntrees=ntrees, sample_size=sample_size, random_seed=random_seed
    )
    eif.fit(X)
    return eif.predict(X)


def knn_scores(X, k=20):
    """Mean Euclidean distance to the k nearest neighbours (higher = more isolated)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    return distances[:, 1:].mean(axis=1)   # column 0 is the point itself (distance 0)


def cblof_scores(X, n_clusters=20, alpha=0.9, beta=5.0, random_state=42):
    """Cluster-Based Local Outlier Factor: distance to the nearest large cluster.

    The data is partitioned by KMeans and the clusters are split into large and small by
    the standard alpha/beta rule, scanning from the largest: the boundary falls where the
    large clusters already hold a fraction alpha of all points, or a cluster is beta times
    bigger than the next smaller one. Each point is scored by its distance to the nearest
    large-cluster centroid, so points in small off-population clusters score high. The
    size-weighted variant is omitted, matching the notebook implementation.
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(X)
    sizes = np.bincount(km.labels_, minlength=n_clusters)
    order = np.argsort(sizes)[::-1]
    cumulative = np.cumsum(sizes[order]) / len(X)
    boundary = n_clusters
    for i in range(n_clusters - 1):
        if cumulative[i] >= alpha or sizes[order[i]] / max(sizes[order[i + 1]], 1) >= beta:
            boundary = i + 1
            break
    large_centres = km.cluster_centers_[order[:boundary]]
    return np.linalg.norm(X[:, None, :] - large_centres[None, :, :], axis=2).min(axis=1)


def subspace_scores(X, seed=42, agg="max", n_subspaces=20, k=20):
    """Feature-bagging subspace ensemble (Lazarevic and Kumar 2005).

    A base LOF is run on each of n_subspaces random feature subsets (size between half and
    all-but-one of the features) and the per-subset ranks are aggregated. agg='mean' is the
    standard smooth aggregation; agg='max' preserves the strongest signal from any single
    subspace. Returns the aggregated rank as the score (higher = more anomalous).
    """
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    ranks = np.zeros((len(X), n_subspaces))
    for t in range(n_subspaces):
        size = int(rng.integers(n_features // 2, n_features))
        cols = rng.choice(n_features, size=size, replace=False)
        lof = LocalOutlierFactor(n_neighbors=k).fit(X[:, cols])
        ranks[:, t] = pd.Series(-lof.negative_outlier_factor_).rank(ascending=True).to_numpy()
    return ranks.max(axis=1) if agg == "max" else ranks.mean(axis=1)


def cad_scores(X, context_idx, behaviour_idx, k=50):
    """Conditional anomaly detection: distance to context-neighbours in morphology space.

    context_idx and behaviour_idx are column indices into X selecting the observational
    context (e.g. DM, peak frequency) and the intrinsic morphology features respectively.
    For each row, find its k nearest neighbours in context space, then score it by its mean
    distance, in morphology space, from those neighbours (higher = more anomalous: its
    morphology is atypical for rows sharing its observational context).
    """
    Xc = X[:, context_idx]
    Xb = X[:, behaviour_idx]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
    _, neighbours = nn.kneighbors(Xc)
    neighbours = neighbours[:, 1:]
    return np.array([np.linalg.norm(Xb[i] - Xb[neighbours[i]], axis=1).mean() for i in range(len(X))])
