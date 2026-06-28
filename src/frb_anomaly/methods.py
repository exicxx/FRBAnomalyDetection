"""Unsupervised outlier-detection scorers for Catalog 2 morphological features.

Each scorer takes a feature matrix and returns a per-row anomaly score under the shared
convention that a higher score means more anomalous. These are the standalone, reusable
form of the methods used in 02_methods_cat2.ipynb, imported by the candidate-validation
notebook (04). CAD and the Extended Isolation Forest are not included here: CAD needs a
separate context/behaviour split and EIF depends on the optional isotree build, so both
remain in the notebooks.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def lof_scores(X, k=75):
    """Local Outlier Factor score (higher = more locally isolated).

    sklearn returns a negated factor (inliers near -1, outliers more negative); the sign
    is flipped so the score follows the higher-is-more-anomalous convention.
    """
    lof = LocalOutlierFactor(n_neighbors=k).fit(X)
    return -lof.negative_outlier_factor_


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
