"""Tests for src/frb_anomaly/methods.py."""

import numpy as np

from frb_anomaly.methods import (
    cad_scores,
    cblof_scores,
    eif_scores,
    knn_scores,
    lof_scores,
    subspace_scores,
)


def _planted_in_top(scores, n_outliers):
    """True if the n_outliers planted rows (the last ones) are all in the top n_outliers."""
    top = set(np.argsort(scores)[::-1][:n_outliers])
    planted = set(range(len(scores) - n_outliers, len(scores)))
    return planted <= top


# ---------------------------------------------------------------------------
# shape and convention
# ---------------------------------------------------------------------------

def test_scores_return_one_value_per_row(synthetic_features):
    X, _ = synthetic_features
    for score in (lof_scores(X), knn_scores(X), cblof_scores(X), subspace_scores(X)):
        assert score.shape == (len(X),)


# ---------------------------------------------------------------------------
# each method ranks obvious global outliers at the top (higher = more anomalous)
# ---------------------------------------------------------------------------

def test_lof_recovers_global_outliers(synthetic_features):
    X, k = synthetic_features
    assert _planted_in_top(lof_scores(X), k)


def test_knn_recovers_global_outliers(synthetic_features):
    X, k = synthetic_features
    assert _planted_in_top(knn_scores(X), k)


def test_cblof_recovers_global_outliers(synthetic_features):
    X, k = synthetic_features
    assert _planted_in_top(cblof_scores(X), k)


def test_subspace_max_recovers_global_outliers(synthetic_features):
    X, k = synthetic_features
    assert _planted_in_top(subspace_scores(X, agg="max"), k)


def test_eif_recovers_global_outliers(synthetic_features):
    X, k = synthetic_features
    assert _planted_in_top(eif_scores(X), k)


def test_cad_recovers_global_outliers(synthetic_features):
    X, k = synthetic_features
    # Arbitrary context/behaviour split; the planted outliers are displaced on every
    # feature, so they read as anomalous regardless of which columns play which role.
    context_idx, behaviour_idx = [0, 1], [2, 3, 4, 5]
    assert _planted_in_top(cad_scores(X, context_idx, behaviour_idx), k)


# ---------------------------------------------------------------------------
# subspace aggregation modes and determinism
# ---------------------------------------------------------------------------

def test_subspace_agg_modes_both_return_scores(synthetic_features):
    X, _ = synthetic_features
    assert subspace_scores(X, agg="max").shape == subspace_scores(X, agg="mean").shape == (len(X),)


def test_subspace_is_deterministic_for_a_seed(synthetic_features):
    X, _ = synthetic_features
    assert np.allclose(subspace_scores(X, seed=1), subspace_scores(X, seed=1))


def test_subspace_seed_changes_result(synthetic_features):
    X, _ = synthetic_features
    assert not np.allclose(subspace_scores(X, seed=1), subspace_scores(X, seed=2))
