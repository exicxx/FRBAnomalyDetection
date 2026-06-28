"""Tests for src/frb_anomaly/injection.py."""

import numpy as np

from frb_anomaly.injection import (
    gen_type_a,
    gen_type_b,
    gen_type_c,
    gen_type_c_resolvable,
    gen_type_d,
    gen_type_e,
    make_generators,
    measure_recovery,
)


def _mean_pairwise_distance(points):
    """Mean Euclidean distance over all pairs of rows."""
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(len(points), k=1)
    return dist[iu].mean()


# ---------------------------------------------------------------------------
# Calibration / from_features
# ---------------------------------------------------------------------------

def test_from_features_resolves_indices(calibration):
    # context_cols default to (dm_fitb, peak_freq) -> indices 0 and 6 in the fixture columns.
    assert calibration.context_idx == (0, 6)
    assert calibration.sp_idx_i == 4
    assert calibration.sp_run_i == 5
    assert calibration.n_features == 8


# ---------------------------------------------------------------------------
# generator output shapes
# ---------------------------------------------------------------------------

def test_point_generators_return_requested_count(calibration, rng):
    for gen in (gen_type_a, gen_type_b, gen_type_c, gen_type_d, gen_type_e):
        out = gen(20, 1.0, rng, calibration)
        assert out.shape == (20, calibration.n_features)


def test_resolvable_cluster_ignores_count_and_is_large(calibration, rng):
    # A collective anomaly's size is part of its definition, so n is ignored and the knot is
    # fixed at 80 points (larger than the density methods' neighbourhood size).
    out = gen_type_c_resolvable(20, 1.0, rng, calibration)
    assert out.shape == (80, calibration.n_features)


# ---------------------------------------------------------------------------
# geometry properties
# ---------------------------------------------------------------------------

def test_type_b_is_spread_not_a_clump(calibration, rng):
    # Regression guard for the 2026-06-27 bug: Type B once dropped all points at one shared
    # centre, making a tight clump (a Type C geometry) that LOF reads as normal. The fix
    # places each point individually, so Type B must be spread out while Type C stays tight.
    b = gen_type_b(20, 1.0, rng, calibration)
    c = gen_type_c(20, 1.0, rng, calibration)
    assert _mean_pairwise_distance(c) < 0.5
    assert _mean_pairwise_distance(b) > 1.0


def test_type_d_inverts_correlation_within_normal_marginals(calibration, rng):
    cal = calibration
    out = gen_type_d(50, 1.0, rng, cal)
    sp_idx, sp_run = out[:, cal.sp_idx_i], out[:, cal.sp_run_i]
    # marginals stay within the clamped normal range (the anomaly is quiet, not extreme)
    assert np.abs(sp_idx).max() <= 1.5 + 1e-9
    assert np.abs(sp_run).max() <= 2.0 + 1e-9
    # the real correlation is negative (r_sp < 0); at full strength it is inverted to positive
    assert np.corrcoef(sp_idx, sp_run)[0, 1] > 0


def test_type_e_preserves_real_context(calibration, rng):
    cal = calibration
    out = gen_type_e(20, 1.0, rng, cal)
    real_context = {tuple(np.round(r, 9)) for r in cal.Xc}
    injected_context = out[:, list(cal.context_idx)]
    assert all(tuple(np.round(r, 9)) in real_context for r in injected_context)


def test_generators_are_deterministic_for_a_seed(calibration):
    a = gen_type_a(10, 1.0, np.random.default_rng(3), calibration)
    b = gen_type_a(10, 1.0, np.random.default_rng(3), calibration)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# make_generators
# ---------------------------------------------------------------------------

def test_make_generators_exposes_all_geometries(calibration):
    gens = make_generators(calibration)
    assert set(gens) == {"A_global", "B_local", "C_collective", "C_separated", "D_subspace", "E_contextual"}
    # the bound callables take (n, strength, rng) without an explicit calibration
    out = gens["A_global"](5, 1.0, np.random.default_rng(0))
    assert out.shape == (5, calibration.n_features)


# ---------------------------------------------------------------------------
# measure_recovery
# ---------------------------------------------------------------------------

def test_measure_recovery_perfect_when_injected_rank_highest():
    scores = {"m": np.array([10.0, 9.0, 8.0, 1.0, 0.0])}
    mask = np.array([True, True, False, False, False])
    rec = measure_recovery(scores, mask, thresholds=(("top40", 0.4),))
    assert rec["m"]["top40"] == 1.0


def test_measure_recovery_zero_when_injected_rank_lowest():
    scores = {"m": np.array([0.0, 1.0, 8.0, 9.0, 10.0])}
    mask = np.array([True, True, False, False, False])
    rec = measure_recovery(scores, mask, thresholds=(("top40", 0.4),))
    assert rec["m"]["top40"] == 0.0


def test_measure_recovery_fraction_in_unit_interval():
    rng = np.random.default_rng(0)
    scores = {"m": rng.normal(size=100)}
    mask = np.zeros(100, dtype=bool)
    mask[:10] = True
    rec = measure_recovery(scores, mask)
    assert all(0.0 <= v <= 1.0 for v in rec["m"].values())
