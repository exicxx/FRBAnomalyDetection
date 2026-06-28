"""Synthetic anomaly injection and recovery measurement for Phase 1 validation.

The injection-recovery harness (03_injection_recovery_cat2.ipynb) validates the
outlier-detection methods by injecting synthetic anomalies of known geometry into the real
feature table and measuring how well each method recovers them. This module holds the
reusable pieces: a Calibration container of dataset-derived quantities, one generator per
anomaly geometry from the Chandola, Banerjee and Kumar (2009) taxonomy, and the recovery
metric. Each generator is a pure function of (count, strength, rng, calibration); higher
strength makes the intended anomaly more pronounced.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class Calibration:
    """Dataset-derived quantities the anomaly generators depend on.

    Built once from the real feature matrix (see from_features) so each generator stays a
    pure function of its arguments rather than reading module-level state.
    """

    X_real: np.ndarray
    mu: np.ndarray
    d_max: float                 # distance of the most extreme real burst (Type A anchor)
    r_sp: float                  # sp_idx-sp_run correlation (Type D)
    sp_idx_i: int
    sp_run_i: int
    context_idx: tuple           # feature indices treated as observational context (Type E)
    behaviour_idx: tuple         # feature indices treated as morphology (Type E)

    @property
    def n_real(self):
        return self.X_real.shape[0]

    @property
    def n_features(self):
        return self.X_real.shape[1]

    @property
    def Xc(self):
        """Context-feature view of the real data."""
        return self.X_real[:, list(self.context_idx)]

    @property
    def Xb(self):
        """Morphology-feature view of the real data."""
        return self.X_real[:, list(self.behaviour_idx)]


def from_features(X_real, feature_cols, context_cols=("dm_fitb", "peak_freq"), r_sp=-0.885):
    """Build a Calibration from a real feature matrix and its column names."""
    feature_cols = list(feature_cols)
    mu = X_real.mean(axis=0)
    d_max = float(np.linalg.norm(X_real - mu, axis=1).max())
    context_idx = tuple(feature_cols.index(c) for c in context_cols)
    behaviour_idx = tuple(i for i in range(len(feature_cols)) if i not in context_idx)
    return Calibration(
        X_real=X_real, mu=mu, d_max=d_max, r_sp=r_sp,
        sp_idx_i=feature_cols.index("sp_idx"), sp_run_i=feature_cols.index("sp_run"),
        context_idx=context_idx, behaviour_idx=behaviour_idx,
    )


def gen_type_a(n, strength, rng, cal):
    """Type A, global outlier: n independent points, each at strength*d_max from the data
    mean along an independent random direction. Each point is individually globally far."""
    directions = rng.normal(size=(n, cal.n_features))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return cal.mu + strength * cal.d_max * directions + rng.normal(scale=0.05, size=(n, cal.n_features))


def gen_type_b(n, strength, rng, cal):
    """Type B, local outlier: each point is a real burst displaced individually (four
    local-neighbour spacings, scaled by strength) into a nearby low-density pocket along an
    independent direction. The points are spread out and individually isolated, not a clump
    (a clump would form its own micro-cluster and read as normal to LOF)."""
    k_local, push = 20, 4.0
    idx = rng.integers(0, cal.n_real, size=n)
    anchors = cal.X_real[idx].copy()
    nn = NearestNeighbors(n_neighbors=k_local + 1).fit(cal.X_real)
    local_dist, _ = nn.kneighbors(anchors)
    local_scale = local_dist[:, 1:].mean(axis=1, keepdims=True)
    dirs = rng.normal(size=(n, cal.n_features))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return anchors + strength * push * local_scale * dirs


def gen_type_c(n, strength, rng, cal):
    """Type C, collective micro-cluster: a single tight knot (sigma=0.05) of n near-identical
    points embedded near the real distribution, its centre a real burst displaced strength
    local-neighbour spacings into a sparser pocket. The points sit at normal individual
    distances and are anomalous only through their collective over-density."""
    sigma = 0.05
    anchor = cal.X_real[int(rng.integers(0, cal.n_real))]
    nn = NearestNeighbors(n_neighbors=21).fit(cal.X_real)
    local_scale = nn.kneighbors(anchor.reshape(1, -1))[0][0, 1:].mean()
    direction = rng.normal(size=cal.n_features)
    direction /= np.linalg.norm(direction)
    centre = anchor + strength * local_scale * direction
    return centre + rng.normal(scale=sigma, size=(n, cal.n_features))


def gen_type_c_resolvable(n, strength, rng, cal, cluster_size=80):
    """Type C, collective micro-cluster, resolvable variant: a larger knot (cluster_size
    points, chosen to exceed both the density methods' neighbourhood size and the cluster
    method's resolution) at strength*3 local-neighbour spacings off a real anchor, in a
    sparser pocket but not the global tail. n is ignored: a collective anomaly's size is
    part of what defines it."""
    sigma = 0.05
    anchor = cal.X_real[int(rng.integers(0, cal.n_real))]
    nn = NearestNeighbors(n_neighbors=21).fit(cal.X_real)
    local_scale = nn.kneighbors(anchor.reshape(1, -1))[0][0, 1:].mean()
    direction = rng.normal(size=cal.n_features)
    direction /= np.linalg.norm(direction)
    centre = anchor + strength * 3.0 * local_scale * direction
    return centre + rng.normal(scale=sigma, size=(cluster_size, cal.n_features))


def gen_type_d(n, strength, rng, cal):
    """Type D, subspace correlation-break: copy all features from n real bursts, then set
    sp_idx and sp_run to individually normal values whose pairing violates the real
    sp_idx-sp_run correlation. strength interpolates from the normal correlation (0) through
    decorrelated (0.5) to fully inverted (1.0); both marginals stay in the normal range, so
    the point is anomalous only in the relationship, not in global distance."""
    idx = rng.integers(0, cal.n_real, size=n)
    anomalies = cal.X_real[idx].copy()
    sp_idx_vals = rng.normal(0, 1, size=n).clip(-1.5, 1.5)
    expected_sp_run = cal.r_sp * sp_idx_vals
    sp_run_vals = expected_sp_run * (1 - 2 * strength)
    anomalies[:, cal.sp_idx_i] = sp_idx_vals
    anomalies[:, cal.sp_run_i] = np.clip(sp_run_vals, -2.0, 2.0)
    return anomalies


def gen_type_e(n, strength, rng, cal):
    """Type E, contextual: keep each burst's observational context (DM, peak_freq) from a
    real burst and blend its morphology toward that of a donor drawn from a distant context.
    The donor morphology is itself globally in-distribution, so the point is normal in both
    context and marginal morphology and is anomalous only in the morphology-given-context
    relationship. strength sets the blend from the anchor's own morphology (0) to the donor's
    (1)."""
    n_cand = 30
    Xc, Xb = cal.Xc, cal.Xb
    behaviour_idx = list(cal.behaviour_idx)
    idx = rng.integers(0, cal.n_real, size=n)
    anomalies = cal.X_real[idx].copy()
    for k in range(n):
        a = idx[k]
        cand = rng.integers(0, cal.n_real, size=n_cand)
        donor = cand[np.linalg.norm(Xc[cand] - Xc[a], axis=1).argmax()]
        anomalies[k, behaviour_idx] = (1 - strength) * Xb[a] + strength * Xb[donor]
    return anomalies


def make_generators(cal):
    """Bind a Calibration into each generator, returning the
    name -> callable(n, strength, rng) map used by the harness."""
    raw = {
        "A_global": gen_type_a,
        "B_local": gen_type_b,
        "C_collective": gen_type_c,
        "C_separated": gen_type_c_resolvable,
        "D_subspace": gen_type_d,
        "E_contextual": gen_type_e,
    }
    return {name: partial(fn, cal=cal) for name, fn in raw.items()}


DEFAULT_THRESHOLDS = (("top1pct", 0.01), ("top5pct", 0.05), ("top10pct", 0.10))


def measure_recovery(scores, inject_mask, thresholds=DEFAULT_THRESHOLDS):
    """Fraction of injected anomalies ranked in each top-X% tier, per method.

    scores: dict of method -> per-row score array (higher = more anomalous).
    inject_mask: boolean array, True for the injected (synthetic) rows.
    Returns dict of method -> {tier_label: recovery_fraction}.
    """
    inject_mask = np.asarray(inject_mask)
    n_total = len(inject_mask)
    n_inject = int(inject_mask.sum())
    out = {}
    for method, s in scores.items():
        ranks = pd.Series(s).rank(ascending=False, method="first").to_numpy()
        out[method] = {
            label: int(((ranks <= max(1, round(frac * n_total))) & inject_mask).sum()) / n_inject
            for label, frac in thresholds
        }
    return out
