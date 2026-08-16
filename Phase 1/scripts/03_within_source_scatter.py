"""
Empirical floor on the internal homogeneity of a real FRB sub-population.

The Type C collective injection (``gen_type_c`` in ``src/frb_anomaly/injection.py``)
places twenty near-identical points at a scatter of sigma = 0.05 in the scaled feature
space. No detector in the Phase 1 sweep recovers that geometry at any injected strength.
Read on its own, that null describes only the resolution of the detectors.

This script converts it into a statement about the data by measuring how tightly real FRB
sub-populations actually cluster. Repeating sources are the only genuine sub-populations
available: every sub-burst carrying the same ``repeater_name`` originates from one object.
The within-source spread of those sources is therefore an empirical floor on the
homogeneity of any real population in these parameters.

The features are standardised, so the retained population has a standard deviation of
exactly one in every column. A within-source standard deviation of 0.50 means the source
varies half as much as the whole catalogue; the injected knot at 0.05 varies a twentieth
as much.

Three limits apply to the comparison and belong alongside any reported figure.

1. The measured spreads contain catalogue fit error in addition to genuine burst-to-burst
   variation, whereas the synthetic knot contains none. The reported values are therefore
   upper limits on the astrophysical spread.
2. The comparison addresses internal homogeneity alone. The synthetic knot is displaced
   into a sparse region by construction, whereas repeating sources occupy a dense one, and
   density-based detectors score on local contrast. The result does not establish that a
   real sub-population would have been recovered.
3. Dispersion measure is a line-of-sight quantity and is close to constant within a source,
   so its within-source spread is near zero and depresses the eight-feature mean. The
   summary is reported both with and without it.

Reads ``data/processed/phase_1/catalog2_features_scaled.csv``. Fits nothing and writes
nothing.
"""

from pathlib import Path

import pandas as pd

# parents[0] = scripts/, parents[1] = Phase 1/, parents[2] = project root, where data/ lives.
SCALED = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "phase_1" / "catalog2_features_scaled.csv"
)

FEATURES = [
    "dm_fitb", "width_fitb", "flux", "fluence",
    "sp_idx", "sp_run", "peak_freq", "bandwidth",
]

# Line-of-sight rather than morphological; excluded from the second summary. See limit 3.
CONTEXT_FEATURE = "dm_fitb"

# Scatter of the Type C collective injection, from gen_type_c in injection.py.
INJECTED_SCATTER = 0.05

# Minimum sub-bursts required for a source to enter the summary. A standard deviation from
# two points reduces to the absolute difference over root two, which is dominated by noise
# and biased low whenever the pair happens to fall close together. Five is the working
# threshold; MIN_BURSTS_SWEEP below reports the sensitivity to it.
MIN_BURSTS = 5
MIN_BURSTS_SWEEP = [2, 3, 5, 8, 10, 20]


def per_source_spread(frame, min_bursts):
    """Return the per-source, per-feature standard deviation for sufficiently sampled sources.

    Sources are identified by ``repeater_name``. Only sources contributing at least
    ``min_bursts`` retained sub-bursts are included. The returned frame is indexed by source
    and carries one column per feature.
    """
    sizes = frame.groupby("repeater_name").size()
    kept = sizes[sizes >= min_bursts].index
    subset = frame[frame["repeater_name"].isin(kept)]
    return subset.groupby("repeater_name")[FEATURES].std(ddof=1), len(subset)


def main():
    scaled = pd.read_csv(SCALED)
    repeaters = scaled[scaled["is_repeater"].astype(bool)]

    print(f"retained sub-bursts            {len(scaled)}")
    print(f"repeater sub-bursts            {len(repeaters)}")
    print(f"distinct repeating sources     {repeaters['repeater_name'].nunique()}")

    spread, n_sub = per_source_spread(repeaters, MIN_BURSTS)

    # Mean across features collapses each source to a single scatter figure. The mean is used
    # rather than a norm so the value stays on the same scale as an individual feature's
    # standard deviation, and so it is directly comparable to INJECTED_SCATTER.
    all_features = spread.mean(axis=1)
    morphological = spread.drop(columns=CONTEXT_FEATURE).mean(axis=1)

    print(f"\nsources with >= {MIN_BURSTS} sub-bursts   {len(spread)}, covering {n_sub} sub-bursts")
    print(f"median within-source scatter   {all_features.median():.3f}  (all {len(FEATURES)} features)")
    print(f"                               {morphological.median():.3f}  ({CONTEXT_FEATURE} excluded)")
    print(f"tightest source                {all_features.min():.3f}  ({all_features.idxmin()})")
    print(f"loosest source                 {all_features.max():.3f}  ({all_features.idxmax()})")

    print(f"\ninjected collective scatter    {INJECTED_SCATTER:.3f}")
    print(f"median source is tighter by    {all_features.median() / INJECTED_SCATTER:.1f}x")
    print(f"tightest source is tighter by  {all_features.min() / INJECTED_SCATTER:.1f}x")

    # The median is the figure to quote. The tightest-source figure depends on the threshold,
    # because low-count sources yield unreliable standard deviations, and the sweep exposes that.
    print(f"\nsensitivity to the {MIN_BURSTS}-sub-burst threshold")
    print(f"{'min':>5}  {'sources':>7}  {'median':>7}  {'tightest':>8}  {'median/injected':>15}")
    for minimum in MIN_BURSTS_SWEEP:
        sweep, _ = per_source_spread(repeaters, minimum)
        means = sweep.mean(axis=1)
        print(
            f"{minimum:>5}  {len(means):>7}  {means.median():>7.3f}  {means.min():>8.3f}"
            f"  {means.median() / INJECTED_SCATTER:>15.1f}"
        )


if __name__ == "__main__":
    main()
