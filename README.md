# FRB Morphological Anomaly Detection

An independent research project to identify fast radio bursts (FRBs) whose
properties are unusual relative to the population, using unsupervised
anomaly detection on the public CHIME/FRB catalogues.

The guiding question: are there bursts the standard catalogue parameters flag as
odd, and (in the later phase) bursts that look ordinary in the numbers but
strange in their time-frequency structure? Those are the ones a repeater-vs-nonrepeater
framing would miss.

## Status

- **Phase 1 (tabulated-parameter baseline): analytically complete.** The full
  Catalog 2 pipeline is built and run end to end: cleaning, six unsupervised
  outlier-detection methods, validation of the detector by injection-recovery, a
  real-data specificity check, and a flagged candidate shortlist. A short written
  report is in progress.
- **Phase 2 (the core): not started.** Unsupervised representation learning on the
  dynamic spectra (waterfalls), starting from the Phase 1 candidates, with an
  instrumental-vs-astrophysical triage and a comparison against this baseline.

Phase 1 is deliberately framed as preliminary baseline work, not the end goal. It
exists to learn the data hands-on and to establish the baseline against which Phase
2 is measured.

## Data

All data is public CHIME/FRB catalogue data. Provenance is recorded in
[`data/raw/SOURCE.txt`](data/raw/SOURCE.txt).

- **Catalog 2** (~4,500 bursts): CANFAR data archive,
  DOI [10.11570/25.0066](https://doi.org/10.11570/25.0066), The CHIME/FRB
  Collaboration 2026, ApJS 283 (arXiv:2601.09399).

## Method (Phase 1)

Each burst is described by its tabulated morphological parameters: dispersion
measure, width, flux, fluence, two empirical spectral-shape parameters, peak
frequency and bandwidth. Eight features are used; scattering time is excluded
because Catalog 2 stores it as zero for roughly 60% of bursts, which would
introduce a brightness selection effect.

Six unsupervised, non-parametric outlier scores are computed per burst, chosen to
span the standard anomaly geometries (Chandola, Banerjee and Kumar 2009): Local
Outlier Factor (local density), kNN distance (global isolation), Extended Isolation
Forest (partition-based), CBLOF (cluster-based, for collective anomalies), a
random-subspace ensemble (feature bagging, for correlation-breaking anomalies) and
a non-parametric conditional detector (for context-dependent anomalies). Scoring is
label-blind: the repeater tag is not used.

### Validation by injection-recovery

There is no ground-truth anomaly label, so the detector is validated by
injection-recovery rather than against a proxy. Synthetic anomalies of known
geometry (global, local, collective, subspace, contextual) are injected into the
real feature table across a range of strengths; each method scores the augmented
data blind, and recovery is the fraction of injected anomalies ranked into the
anomalous tail. This measures, per geometry, how sensitive the detector is and
which geometries it is blind to.

Sensitivity is only half the question: a method can recover injected anomalies
perfectly while flagging everything on real data. A specificity check on the real
bursts (score concentration, and stability of the top-ranked bursts across random
seeds) is therefore applied before a method is trusted.

## Key Phase 1 result (Catalog 2)

- The validated detector is **CBLOF + LOF + kNN**. A max-aggregated subspace
  variant had the highest injection-recovery sensitivity but was rejected by the
  specificity check, because its top-ranked real bursts were unstable across random
  seeds with no separated score tail. Sensitivity without specificity is a mirage.
- The pipeline reliably recovers global, local, subspace and contextual anomalies,
  and (via CBLOF) collective anomalies large enough to form their own cluster. One
  honest blind spot remains: a small embedded micro-cluster, below the cluster
  method's resolution, is not recovered by any method.
- The output is a flagged candidate shortlist of morphologically anomalous bursts,
  those ranked anomalous by two or more independent methods. Each is annotated by
  the reliability of the fit that drives its anomaly, but nothing is removed: a
  large relative fit error on a narrow or narrowband burst is the expected signature
  of a real extreme event, not a fit failure, so the artifact-versus-real call is
  deferred to Phase 2, where the waterfall is visible directly.

## Repository layout

```
src/frb_anomaly/            Importable package: catalogue reader, outlier scorers, anomaly generators
Phase 1/notebooks/cat2/     Catalog 2 pipeline: 01 cleaning, 02 methods, 03 injection-recovery,
                            04 candidate-validation, 05 artifact-flagging
Phase 1/notebooks/archive/  Retired approaches, kept for provenance
Phase 1/scripts/            Standalone analyses, catalogue inspection and within-source scatter
tests/                      pytest suite for the package
data/raw/                   Downloaded catalogue + SOURCE.txt provenance
data/processed/             Feature tables, method scores and the candidate shortlist
reports/Phase 1/figures/    Result figures
reports/Phase 1/results_data/
                            Persisted Results-section source data and LaTeX table fragments
requirements.txt            Runtime dependencies
requirements-dev.txt        Development dependencies (pytest)
```

## Package

`src/frb_anomaly/` is an importable Python package of the reusable, unit-tested
pieces shared across the notebooks.

- `data.py`: catalogue loading. `read_votable(path)` hand-parses a VOTable file
  (the XML catalogue format used by CHIME/FRB and IVOA-compliant archives) into a
  `(fields, df)` pair using the Python standard library only; `load_catalog2()`
  loads the Catalog 2 CSV; `_local(tag)` strips XML namespace prefixes.
- `methods.py`: the six outlier scorers (LOF, kNN, EIF, CBLOF, the subspace
  ensemble and the conditional detector), each returning a per-burst score with
  the convention that higher means more anomalous.
- `injection.py`: the synthetic anomaly generators (one per geometry), a
  `Calibration` that binds them to a dataset, and the recovery metric.

The `tests/` suite covers all three modules, including a named regression test
guarding the local-outlier generator against a past bug.

## Reproducing

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order within `Phase 1/notebooks/cat2/`: `01_cleaning_cat2`,
`02_methods_cat2`, `03_injection_recovery_cat2`, `04_candidate_validation_cat2`,
`05_artifact_flagging_cat2`. They read from `data/raw/`, write feature tables,
scores and the candidate shortlist to `data/processed/phase_1/`, and write figures
to `reports/Phase 1/figures/`. Developed against Python 3.11.

To run the test suite:

```bash
pip install -r requirements-dev.txt
pytest
```

## References

This project builds directly on representation learning for FRB dynamic spectra
(arXiv:2412.12394), taking it to full Catalog 2 scale with anomaly detection as
the primary goal. Related work:

- The ROAD to discovery: ML anomaly detection in radio astronomy spectrograms (arXiv:2307.01054)
- Repeating vs nonrepeating FRBs, deep learning morphological characterization (arXiv:2509.06208)
- Spectral morphological division of FRBs with CHIME/FRB Catalog 2 (arXiv:2601.16048)
