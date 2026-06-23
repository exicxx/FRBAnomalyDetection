# FRB Morphological Anomaly Detection

An independent research project to identify fast radio bursts (FRBs) whose
properties are unusual relative to the population, using unsupervised
anomaly detection on the public CHIME/FRB catalogues.

The guiding question: are there bursts the standard catalogue parameters flag as
odd, and (in the later phase) bursts that look ordinary in the numbers but
strange in their time-frequency structure? Those are the ones a repeater-vs-nonrepeater
framing would miss.

## Status

- **Phase 1 (tabulated-parameter baseline): analytically complete.** Cleaning,
  three outlier-detection methods, and a source-level validation test are all
  built and run end to end on both catalogues. A short written report is in
  progress.
- **Phase 2 (the core): not started.** Unsupervised representation learning on
  the dynamic spectra (waterfalls), with a systematic instrumental-vs-astrophysical
  triage of the outliers and a comparison against the Phase 1 baseline.

Phase 1 is deliberately framed as preliminary baseline work, not the end goal.
It exists to learn the data hands-on and to establish the baseline against which
Phase 2 is measured.

## Data

All data is public CHIME/FRB catalogue data. Provenance is recorded in
[`data/raw/SOURCE.txt`](data/raw/SOURCE.txt).

- **Catalog 2** (~4,500 bursts): CANFAR data archive,
  DOI [10.11570/25.0066](https://doi.org/10.11570/25.0066), The CHIME/FRB
  Collaboration 2026, ApJS 283 (arXiv:2601.09399).

## Method (Phase 1)

Each burst is described by its tabulated morphological parameters (width,
bandwidth, fluence, spectral index, and so on). Eight features are used;
scattering time is excluded because Catalog 2 stores it as zero for roughly
60% of bursts, which would introduce a brightness selection effect into the
analysis.

Three complementary unsupervised outlier scores are computed per burst:

- **Mahalanobis distance** (distance from the population centre in whitened space)
- **Local Outlier Factor (LOF)** (local-density-based)
- **Extended Isolation Forest (EIF)** (partition-based)

Because there is no ground-truth "anomaly" label, the scores are validated against
a proxy: known repeaters should sit toward the anomalous tail. A source-level
permutation test (per-source aggregation, so the result is not driven by a couple
of prolific repeaters) reports both a ranking AUC and top-K tail enrichment, with
Bonferroni correction across methods.

## Key Phase 1 result (Catalog 2)

On the honest, source-controlled tests, the signal is real but more nuanced than
a naive read suggests:

- **Per-sub-burst tail enrichment** (top 5/10/20%) is significant for all three
  methods (p = 0.0001).
- **Median-per-source ranking AUC**: Mahalanobis strong (0.783, p_Bonf 0.0003),
  EIF weak but real (0.596), LOF not significant (0.562).

The max-aggregated per-source AUC looks much stronger for all three, but it is
upward-biased by a max-of-many effect from prolific repeaters that the null does
not reproduce, so it is explicitly **not** the headline. The defensible headline
is the median AUC plus the per-sub-burst tail enrichment.

Top non-repeater anomalies feeding the Phase 2 shortlist: **FRB20200321E** and
**FRB20181119D**.

## Repository layout

```
src/frb_anomaly/        Shared Python module (catalogue readers, utilities)
Phase 1/notebooks/cat2/ Catalog 2 pipeline: cleaning, methods, validation
Phase 1/scripts/        One-off exploration scripts
data/raw/               Downloaded catalogue + SOURCE.txt provenance
data/processed/         Feature tables and method scores produced by the notebooks
reports/                Result figures (written report to follow)
requirements.txt        Runtime dependencies
requirements-dev.txt    Development dependencies (pytest)
```

## Reproducing

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order within `Phase 1/notebooks/cat2/`: `01_cleaning_cat2`,
then `02_methods_cat2`, then `03_validation_cat2`. They read from `data/raw/`,
write feature tables and scores to `data/processed/phase_1/`, and write figures
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
