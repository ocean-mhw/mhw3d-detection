# mhw3d-detection

Project context for Claude Code sessions in this repository.

---

## What this package is

`mhw3d` is a Python package for detecting marine heatwaves (MHWs) in
3D ocean data — time × depth (for moorings) or time × lat × lon (for
satellite/model output) — following the Hobday et al. 2016 detection
convention and Hobday et al. 2018 severity categories.

It is built on **xarray** throughout and is **dask-parallelizable** via
`xr.apply_ufunc(dask="parallelized")`, so the same code runs on a single
mooring depth profile or a global 3D reanalysis.

The package accompanies a paper (in preparation, Malan et al.) and
feeds downstream pipelines including Parcels Lagrangian tracking and
ACCESS-OM2-01 water-mass typology classification.

Primary developer: Neil Malan (CCRC/UNSW). Collaborator: @bnjmnr.

---

## Two modes

The package is split into two submodules with explicitly different
contracts:

### `mhw3d.legacy` — frozen contract

Reproduces the results of [`ecjoliver/marineHeatWaves`](https://github.com/ecjoliver/marineHeatWaves)
bit-exactly, but fast on 3D xarray/dask data.

- **No new features.** No default changes. No API additions.
- Only permitted changes: performance improvements and bug fixes that
  preserve numerical output.
- Defaults are locked at Hobday 2016: `minDuration=5`, `maxGap=2`,
  90th percentile, ±5-day pooling, 31-day smoother.
- Regression-tested against the vendored copy of Oliver's code in
  `tests/data/legacy/marineHeatWaves.py`. Every change to `legacy/`
  must pass `pytest tests/test_reproducibility.py` with agreement
  to `atol=1e-10` on intensities and exact equality on dates/durations.

### `mhw3d.best_practice` — living API

Current-literature methods. Breaking changes allowed between minor
versions, documented in `CHANGELOG.md`.

Scope:
- Full control over baseline length, detrending, percentile, duration thresholds
- Gappy / missing data handling (unified with `min_samples` parameter)
- Monthly / weekly data with error quantification vs daily
- 360-day calendar support
- Deseasoning before percentile calculation (Brunner et al. 2024, Zhao et al. in prep)
- User-friendly data-preparation helpers

### `mhw3d.common` — shared utilities

Code genuinely used by both modes: `_to_da` coercion, run-length
encoding helpers, event-output post-processing (trim all-NaN events,
integer-index conversion), I/O helpers.

**Rule:** the moment a "shared" function needs an `if legacy:` branch,
it is not shared. Split it.

---

## Conventions

- **xarray everywhere.** Functions accept `DataArray` or `Dataset`; use
  `_to_da` for coercion.
- **dask-safe.** Never call `.load()`, `.values`, or `.compute()` inside
  pipeline functions. Keep everything lazy until the user asks for it.
- **Time handling.** Dates are `datetime64[ns]` at the user-facing API
  boundary. Internally, `apply_ufunc` uses float-ordinal for single-dtype
  compatibility and casts back to `datetime64[ns]` afterwards.
- **Event output.** Detection returns a fixed-shape array of length
  `maxEvt` (default 200) with NaN padding, so `apply_ufunc` can vectorize
  over grids. The `common.trim_events` helper removes all-NaN rows
  post hoc.
- **Severity formula.** `severity = T_anom / (thresh - seas)`. Severity > 1
  is an MHW state. Handle division by zero with
  `xr.where(denom > eps, a/denom, np.nan)`, **not** by adding `+ 1e-9`
  to the denominator.
- **MHW categories** follow Hobday 2018: 1–2 moderate, 2–3 strong,
  3–4 severe, >4 extreme.

---

## Rules for Claude Code

1. **Never modify `legacy/` behaviour without running the regression test.**
   If `pytest tests/test_reproducibility.py` fails after your change, the
   change is wrong, not the test.

2. **Rechunk `time` to a single chunk before `apply_ufunc` over time.**
   Any function with `input_core_dims=[['time'], ...]` requires the time
   dimension to be one dask chunk. The user-facing wrappers should do this
   automatically and warn if they had to.

3. **Do not mess with the way the code is structured in terms of efficiency**
  This has been iterated many times by the developers so try not to change the logic

4. **Any change to threshold or severity math needs a test.** This is
   where silent scientific bugs hide.

5. **Prefer explicit over implicit.** Do not hardcode variable-name
   guesses (`'sst'`, `'temp'`, `'thetao'` etc.) in `best_practice/`.
   Require `varname=` or document the detection logic. The `legacy/`
   module may keep the current guessing behaviour for backward compatibility.

6. **Don't split a commit across the legacy/best-practice boundary.**
   Changes to `legacy/` and `best_practice/` should be separate commits
   (and usually separate PRs) so the regression story stays clean.

---

## Repository layout

```
mhw3d-detection/
├── src/mhw3d/
│   ├── __init__.py
│   ├── legacy/           # frozen Hobday 2016 contract
│   │   ├── __init__.py
│   │   └── climatology.py
│   ├── best_practice/    # configurable, current-literature methods
│   │   ├── __init__.py
│   │   └── climatology.py
│   └── common/           # shared utilities (I/O, event post-processing)
│       ├── __init__.py
│       └── core.py
├── tests/
│   ├── test_reproducibility.py      # legacy event detection vs Oliver regression
│   ├── test_threshold.py            # climatology/threshold vs Oliver (TODO)
│   ├── test_detection_edges.py      # run-length edge cases (TODO)
│   └── data/legacy/marineHeatWaves.py   # vendored reference implementation
├── examples/
│   ├── Check_against_oliver.ipynb   # manual validation of clim/threshold vs Oliver
│   └── speed_benchmark.ipynb        # 3D dask performance check
├── environment.yml                  # unpinned dev environment
├── environment-paper.yml            # pinned versions for paper reproduction
└── CLAUDE.md                        # this file
```

The split into `legacy/`, `best_practice/`, and `common/` is complete.
`bipolarMhwToolBox.py` has been deleted.

---

## Commands

```bash
# Create environment
conda env create -f environment.yml
conda activate mhw3d

# Install in editable mode
pip install -e .

# Run tests
pytest tests/                        # all tests
pytest tests/test_reproducibility.py # legacy regression only
pytest -k "gappy"                    # just the gappy-data tests

# Run the speed benchmark (requires a running dask cluster)
jupyter lab examples/speed_benchmark.ipynb
```

---

## Current priorities

Tracked in the open GitHub issue. In rough order:

1. ~~Performance issue (done)~~
2. ~~Variable-name consolidation (`ssta` → `T_anom`, etc.)~~
3. ~~Restructure into `legacy/` / `best_practice/` / `common/`~~
4. Tighten `test_reproducibility.py`: seed RNG, `atol=1e-10`, add a
   second synthetic case; promote `Check_against_oliver.ipynb` content
   into `test_threshold.py` (clim/threshold agreement vs Oliver)
5. Unified gappy threshold with `min_samples` parameter → `best_practice`
6. Event-trimming helper, integer-index output helper → `common`
7. 360-day calendar support (bnjmnr)
8. Deseasoning preprocessing step (Rose et al. 2024 approach)
9. Monthly / weekly data with error quantification (building on Welch)
10. Pin `environment-paper.yml`; tag `v1.0.0-paper`; archive on Zenodo

---

## Known gotchas

- **Rechunk before detection.** Calling `calculate_MHWs_metrics` on a
  dask-backed dataset without first doing `ds.chunk({"time": -1, ...})`
  will fail inside `apply_ufunc`. Documented in `speed_benchmark.ipynb`.

- **Gappy-threshold strictness (legacy).** `compute_threshold` enforces
  strict `skipna=False` over the pooled ±5-day × N-year window: if any
  timestamp in that window has a NaN value but is present in the record,
  the threshold for that DOY is set to NaN. For mooring data with heavy
  temporal gaps this zeroes out most of the threshold field. Mitigation
  lives in `best_practice` (unified `min_samples` version); legacy keeps
  strict behaviour to match Oliver.

- **Non-leap years and DOY 366.** The pooled-window machinery uses
  `dayofyear = 1..366` with 1996 as a reference leap year for the
  smoothing step. DOY 366 in non-leap years is handled via circular
  padding; do not change this without regression testing.

- **Oliver's DOY convention vs Python's.** Oliver maps every date to its
  position in leap year 2012 (so Mar 1 is always DOY 61, Dec 31 always DOY 366,
  regardless of whether the actual year is a leap year). `_clim_doy()` in
  `common/core.py` replicates this internally: for non-leap years it adds 1
  to all DOYs ≥ 60, so pooling and severity indexing use Oliver's calendar.
  The public-facing output array still uses `dayofyear = 1..366` anchored to
  a leap year, so users can index with `ds.time.dt.dayofyear` on leap-year
  data and get exact results. On mixed-year data a residual of ~0.01 °C
  remains at DOYs 45–75 in both leap and non-leap years: the 31-day smoother
  integrates pool differences from the structural NaN at DOY 60 that non-leap
  years leave in the `(year, doy)` 2D grid. Irreducible without abandoning
  the 2D grid design. Leap-year-only datasets agree with Oliver at
  floating-point precision (`atol=1e-14`).

- **`+ 1e-9` denominator fudge.** Present in legacy code in three
  places. Do not replicate in `best_practice` — use `xr.where` instead.

---

## External references

- Hobday et al. 2016, *Progress in Oceanography* — detection convention
- Hobday et al. 2018, *Oceanography* — severity categories
- Oliver 2016, [`ecjoliver/marineHeatWaves`](https://github.com/ecjoliver/marineHeatWaves)
  — reference implementation
- Brunner et al. 2024, *Nature Communications* — deseasoning-before-percentile
  approach (`https://www.nature.com/articles/s41467-024-46349-x`)
