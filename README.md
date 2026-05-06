# mhw3d

A scalable xarray/dask framework for detecting marine heatwaves (MHWs) in 3D ocean data (time × depth or time × lat × lon), following Hobday et al. (2016) and Hobday et al. (2018).

## Installation

```bash
conda env create -f environment.yml
conda activate mhw3d
pip install -e .
```

## Module structure

The package is split into three submodules with different contracts.

### `mhw3d.legacy` — frozen Hobday 2016 contract

Reproduces the results of [ecjoliver/marineHeatWaves](https://github.com/ecjoliver/marineHeatWaves) on 3D xarray/dask data. No API changes are permitted; all changes must pass the regression test.

| Function | Description |
|---|---|
| `legacy.compute_climatology(da, climatologyPeriod=(1982, 2011), ...)` | Per-DOY climatological mean, replicating Oliver et al. (2016) exactly |
| `legacy.compute_threshold(da, pctile=0.9, climatologyPeriod=(1982, 2011), ...)` | Per-DOY 90th-percentile threshold, replicating Oliver et al. (2016) exactly |

Both functions return a DataArray with a `dayofyear` dimension (1–366). `climatologyPeriod` selects the years used to compute the baseline; the result can be applied to any date range.

### `mhw3d.best_practice` — current-literature methods

Configurable methods following current best practice. Breaking changes are allowed between minor versions.

| Function | Description |
|---|---|
| `best_practice.compute_climatology(da, ...)` | Per-DOY climatological mean with optional smoothing |
| `best_practice.compute_threshold(da, pctile=0.9, windowHalfWidth=5, ...)` | Per-DOY threshold with correct cross-year boundary handling |

### `mhw3d.common` — shared utilities

| Function | Description |
|---|---|
| `common.calculate_severity(da, seas, thresh)` | Computes `T_anom` and `severity = T_anom / (thresh - seas)` |
| `common.calculate_mhw_metrics(ds)` | Vectorised event detection and metrics over grids; returns a Dataset of events |

## Basic usage

```python
import xarray as xr
from mhw3d import legacy, common

# Load data
ds = xr.open_dataset("sst.nc").chunk({"time": -1})

# 1. Compute climatology and threshold from the baseline period (1982–2011 by default)
seas   = legacy.compute_climatology(ds["sst"])
thresh = legacy.compute_threshold(ds["sst"])

# 2. Compute anomalies and severity for the full record
severity_ds = common.calculate_severity(ds["sst"], seas, thresh)

# 3. Detect events
events = common.calculate_mhw_metrics(severity_ds)

# 4. Drop empty event slots
events = events.dropna("event", how="all")
```

`seas` and `thresh` are indexed by `dayofyear` (1–366) and can be applied to any time range. To use a different baseline period, pass `climatologyPeriod=(start_year, end_year)`:

```python
# Compute baseline from 1982–2011, detect events through 2024
seas   = legacy.compute_climatology(ds["sst"], climatologyPeriod=(1982, 2011))
thresh = legacy.compute_threshold(ds["sst"],   climatologyPeriod=(1982, 2011))

full_record = xr.open_dataset("sst_1982_2024.nc").chunk({"time": -1})
severity_ds = common.calculate_severity(full_record["sst"], seas, thresh)
events = common.calculate_mhw_metrics(severity_ds)
```

## Notes

- **Rechunk before detection.** Call `.chunk({"time": -1})` on your dataset before passing it to any function that operates over time. Dask will raise an error otherwise.
- **Severity > 1** corresponds to an MHW state (Hobday et al. 2018).
- **MHW categories** follow Hobday (2018): 1–2 moderate, 2–3 strong, 3–4 severe, >4 extreme.

## References

- Hobday et al. (2016), *Progress in Oceanography* — MHW detection convention
- Hobday et al. (2018), *Oceanography* — severity categories
- Oliver (2016), [ecjoliver/marineHeatWaves](https://github.com/ecjoliver/marineHeatWaves) — reference implementation
