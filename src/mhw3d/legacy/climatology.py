import xarray as xr
import numpy as np
import pandas as pd

# DOY constants matching Oliver's 2012 leap-year reference
_FEB28 = 59  # DOY 59
_FEB29 = 60  # DOY 60
_MAR1  = 61  # DOY 61


def _interp_feb29(da):
    """
    Linearly interpolate DOY 60 (Feb 29) from DOY 59 (Feb 28) and DOY 61 (Mar 1),
    replicating Oliver's treatment of leap-day values.  Oliver's loop skips DOY 60
    entirely and sets it by interpolation; we overwrite the pool result with the
    interpolated value, which gives the same outcome.
    """
    feb29_val = 0.5 * da.sel(dayofyear=_FEB28) + 0.5 * da.sel(dayofyear=_MAR1)
    return da.where(da.dayofyear != _FEB29, feb29_val)


def _build_cross_year_pad(arr, w, nan_dtype=None):
    """
    Build left and right year-boundary pads for a (year, doy, *spatial) DataArray.

    left_pad[Y]  = arr[Y-1, -w:]   (end of previous year; NaN for first year)
    right_pad[Y] = arr[Y+1,  :w]   (start of next year;   NaN for last year)

    Uses full_like + concat with dropped year coords to avoid alignment issues and
    any extra in-memory copies — all operations remain lazy in dask.
    nan_dtype: if given, the NaN boundary slice is cast to this dtype before
               full_like (needed when arr is integer and NaN can't be represented).
    """
    year_vals = arr.year.values

    # --- left pad ---
    _end = arr.isel(doy=slice(-w, None))
    _nan_src = _end.isel(year=[0])
    if nan_dtype is not None:
        _nan_src = _nan_src.astype(nan_dtype)
    _nan_l = xr.full_like(_nan_src, np.nan).drop_vars("year")
    _dat_l = _end.isel(year=slice(None, -1)).drop_vars("year")
    left_pad = xr.concat([_nan_l, _dat_l], dim="year") \
                 .assign_coords(year=year_vals)

    # --- right pad ---
    _start = arr.isel(doy=slice(None, w))
    _nan_src = _start.isel(year=[-1])
    if nan_dtype is not None:
        _nan_src = _nan_src.astype(nan_dtype)
    _nan_r = xr.full_like(_nan_src, np.nan).drop_vars("year")
    _dat_r = _start.isel(year=slice(1, None)).drop_vars("year")
    right_pad = xr.concat([_dat_r, _nan_r], dim="year") \
                  .assign_coords(year=year_vals)

    return left_pad, right_pad


def _pool_window(da, windowHalfWidth):
    """
    Reshape da (time) → (year, doy), build the ±windowHalfWidth rolling window
    with correct cross-year boundary behaviour, and return stacked samples ready
    for reduction.  Also returns the presence map for skipna=False enforcement.

    Cross-year padding is constructed with full_like+concat (not roll+where or shift)
    so that no extra in-memory copies are made and the dask graph stays clean.
    """
    year = da.time.dt.year
    doy  = da.time.dt.dayofyear

    da_y_doy = (
        da.assign_coords(year=("time", year.data), doy=("time", doy.data))
          .set_index(time=["year", "doy"])
          .sortby("time")
          .unstack("time")
          .sortby("year").sortby("doy")
          .reindex(doy=np.arange(1, 367))
    )
    if hasattr(da_y_doy.data, "chunks"):
        da_y_doy = da_y_doy.chunk({"year": -1, "doy": -1})

    ones = xr.DataArray(np.ones(da.sizes["time"], dtype="int8"),
                        coords={"time": da.time}, dims=["time"])
    present = (
        ones.assign_coords(year=("time", year.data), doy=("time", doy.data))
            .set_index(time=["year", "doy"])
            .sortby("time")
            .unstack("time")
            .sortby("year").sortby("doy")
            .reindex(doy=np.arange(1, 367))
    )
    if hasattr(present.data, "chunks"):
        present = present.chunk({"year": -1, "doy": -1})

    Nday = 366
    w = 2 * int(windowHalfWidth) + 1

    left_pad,  right_pad  = _build_cross_year_pad(da_y_doy, w)
    left_pres, right_pres = _build_cross_year_pad(present,  w, nan_dtype=float)

    pad_vals = xr.concat([left_pad,  da_y_doy, right_pad],  dim="doy")
    pad_pres = xr.concat([left_pres, present,  right_pres], dim="doy")

    if hasattr(pad_vals.data, "chunks"):
        pad_vals = pad_vals.chunk({"doy": -1})
    if hasattr(pad_pres.data, "chunks"):
        pad_pres = pad_pres.chunk({"doy": -1})

    win_vals = pad_vals.rolling(doy=w, center=True, min_periods=w).construct("w")
    win_pres = pad_pres.rolling(doy=w, center=True, min_periods=w).construct("w")
    centre_vals = win_vals.isel(doy=slice(w, Nday + w))
    centre_pres = win_pres.isel(doy=slice(w, Nday + w))

    samples        = centre_vals.stack(samples=("year", "w")).chunk({"samples": -1})
    present_sample = centre_pres.stack(samples=("year", "w")).chunk({"samples": -1})

    return samples, present_sample


def _smooth_doy(da, smoothPercentileWidth):
    """31-day circular running-mean smoothing over the dayofyear dimension."""
    year = 1996  # Dummy leap year
    date_doy = pd.to_datetime(da.dayofyear - 1, unit='D', origin=f'{year}-01-01', errors='coerce')
    time_concat = [*(date_doy[-smoothPercentileWidth:] - pd.DateOffset(years=1)),
                   *date_doy,
                   *(date_doy[:smoothPercentileWidth] + pd.DateOffset(years=1))]
    stacked = xr.concat([da.isel(dayofyear=slice(-smoothPercentileWidth, None)),
                         da,
                         da.isel(dayofyear=slice(None, smoothPercentileWidth))], dim='dayofyear') \
               .assign_coords({'time': ('dayofyear', time_concat)}) \
               .swap_dims({'dayofyear': 'time'})
    if hasattr(stacked.data, "chunks"):
        stacked = stacked.chunk({"time": -1})
    sm = stacked.rolling(time=smoothPercentileWidth, min_periods=1, center=True).mean()
    mid = sm.groupby('time.year')[1996].swap_dims({'time': 'dayofyear'}).drop_vars('time')
    return mid.rename({'time': 'dayofyear'}) if 'time' in mid.dims else mid


def compute_climatology(obj, varname=None, smoothPercentile=True, smoothPercentileWidth=31,
                        windowHalfWidth=5):
    """
    Climatological seasonal cycle, replicating Oliver et al. (2016) exactly.

    Pools all data within ±windowHalfWidth actual calendar days across all years
    (same pooling as the threshold) and takes the nanmean.  DOY 60 (Feb 29) is
    linearly interpolated from Feb 28 and Mar 1, exactly as Oliver does.

    Returns a DataArray with a 'dayofyear' dimension.
    """
    from mhw3d.common.core import _to_da
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")

    spatial_dims = [d for d in da.dims if d != "time"]

    samples, _ = _pool_window(da, windowHalfWidth)

    clim_vec = xr.apply_ufunc(
        lambda arr: np.nanmean(arr, axis=0),
        samples,
        input_core_dims=[["samples"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[samples.dtype],
        keep_attrs=True,
    )

    if "doy" in clim_vec.dims:
        clim_vec = clim_vec.rename({"doy": "dayofyear"})
    clim_vec = clim_vec.assign_coords(dayofyear=np.arange(1, 367))
    clim_vec = clim_vec.transpose(*(spatial_dims + ["dayofyear"]) if spatial_dims else ("dayofyear",))

    clim_vec = _interp_feb29(clim_vec)

    if smoothPercentile:
        out = _smooth_doy(clim_vec, smoothPercentileWidth)
    else:
        out = clim_vec

    return out


def compute_threshold(obj, pctile=0.9, windowHalfWidth=5,
                      smoothPercentile=True, smoothPercentileWidth=31,
                      varname=None):
    """
    Threshold (quantile) per DOY, replicating Oliver et al. (2016) exactly.

    Pools all data within ±windowHalfWidth actual calendar days across all years,
    computes the pctile quantile, interpolates Feb 29 from Feb 28 and Mar 1, then
    optionally smooths with a 31-day running mean.

    Enforces strict skipna=False: any measured-but-NaN value in the pooling window
    sets the threshold to NaN (conservative behaviour for gappy mooring records).

    Returns a DataArray with dims (*spatial, dayofyear).
    """
    from mhw3d.common.core import _to_da
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")

    spatial_dims = [d for d in da.dims if d != "time"]
    coords = {d: (d, da.coords[d].values if d in da.coords else np.arange(da.sizes[d]))
              for d in spatial_dims}

    samples, present_sample = _pool_window(da, windowHalfWidth)

    def _nanquantile_linear(a, q, axis=0):
        try:
            return np.nanquantile(a, q, axis=axis, method="linear")
        except TypeError:
            return np.nanquantile(a, q, axis=axis, interpolation="linear")

    q_vec = xr.apply_ufunc(
        lambda arr: _nanquantile_linear(arr, pctile, axis=0),
        samples,
        input_core_dims=[["samples"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[samples.dtype],
        keep_attrs=True,
    )

    # Strict skipna=False
    any_meas_nan = xr.apply_ufunc(
        lambda val, pres: np.any(np.isnan(val) & np.isfinite(pres), axis=0),
        samples, present_sample,
        input_core_dims=[["samples"], ["samples"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
    )
    q_vec = q_vec.where(~any_meas_nan)

    if "doy" in q_vec.dims:
        q_vec = q_vec.rename({"doy": "dayofyear"})
    q_vec = q_vec.assign_coords(dayofyear=np.arange(1, 367))
    thresh = q_vec.transpose(*(spatial_dims + ["dayofyear"]) if spatial_dims else ("dayofyear",))
    for d in spatial_dims:
        if d not in thresh.coords:
            thresh = thresh.assign_coords({d: coords[d][1]})

    thresh = _interp_feb29(thresh)

    if smoothPercentile:
        out = _smooth_doy(thresh, smoothPercentileWidth)
    else:
        out = thresh

    return out
