import xarray as xr
import numpy as np
import pandas as pd


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


def detrend(obj, period=None, varname=None, deg=1):
    """
    Remove a polynomial trend from a DataArray, fitted over a baseline period.

    The trend is fit on `period` only, then evaluated and subtracted over the
    full time range.  Use this before computing climatology/threshold when the
    data contains a long-term trend that would otherwise inflate the threshold.

    Parameters
    ----------
    obj : DataArray or Dataset
    period : slice, optional
        Time slice used to fit the trend, e.g. slice('1982', '2011').
        If None, uses the full record.
    varname : str, optional
        Variable name if obj is a Dataset.
    deg : int
        Polynomial degree. 1 = linear trend (default).

    Returns
    -------
    DataArray with the trend removed, same shape and coordinates as input.
    """
    from mhw3d.common.core import _to_da
    da = _to_da(obj, varname)
    if 'time' not in da.dims:
        raise ValueError("No 'time' dimension found.")

    da_fit = da.sel(time=period) if period is not None else da
    coeffs = da_fit.polyfit('time', deg=deg)
    coeff_da = next(iter(coeffs.data_vars.values()))
    trend = xr.polyval(da.time, coeff_da)
    return (da - trend).rename(da.name)


def compute_climatology(obj, varname=None, smoothPercentile=True, smoothPercentileWidth=31,
                        baseline_period=None):
    """
    Climatological seasonal cycle (DOY mean) with optional 31-day running-mean smoothing.
    Accepts DataArray or Dataset. Returns a DataArray with a 'dayofyear' dimension.

    Best-practice version: computes a clean per-DOY mean then smooths.
    Feb 29 is included naturally from actual data — no interpolation.

    baseline_period : slice, optional
        Time slice used to compute the climatology, e.g. slice('1982', '2011').
        If None, uses the full record.
    """
    from mhw3d.common.core import _to_da
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")
    if baseline_period is not None:
        da = da.sel(time=baseline_period)

    clim = da.groupby("time.dayofyear").mean()

    if smoothPercentile:
        year = 1996  # Dummy leap year for DOY → datetime conversion
        date_doy = pd.to_datetime(clim.dayofyear - 1, unit='D', origin=f'{year}-01-01', errors='coerce')
        time_concat = [*(date_doy[-smoothPercentileWidth:] - pd.DateOffset(years=1)),
                       *date_doy,
                       *(date_doy[:smoothPercentileWidth] + pd.DateOffset(years=1))]
        stacked = xr.concat([clim.isel(dayofyear=slice(-smoothPercentileWidth, None)),
                             clim,
                             clim.isel(dayofyear=slice(None, smoothPercentileWidth))], dim='dayofyear') \
                    .assign_coords({'time': ('dayofyear', time_concat)}) \
                    .swap_dims({'dayofyear': 'time'})

        if hasattr(stacked.data, "chunks"):
            stacked = stacked.chunk({"time": -1})
        smoothed = stacked.rolling(time=smoothPercentileWidth, min_periods=1, center=True).mean()
        mid_sel = smoothed.groupby('time.year')[1996].swap_dims({'time': 'dayofyear'}).drop_vars('time')
        out = mid_sel.rename({'time': 'dayofyear'}) if 'time' in mid_sel.dims else mid_sel
    else:
        out = clim

    return out.assign_coords(dayofyear=clim.dayofyear.data)


def compute_threshold(obj, pctile=0.9, windowHalfWidth=5,
                      smoothPercentile=True, smoothPercentileWidth=31,
                      varname=None, baseline_period=None):
    """
    Threshold (quantile) per DOY with optional smoothing. Accepts DataArray or Dataset.
    Returns a DataArray with dims (*spatial, dayofyear).

    Best-practice version: pools ±windowHalfWidth actual calendar days across all years
    with correct cross-year boundary behaviour (Jan 1 draws from Dec of previous year).
    Feb 29 is included naturally from actual data — no interpolation.
    Enforces strict skipna=False: any measured-but-NaN value in the pooling window → NaN.

    baseline_period : slice, optional
        Time slice used to compute the threshold, e.g. slice('1982', '2011').
        If None, uses the full record.
    """
    from mhw3d.common.core import _to_da
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")
    if baseline_period is not None:
        da = da.sel(time=baseline_period)

    spatial_dims = [d for d in da.dims if d != "time"]
    coords = {d: (d, da.coords[d].values if d in da.coords else np.arange(da.sizes[d]))
              for d in spatial_dims}
    coords["dayofyear"] = ("dayofyear", np.arange(1, 367))

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

    # Strict skipna=False: any measured-but-NaN value in the window → NaN output
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

    if smoothPercentile:
        year = 1996
        date_doy = pd.to_datetime(thresh.dayofyear - 1, unit='D', origin=f'{year}-01-01', errors='coerce')
        time_concat = [*(date_doy[-smoothPercentileWidth:] - pd.DateOffset(years=1)),
                       *date_doy,
                       *(date_doy[:smoothPercentileWidth] + pd.DateOffset(years=1))]
        stacked = xr.concat([thresh.isel(dayofyear=slice(-smoothPercentileWidth, None)),
                             thresh,
                             thresh.isel(dayofyear=slice(None, smoothPercentileWidth))], dim='dayofyear') \
                   .assign_coords({'time': ('dayofyear', time_concat)}) \
                   .swap_dims({'dayofyear': 'time'})

        if hasattr(stacked.data, "chunks"):
            stacked = stacked.chunk({"time": -1})
        sm = stacked.rolling(time=smoothPercentileWidth, min_periods=1, center=True).mean()
        mid = sm.groupby('time.year')[1996].swap_dims({'time': 'dayofyear'}).drop_vars('time')
        out = mid.rename({'time': 'dayofyear'}) if 'time' in mid.dims else mid
    else:
        out = thresh

    return out
