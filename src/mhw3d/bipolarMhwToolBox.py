import xarray as xr
import numpy as np
import datetime as dt
# import bottleneck as bn  # optional; can be injected
import pandas as pd
from pathlib import Path

def _to_da(obj, varname=None):
    """
    Return a DataArray from either a DataArray or Dataset.
    If varname is None and obj is a Dataset, pick the first data_var that has 'time' in dims.
    """
    if isinstance(obj, xr.DataArray):
        return obj
    if isinstance(obj, xr.Dataset):
        if varname is not None:
            return obj[varname]
        # pick first var with 'time' in dims
        for v in obj.data_vars:
            if 'time' in obj[v].dims:
                return obj[v]
        # fallback: first variable
        return next(iter(obj.data_vars.values()))
    raise TypeError("Expected xarray DataArray or Dataset")

def detectMHWs_fromSeverity(time, ts_severity, l_return=200, minDuration=5, maxGap=2):
    """
    Identify MHW events from a severity timeseries (severity = SSTa / (Thresh - Seas)).
    Returns a DataArray (variable,event) with: index_start, index_end, date_start, date_end, duration.
    NOTE: dates are returned as floats here (ordinal-like) to satisfy xarray.apply_ufunc's single dtype;
          they are cast back to datetime64[ns] later in calculate_MHWs_metrics.
    """
    # Severity > 1 is above threshold
    boolSever = ts_severity > 1

    mhw = {}
    mhw['index_start'] = np.full(l_return, np.nan, dtype=float)
    mhw['index_end']   = np.full(l_return, np.nan, dtype=float)
    mhw['date_start']  = np.full(l_return, np.nan, dtype=float)  # cast later
    mhw['date_end']    = np.full(l_return, np.nan, dtype=float)  # cast later
    mhw['duration']    = np.full(l_return, np.nan, dtype=float)

    ia = np.asarray(boolSever)
    n = len(ia)

    if (~np.isnan(ia)).any():  # ensure not all NaN
        y = ia[1:] != ia[:-1]
        i  = np.append(np.where(y), n - 1)
        l_cont = np.diff(np.append(-1, i))
        run_ends = i
        run_starts = i - l_cont + 1

        # keep runs where ia is True
        run_mask = ia[run_ends]
        i_start = run_starts[run_mask]
        i_end   = run_ends[run_mask]

        # enforce minDuration
        keep = (i_end - i_start + 1) >= minDuration
        i_start = i_start[keep]
        i_end   = i_end[keep]

        # merge gaps <= maxGap
        if len(i_start) > 1 and maxGap > 0:
            indGap = np.where(i_start[1:] - i_end[:-1] <= maxGap)[0]
            i_start = np.delete(i_start, indGap + 1)
            i_end   = np.delete(i_end,   indGap)

        n_events = len(i_start)
        t = np.asarray(time)
        mhw['index_start'][:n_events] = i_start
        mhw['index_end'][:n_events]   = i_end
        mhw['date_start'][:n_events]  = t[i_start]
        mhw['date_end'][:n_events]    = t[i_end]
        mhw['duration'][:n_events]    = (i_end - i_start + 1).astype(float)

    ds_mhw = xr.Dataset({var: ('event', data) for var, data in mhw.items()},
                        coords={'event': ('event', np.arange(l_return))})
    return ds_mhw.to_array()

def add_metrics_MHWs(time, ssta, date_start, date_end, l_return=200):
    """
    Compute peak date and intensities (max, mean, cumulative) for each detected event.
    NOTE: date_peak is returned as float here (ordinal-like) to satisfy xarray.apply_ufunc's single dtype;
          it is cast back to datetime64[ns] later in calculate_MHWs_metrics.
    """
    time_np = np.asarray(time)
    ssta_np = np.asarray(ssta)

    mhw_out = {
        'index_peak':   np.full(l_return, np.nan, dtype=float),
        'date_peak':    np.full(l_return, np.nan, dtype=float),  # cast later
        'intensity_max':np.full(l_return, np.nan, dtype=float),
        'intensity_mean':np.full(l_return, np.nan, dtype=float),
        'intensity_cumul':np.full(l_return, np.nan, dtype=float),
    }

    dstart = np.asarray(date_start)
    dend   = np.asarray(date_end)
    # valid events: those with real start/end
    valid_evt = ~np.isnan(dstart) & ~np.isnan(dend)
    if valid_evt.any():
        # time x event mask
        mask = (time_np[:, None] >= dstart[None, :]) & (time_np[:, None] <= dend[None, :])
        np_ssta_evts = np.where(mask, ssta_np[:, None], np.nan)

        # which event columns have any valid values?
        valid_cols = ~np.isnan(np_ssta_evts).all(axis=0)
        if valid_cols.any():
            idx_peak_cols = np.nanargmax(np_ssta_evts[:, valid_cols], axis=0)
            # map back to full-length arrays
            idx_peak = np.full(np_ssta_evts.shape[1], 0, dtype=int)
            idx_peak[valid_cols] = idx_peak_cols

            mhw_out['index_peak'][valid_evt] = idx_peak[valid_evt]
            mhw_out['date_peak'][valid_evt]  = time_np[idx_peak][valid_evt]
            mhw_out['intensity_max'][valid_evt]   = np.nanmax(np_ssta_evts[:, valid_cols], axis=0)
            mhw_out['intensity_mean'][valid_evt]  = np.nanmean(np_ssta_evts[:, valid_cols], axis=0)
            mhw_out['intensity_cumul'][valid_evt] = np.nansum(np_ssta_evts[:, valid_cols], axis=0)

    ds_mhw_out = xr.Dataset({var: ('event', data) for var, data in mhw_out.items()},
                            coords={'event': ('event', np.arange(l_return))})
    return ds_mhw_out.to_array()

def calculate_MHWs_metrics(ds, maxEvt=200):
    """
    Vectorized detection + metrics over grids. Expects ds to contain:
      - time (coordinate)
      - either:
           (a) variables 'ssta' and 'severity', or
           (b) variables 'Seas' and 'Thresh' and a temperature var (e.g., 'sst','temp','thetao')
               from which 'ssta' and 'severity' will be derived.
    Returns an xr.Dataset with event-wise metrics per grid cell.
    """
    work = ds.copy()

    def _pick_temp_var(_ds):
        for k in ('sst','temp','thetao','tas','temperature','T','SST'):
            if k in _ds.data_vars and 'time' in _ds[k].dims:
                return k
        for k, v in _ds.data_vars.items():
            if 'time' in v.dims:
                return k
        raise ValueError("Could not find a temperature-like variable with 'time' in dims.")

    if 'ssta' not in work:
        if 'Seas' in work:
            tvar = _pick_temp_var(work)
            doy = work['time'].dt.dayofyear
            seas_on_time = work['Seas'].sel(dayofyear=doy)
            work['ssta'] = work[tvar] - seas_on_time
        else:
            raise ValueError("Missing 'ssta' and no 'Seas' to compute it from.")

    if 'severity' not in work:
        if 'Thresh' in work and 'Seas' in work:
            doy = work['time'].dt.dayofyear
            seas_on_time = work['Seas'].sel(dayofyear=doy)
            thresh_on_time = work['Thresh'].sel(dayofyear=doy)
            denom = (thresh_on_time - seas_on_time)
            work['severity'] = work['ssta'] / denom
        else:
            raise ValueError("Missing 'severity' and cannot compute it (need 'Thresh' and 'Seas').")

    mhw = xr.apply_ufunc(
        detectMHWs_fromSeverity, work.time, work.severity,
        kwargs={'l_return': maxEvt},
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['variable', 'event']],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"variable": 5, "event": maxEvt}},
        output_dtypes=['float'],
        vectorize=True
    )
    ds_mhw = mhw.assign_coords({'variable': ('variable',
                    ['index_start','index_end','date_start','date_end','duration'])}).to_dataset(dim='variable')
    ds_mhw['date_start'] = ds_mhw.date_start.astype('datetime64[ns]')
    ds_mhw['date_end']   = ds_mhw.date_end.astype('datetime64[ns]')

    ds_mhw_more = xr.apply_ufunc(
        add_metrics_MHWs, work.time, work.ssta,
        ds_mhw.date_start, ds_mhw.date_end,
        kwargs={'l_return': maxEvt},
        input_core_dims=[['time'], ['time'], ['event'], ['event']],
        output_core_dims=[['variable', 'event']],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"variable": 5, "event": maxEvt}},
        output_dtypes=['float'],
        vectorize=True
    ).assign_coords({'variable': ('variable',
                    ['index_peak','date_peak','intensity_max','intensity_mean','intensity_cumul'])}).to_dataset(dim='variable')

    ds_mhw_more['date_peak'] = ds_mhw_more.date_peak.astype('datetime64[ns]')

    return xr.merge([ds_mhw, ds_mhw_more])

def smoothedClima_mhw(obj, varname=None):
    """
    Climatology (DOY) smoothed with 31-day running mean.
    Accepts DataArray or Dataset. If Dataset, pass varname or the first time-varying var is used.
    Returns a DataArray (same spatial dims, 'dayofyear' dim).
    """
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")

    clim = da.groupby("time.dayofyear").mean()
    stacked = xr.concat([clim, clim, clim], dim='year').stack(time=['year','dayofyear']).sortby(['year','dayofyear'])
    smoothed = stacked.rolling(time=31, min_periods=1, center=True).mean()
    mid_sel = smoothed.sel(year=1).drop_vars('year', errors='ignore')
    mid = mid_sel.rename({'time':'dayofyear'}) if 'time' in mid_sel.dims else mid_sel
    return mid.assign_coords(dayofyear=clim.dayofyear.data)


def smoothedThresh_mhw(obj, pctile=0.9, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, varname=None):
    """
    Threshold (quantile) per DOY with optional smoothing. Accepts DataArray or Dataset.
    If Dataset, pass varname or the first time-varying var is used.
    Returns a DataArray with dims (*spatial, dayofyear).
    Robust to: missing coordinate variables on spatial dims; scalar-quantile returns.
    """
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")

    # Use dims (not coords) so we preserve dims even when no coordinate variables are defined
    spatial_dims = [d for d in da.dims if d != "time"]
    spatial_sizes = {d: da.sizes[d] for d in spatial_dims}
    # Build coords: use existing coord values if present, otherwise simple range indices
    coords = {d: (d, da.coords[d].values if d in da.coords else np.arange(spatial_sizes[d])) for d in spatial_dims}
    coords["dayofyear"] = ("dayofyear", np.arange(1, 367))

    # Pre-allocate output with dims (*spatial, dayofyear)
    thresh = xr.DataArray(
        np.full([spatial_sizes[d] for d in spatial_dims] + [366], np.nan, dtype=float),
        coords=coords,
        dims=spatial_dims + ["dayofyear"],
    )

    # DOY quantiles with +/- windowHalfWidth pooling
    doy = da["time"].dt.dayofyear
    for tt in range(1, 367):
        lo = tt - windowHalfWidth
        hi = tt + windowHalfWidth
        if lo < 1 or hi > 366:
            # wrap indices
            idx = ((doy >= ((lo - 1) % 366) + 1) | (doy <= ((hi - 1) % 366) + 1))
        else:
            idx = (doy >= lo) & (doy <= hi)

        da_win = da.where(idx, drop=True)
        qt = da_win.quantile(pctile, dim="time", skipna=False)

        # Normalize qt to match spatial dims of 'thresh' slice
        base = thresh.isel(dayofyear=0, drop=True)  # dims = spatial_dims
        if isinstance(qt, xr.Dataset):
            qt = qt.to_array().squeeze()
        if "quantile" in qt.dims:
            qt = qt.squeeze("quantile", drop=True)
        if qt.ndim == 0:
            qt = xr.full_like(base, qt.item())
        else:
            # Align/broadcast to base (handles dim order and size-1 dims)
            qt = qt.broadcast_like(base)

        # assign for this doy
        thresh.loc[{"dayofyear": tt}] = qt

    # Align phase (rolling window centers) as in Eric's code
    #thresh = thresh.roll(dayofyear=windowHalfWidth, roll_coords=False)

    if smoothPercentile:
        stacked = xr.concat([thresh, thresh, thresh], dim="year").stack(time=["year", "dayofyear"]).sortby(["year", "dayofyear"])
        sm = stacked.rolling(time=smoothPercentileWidth, min_periods=1, center=True).mean()
        mid = sm.sel(year=1).drop_vars("year", errors="ignore")
        out = mid.rename({"time": "dayofyear"}) if "time" in mid.dims else mid
    else:
        out = thresh

    return out


    # Pre-allocate output
    coords = {k: v for k, v in da.coords.items() if k != "time"}
    coords["dayofyear"] = np.arange(1, 367)
    dims = [k for k in coords.keys()]
    thresh = xr.DataArray(np.full([coords[d].size for d in dims], np.nan, dtype=float), coords=coords, dims=dims)

    # DOY quantiles with +/- windowHalfWidth pooling
    doy = da['time'].dt.dayofyear
    windowFullWidth = windowHalfWidth * 2 + 1
    for tt in range(1, 367):
        # define window with wraparound
        lo = tt - windowHalfWidth
        hi = tt + windowHalfWidth
        if lo < 1 or hi > 366:
            # wrap indices
            idx = ((doy >= ((lo-1) % 366) + 1) | (doy <= ((hi-1) % 366) + 1))
        else:
            idx = (doy >= lo) & (doy <= hi)
        da_win = da.where(idx, drop=True)
        qt = da_win.quantile(pctile, dim='time', skipna=False)
        # assign for this doy
        thresh.loc[{ 'dayofyear': tt }] = qt

    # Align phase (rolling window centers) as in Eric's code
    #thresh = thresh.roll(dayofyear=windowHalfWidth, roll_coords=False)

    if smoothPercentile:
        stacked = xr.concat([thresh, thresh, thresh], dim='year').stack(time=['year','dayofyear']).sortby(['year','dayofyear'])
        sm = stacked.rolling(time=smoothPercentileWidth, min_periods=1, center=True).mean()
        mid = sm.sel(year=1).drop_vars('year', errors='ignore')
        out = mid.rename({'time':'dayofyear'}) if 'time' in mid.dims else mid
    else:
        out = thresh

    return out