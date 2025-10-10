import xarray as xr
import numpy as np
import datetime as dt
import bottleneck as bn  # optional; can be injected
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

def smoothedClima_mhw(obj, varname=None, smoothPercentile=True, smoothPercentileWidth=31):
    """
    Climatology (DOY) smoothed with 31-day running mean.
    Accepts DataArray or Dataset. If Dataset, pass varname or the first time-varying var is used.
    Returns a DataArray (same spatial dims, 'dayofyear' dim).
    """
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")

    clim = da.groupby("time.dayofyear").mean()

    # If user requests smoothing:
    if smoothPercentile:
        # For optimized concatenation, create doy vector with just the right amount of time steps
        year = 1996 # Dummy year. The reference year needs to be a leap year
        # Create a pandas datetime of 366 dayof year
        date_doy = pd.to_datetime(clim.dayofyear - 1, unit='D', origin=f'{year}-01-01', errors='coerce')
        # Concat the last 31 days, the whole year and the first 31 days (for datetime vector and for data)
        time_concat = [*(date_doy[-smoothPercentileWidth:]- pd.DateOffset(years=1)), 
                       *date_doy, 
                       *(date_doy[:smoothPercentileWidth] + pd.DateOffset(years=1))]
        stacked = xr.concat([clim.isel(dayofyear=slice(-smoothPercentileWidth,None)), 
                             clim, 
                             clim.isel(dayofyear=slice(None,smoothPercentileWidth))], dim='dayofyear') \
                    .assign_coords({'time':('dayofyear',time_concat)}) \
                    .swap_dims({'dayofyear':'time'})

        # PERF: ensure stacked core dim is single-chunk for rolling
        if hasattr(stacked.data, "chunks"):
            stacked = stacked.chunk({"time": -1})
        # The actual smoothing    
        smoothed = stacked.rolling(time=smoothPercentileWidth, min_periods=1, center=True).mean()
        # Now extract the middle year using this little groupby trick. Another way would be to isel(doy=slice())
        mid_sel = smoothed.groupby('time.year')[1996].swap_dims({'time':'dayofyear'}).drop_vars('time')
        # Make sure the names are right
        out = mid_sel.rename({'time':'dayofyear'}) if 'time' in mid_sel.dims else mid_sel
    else: # If no smoothing
        out = clim
    return out.assign_coords(dayofyear=clim.dayofyear.data)


def smoothedThresh_mhw(obj, pctile=0.9, windowHalfWidth=5,
                       smoothPercentile=True, smoothPercentileWidth=31,
                       varname=None):
    """
    Threshold (quantile) per DOY with optional smoothing. Accepts DataArray or Dataset.
    If Dataset, pass varname or the first time-varying var is used.
    Returns a DataArray with dims (*spatial, dayofyear).
    """
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")

    spatial_dims = [d for d in da.dims if d != "time"]
    coords = {d: (d, da.coords[d].values if d in da.coords else np.arange(da.sizes[d]))
              for d in spatial_dims}
    coords["dayofyear"] = ("dayofyear", np.arange(1, 367))

    year = da.time.dt.year
    doy  = da.time.dt.dayofyear

    # Map to (year, doy) and sort explicitly
    da_y_doy = (
        da.assign_coords(year=("time", year.data), doy=("time", doy.data))
          .set_index(time=["year", "doy"])
          .sortby("time")
          .unstack("time")
          .sortby("year").sortby("doy")
          .reindex(doy=np.arange(1, 367))
    )
    # PERF: make rolling axis single-chunk (and year) to avoid overlap shuffles
    if hasattr(da_y_doy.data, "chunks"):
        da_y_doy = da_y_doy.chunk({"year": -1, "doy": -1})

    # Presence map (distinguish “date absent” vs “value NaN”)
    ones = xr.DataArray(np.ones(da.sizes["time"], dtype="int8"),
                        coords={"time": da.time}, dims=["time"])
    present = (
        ones.assign_coords(year=("time", year.data), doy=("time", doy.data))
            .set_index(time=["year","doy"])
            .sortby("time")
            .unstack("time")
            .sortby("year").sortby("doy")
            .reindex(doy=np.arange(1, 367))
    )
    if hasattr(present.data, "chunks"):
        present = present.chunk({"year": -1, "doy": -1})

    # Circular ±halfwidth pooling via 3× concat + rolling
    Nday = 366
    w = 2*int(windowHalfWidth) + 1

    pad_vals = xr.concat([da_y_doy.isel(doy=slice(-w,None)), da_y_doy, da_y_doy.isel(doy=slice(None,w))], dim="doy")
    pad_pres = xr.concat([present.isel(doy=slice(-w,None)), present, present.isel(doy=slice(None,w))], dim="doy")
    # PERF: keep 'doy' single-chunk after concat
    if hasattr(pad_vals.data, "chunks"):
        pad_vals = pad_vals.chunk({"doy": -1})
    if hasattr(pad_pres.data, "chunks"):
        pad_pres = pad_pres.chunk({"doy": -1})

    win_vals = pad_vals.rolling(doy=w, center=True, min_periods=w).construct("w")
    win_pres = pad_pres.rolling(doy=w, center=True, min_periods=w).construct("w")
    centre_vals = win_vals.isel(doy=slice(w, Nday+w))   # year, doy, w, (*spatial)
    centre_pres = win_pres.isel(doy=slice(w, Nday+w))   # year, doy, w

    # Collapse (year,w) → 'samples'
    samples        = centre_vals.stack(samples=("year","w"))
    present_sample = centre_pres.stack(samples=("year","w"))
    # PERF: core dim must be single-chunk for apply_ufunc
    samples        = samples.chunk({"samples": -1})
    present_sample = present_sample.chunk({"samples": -1})

    # Vectorized quantile (nan-ignoring here; we enforce skipna=False next)
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

    # Emulate skipna=False: if ANY selected measurement is NaN, set output NaN
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

    # Normalize dims/coords → (*spatial, dayofyear)
    if "doy" in q_vec.dims:
        q_vec = q_vec.rename({"doy": "dayofyear"})
    q_vec = q_vec.assign_coords(dayofyear=np.arange(1, 367)) # @NEIL: is this really necessary? It doesn't seem to do anything.
    thresh = q_vec.transpose(*(spatial_dims + ["dayofyear"]) if spatial_dims else ("dayofyear",))
    for d in spatial_dims:
        if d not in thresh.coords:
            thresh = thresh.assign_coords({d: coords[d][1]})

    # Smoothing (optimized concat) + PERF: single-chunk the stacked core dim
    if smoothPercentile:
        # For optimized concatenation, create doy vector with just the right amount of time steps
        year = 1996 # The reference year needs to be a leap year
        # Create a datetime from the dayofyear for a dummy year
        date_doy = pd.to_datetime(thresh.dayofyear - 1, unit='D', origin=f'{year}-01-01', errors='coerce')
        # Concat the last 31 days, the whole year and the first 31 days (for datetime vector and for data)
        time_concat = [*(date_doy[-smoothPercentileWidth:]- pd.DateOffset(years=1)), 
                       *date_doy, 
                       *(date_doy[:smoothPercentileWidth] + pd.DateOffset(years=1))]
        stacked = xr.concat([thresh.isel(dayofyear=slice(-smoothPercentileWidth,None)), 
                             thresh, 
                             thresh.isel(dayofyear=slice(None,smoothPercentileWidth))], dim='dayofyear') \
                   .assign_coords({'time':('dayofyear',time_concat)}) \
                   .swap_dims({'dayofyear':'time'})

        if hasattr(stacked.data, "chunks"):
            stacked = stacked.chunk({"time": -1})   # PERF: avoid rechunk-split/merge
        # The actual smoothing
        sm = stacked.rolling(time=smoothPercentileWidth, min_periods=1, center=True).mean()
        # Now extract the middle year using this little groupby trick. Another way would be to isel(doy=slice())
        mid = sm.groupby('time.year')[1996].swap_dims({'time':'dayofyear'}).drop_vars('time')
        out = mid.rename({'time':'dayofyear'}) if 'time' in mid.dims else mid
    else:
        out = thresh

    return out

    def calculate_Severity(obj, seas, thresh, varname=None):
        """
        Severity, computed from temperature (`obj`), climatological seasonal cycle (`seas`)
        and threshold (`thresh`) as computed by `smoothedClima_mhw` and `smoothedThresh_mhw`, resp.
        The severity is defined as T_anomalies/(thresh - clim). Following this definition, the 
        severity can easily be related to the MHW categories (see. Hobday et al. 2018, 
        https://doi.org/10.5670/oceanog.2018.205).
        Inputs:
            - obj: xr.Dataset, containing the temperature [°C]
            - seas: xr.DataArray, containing climatological seasonal cycle [°C]
            - thresh: xr.DataArray, containing climatological threshold (0.9 percentile) [°C]
            - varname (default: none): in case `obj` is a Dataset, need to know which variable is temp.
        Returns a xr.Dataset with `time`, `T_anom` (temperature anomalies, °C) and `severity` (-).
        """
        # Convert to a dataarray if not the case.
        da = _to_da(obj, varname)
        if "time" not in da.dims:
            raise ValueError("No 'time' dimension found.")

        # Compute temperature anomalies
        Tanom = da.groupby('time.dayofyear') - seas
        Tanom = Tanom.rename('T_anom')
        Tanom.attrs['name'] = 'Temperature_anomalies'
        Tanom.attrs['units'] = '°C'
        # Compute severity. Make sure the denominator is not 0
        severity = Tanom.groupby('time.dayofyear') / (thresh - seas + 1e-9)
        severity = severity.rename('severity')
        severity.attrs['name'] = 'Severity'
        severity.attrs['units'] = '-'
        severity.attrs['description'] = 'Severity = T_anom / (thresh - clim)'

        # Put them together in one dataset
        ds_for_detection = xr.Dataset({
            'T_anom': Tanom,
            'severity': severity,
            'time': da.time
        }).drop_vars('dayofyear')
        
        return ds_for_detection

    