import xarray as xr
import numpy as np
import pandas as pd

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
        for v in obj.data_vars:
            if 'time' in obj[v].dims:
                return obj[v]
        return next(iter(obj.data_vars.values()))
    raise TypeError("Expected xarray DataArray or Dataset")


def _detect_events(time, ts_severity, l_return=200, minDuration=5, maxGap=2):
    """
    Identify MHW events from a severity timeseries (severity = T_anom / (thresh - seas)).
    Returns a DataArray (variable, event) with: index_start, index_end, date_start, date_end, duration.
    NOTE: dates are returned as floats (ordinal-like) to satisfy apply_ufunc's single dtype;
          they are cast back to datetime64[ns] later in calculate_mhw_metrics.
    """
    boolSever = ts_severity > 1

    mhw = {}
    mhw['index_start'] = np.full(l_return, np.nan, dtype=float)
    mhw['index_end']   = np.full(l_return, np.nan, dtype=float)
    mhw['date_start']  = np.full(l_return, np.nan, dtype=float)
    mhw['date_end']    = np.full(l_return, np.nan, dtype=float)
    mhw['duration']    = np.full(l_return, np.nan, dtype=float)

    ia = np.asarray(boolSever)
    n = len(ia)

    if (~np.isnan(ia)).any():
        y = ia[1:] != ia[:-1]
        i  = np.append(np.where(y), n - 1)
        l_cont = np.diff(np.append(-1, i))
        run_ends = i
        run_starts = i - l_cont + 1

        run_mask = ia[run_ends]
        i_start = run_starts[run_mask]
        i_end   = run_ends[run_mask]

        keep = (i_end - i_start + 1) >= minDuration
        i_start = i_start[keep]
        i_end   = i_end[keep]

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


def _compute_event_metrics(time, T_anom, date_start, date_end, l_return=200):
    """
    Compute peak date and intensities (max, mean, cumulative) for each detected event.
    NOTE: date_peak is returned as float (ordinal-like) to satisfy apply_ufunc's single dtype;
          it is cast back to datetime64[ns] later in calculate_mhw_metrics.
    """
    time_np   = np.asarray(time)
    T_anom_np = np.asarray(T_anom)

    mhw_out = {
        'index_peak':      np.full(l_return, np.nan, dtype=float),
        'date_peak':       np.full(l_return, np.nan, dtype=float),
        'intensity_max':   np.full(l_return, np.nan, dtype=float),
        'intensity_mean':  np.full(l_return, np.nan, dtype=float),
        'intensity_cumul': np.full(l_return, np.nan, dtype=float),
    }

    dstart = np.asarray(date_start)
    dend   = np.asarray(date_end)
    valid_evt = ~np.isnan(dstart) & ~np.isnan(dend)
    if valid_evt.any():
        mask = (time_np[:, None] >= dstart[None, :]) & (time_np[:, None] <= dend[None, :])
        np_T_anom_evts = np.where(mask, T_anom_np[:, None], np.nan)

        valid_cols = ~np.isnan(np_T_anom_evts).all(axis=0)
        if valid_cols.any():
            idx_peak_cols = np.nanargmax(np_T_anom_evts[:, valid_cols], axis=0)
            idx_peak = np.full(np_T_anom_evts.shape[1], 0, dtype=int)
            idx_peak[valid_cols] = idx_peak_cols

            mhw_out['index_peak'][valid_evt] = idx_peak[valid_evt]
            mhw_out['date_peak'][valid_evt]  = time_np[idx_peak][valid_evt]
            mhw_out['intensity_max'][valid_evt]   = np.nanmax(np_T_anom_evts[:, valid_cols], axis=0)
            mhw_out['intensity_mean'][valid_evt]  = np.nanmean(np_T_anom_evts[:, valid_cols], axis=0)
            mhw_out['intensity_cumul'][valid_evt] = np.nansum(np_T_anom_evts[:, valid_cols], axis=0)

    ds_mhw_out = xr.Dataset({var: ('event', data) for var, data in mhw_out.items()},
                            coords={'event': ('event', np.arange(l_return))})
    return ds_mhw_out.to_array()


def calculate_mhw_metrics(ds, maxEvt=200):
    """
    Vectorized detection + metrics over grids. Expects ds to contain:
      - time (coordinate)
      - either:
           (a) variables 'T_anom' and 'severity', or
           (b) variables 'Seas' and 'Thresh' and a temperature variable,
               from which 'T_anom' and 'severity' will be derived.
    Returns an xr.Dataset with event-wise metrics per grid cell.
    """
    work = ds.copy()

    def _pick_temp_var(_ds):
        for k in ('sst', 'temp', 'thetao', 'tas', 'temperature', 'T', 'SST'):
            if k in _ds.data_vars and 'time' in _ds[k].dims:
                return k
        for k, v in _ds.data_vars.items():
            if 'time' in v.dims:
                return k
        raise ValueError("Could not find a temperature-like variable with 'time' in dims.")

    if 'T_anom' not in work:
        if 'Seas' in work:
            tvar = _pick_temp_var(work)
            doy = work['time'].dt.dayofyear
            work['T_anom'] = work[tvar] - work['Seas'].sel(dayofyear=doy)
        else:
            raise ValueError("Missing 'T_anom' and no 'Seas' to compute it from.")

    if 'severity' not in work:
        if 'Thresh' in work and 'Seas' in work:
            doy = work['time'].dt.dayofyear
            denom = work['Thresh'].sel(dayofyear=doy) - work['Seas'].sel(dayofyear=doy)
            work['severity'] = work['T_anom'] / denom
        else:
            raise ValueError("Missing 'severity' and cannot compute it (need 'Thresh' and 'Seas').")

    mhw = xr.apply_ufunc(
        _detect_events, work.time, work.severity,
        kwargs={'l_return': maxEvt},
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['variable', 'event']],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"variable": 5, "event": maxEvt}},
        output_dtypes=['float'],
        vectorize=True
    )
    ds_mhw = mhw.assign_coords({'variable': ('variable',
                    ['index_start', 'index_end', 'date_start', 'date_end', 'duration'])}).to_dataset(dim='variable')
    ds_mhw['date_start'] = ds_mhw.date_start.astype('datetime64[ns]')
    ds_mhw['date_end']   = ds_mhw.date_end.astype('datetime64[ns]')

    ds_mhw_more = xr.apply_ufunc(
        _compute_event_metrics, work.time, work.T_anom,
        ds_mhw.date_start, ds_mhw.date_end,
        kwargs={'l_return': maxEvt},
        input_core_dims=[['time'], ['time'], ['event'], ['event']],
        output_core_dims=[['variable', 'event']],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"variable": 5, "event": maxEvt}},
        output_dtypes=['float'],
        vectorize=True
    ).assign_coords({'variable': ('variable',
                    ['index_peak', 'date_peak', 'intensity_max', 'intensity_mean', 'intensity_cumul'])}).to_dataset(dim='variable')

    ds_mhw_more['date_peak'] = ds_mhw_more.date_peak.astype('datetime64[ns]')

    return xr.merge([ds_mhw, ds_mhw_more])


def calculate_severity(obj, seas, thresh, varname=None):
    """
    Severity = T_anom / (thresh - seas), following Hobday et al. 2018.
    Severity > 1 corresponds to an MHW state.

    Inputs:
        obj     : xr.DataArray or Dataset containing temperature [°C]
        seas    : xr.DataArray, climatological seasonal cycle (dayofyear dim) [°C]
        thresh  : xr.DataArray, climatological threshold (dayofyear dim) [°C]
        varname : variable name if obj is a Dataset

    Returns xr.Dataset with 'T_anom' [°C] and 'severity' [-].
    """
    da = _to_da(obj, varname)
    if "time" not in da.dims:
        raise ValueError("No 'time' dimension found.")

    T_anom = da.groupby('time.dayofyear') - seas
    T_anom = T_anom.rename('T_anom')
    T_anom.attrs['name'] = 'Temperature_anomalies'
    T_anom.attrs['units'] = '°C'

    severity = T_anom.groupby('time.dayofyear') / (thresh - seas + 1e-9)
    severity = severity.rename('severity')
    severity.attrs['name'] = 'Severity'
    severity.attrs['units'] = '-'
    severity.attrs['description'] = 'Severity = T_anom / (thresh - seas)'

    return xr.Dataset({
        'T_anom': T_anom,
        'severity': severity,
        'time': da.time
    }).drop_vars('dayofyear')
