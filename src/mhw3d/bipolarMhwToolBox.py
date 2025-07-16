import xarray as xr
import numpy as np
import datetime as dt
import bottleneck as bn
import marineHeatWaves as mhw
import pandas as pd
from pathlib import Path

def detectMHWs_fromSeverity(time, ts_severity, l_return=200, minDuration=5, maxGap=2):
    """
    Adapt the marineheatwaves function from Eric to work on the severity. 
    This function is called in a apply_ufunc call, so that it can be parallelized.
    Identify the presence, start and end dates of the mhw on each grid cell.
    The rest of the metrics are calculated using the function `add_metrics_MHWs`.
    Inputs: 
        - time: np.datetime64 vector
        - ts_severity: Severity = SSTa/(Thresh - Seas). (as a vector)
        - l_return: the maximum number of expected MHW events (to pre-allocate the output).
        - minDuration and maxGap: definition-specific parameters.
    Output: 
        - Numpy array with the following variables: index_start, index_end, date_start, date_end, duration.
          It needs to be a numpy array, because of the apply_ufunc call.
    """
    # Create boolean where severity indicates a potential MHWs
    t = time
    boolSever = ts_severity > 1
    # Preallocate the mhw dictionary
    mhw = {}
    mhw['index_start'] = np.empty(l_return) * np.nan
    mhw['index_end'] = np.empty(l_return) * np.nan
    mhw['date_start'] = np.empty(l_return) * np.nan
    mhw['date_end'] = np.empty(l_return) * np.nan
    mhw['duration'] = np.empty(l_return) * np.nan
    # Convert to a np.array for vectorization
    ia = np.asarray(boolSever)                # force numpy
    n = len(ia)       # Length of the array, along time dim
    if ~np.isnan(ia).all(): # Make sure it is not a nan array
        # This a very neat and smart way to find consecutive repetitions
        # Combined with some stackoverflow stuff.
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        l_cont = np.diff(np.append(-1, i))       # run lengths
        p = np.append(0,i+1)[:-1]  #np.cumsum(np.append(0, l_cont))[:-1] # positions
        type = ia[i]
        # Filter for mhws (longer than 5 days, and severity >=1)
        b_mhw = np.where((l_cont >= minDuration) & (type == 1))[0]
        # Extract the first index and the last index of each MHW
        i_start = p[b_mhw]
        i_end = i[b_mhw]
        # Find if gap after mhw is less than 2days and include in mhw if so
        l_gap = i_start[1:] - i_end[:-1] - 1
        boolGap = l_gap <= maxGap
        if any(boolGap):
            indGap = np.where(boolGap)[0]
            ibeg_mhw = np.delete(i_start,indGap+1)
            iend_mhw = np.delete(i_end,indGap)
        else:
            ibeg_mhw = i_start
            iend_mhw = i_end
        # Now can take care of including more stuff in the dictionary
        n_events = len(ibeg_mhw)
        mhw['index_start'][:n_events] = ibeg_mhw
        mhw['index_end'][:n_events] = iend_mhw
        mhw['date_start'][:n_events] = t[ibeg_mhw]
        mhw['date_end'][:n_events] = t[iend_mhw]
        mhw['duration'][:n_events] = iend_mhw - ibeg_mhw + 1
    # Need to return a dataset
    ds_mhw = xr.Dataset({var: ('event', data) for var, data in mhw.items()},
                        coords={'event':('event',range(l_return))})
    return ds_mhw.to_array()

def add_metrics_MHWs(time, ssta, date_start, date_end, l_return=200):
    """
    Calculate more metrics, such as peak date and intensities.  Could add more in this function.
    This function is also called in an apply_ufunc call, to parallelize things.
    Inputs:
        - time: a np.datetime64 vector
        - ssta: sea surface temperature anomaly (vector)
        - date_start: np.datetime64 vector containing the start date of all detected MHW events (for this one ssta)
        - date_end: np.datetime64 vector containing the end date of all detected MHW events (for this one ssta)
        - l_return: the maximum number of expected MHW events (to pre-allocate the output).
    Output:
        - Numpy array containing more metrics, including peak date and various intensities.
    """
    # Force numpy arrays
    time_np = time                # time
    ssta_np = np.asarray(ssta)                # ssta
    date_start_np = np.asarray(date_start)    # date_start
    date_end_np = np.asarray(date_end)        # date_end
    # Make sure it is only 2D, i.e. only one grid cell.
    if len(ssta_np.shape)>2:
        print('Dataset has too many dimensions, computation will be too heavy and crash.')
        print('Try to vectorize or loop over grid cells.')
        return
    # Start a new dictionary
    mhw_out = {}
    mhw_out['index_peak'] = np.empty(l_return) * np.nan
    mhw_out['date_peak'] = np.empty(l_return) * np.nan
    mhw_out['intensity_max'] = np.empty(l_return) * np.nan
    mhw_out['intensity_mean'] = np.empty(l_return) * np.nan
    mhw_out['intensity_cumul'] = np.empty(l_return) * np.nan
    if ~np.isnan(ssta_np).all(): # Make sure it is not a nan array
        boolnan = ~np.isnan(date_start_np)
        dstart = date_start_np[boolnan]
        dend = date_end_np[boolnan]
        # For each event, create a time series of ssta with all values out of MHWs naned-out
        np_ssta_evts = np.where((time_np[:, np.newaxis]>=dstart) & (time_np[:, np.newaxis]<=dend),
                                ssta_np[:, np.newaxis],np.nan)
        # Use the argmax to extract the ssta maximum during the mhw, then convert to a date
        n_events = bn.nansum(boolnan)
        index_peak = bn.nanargmax(np_ssta_evts,axis=0)
        mhw_out['index_peak'][:n_events] = index_peak
        date_peak = time_np[index_peak]
        mhw_out['date_peak'][:n_events] = date_peak
        # # Now calculate intensities
        mhw_out['intensity_max'][:n_events] = bn.nanmax(np_ssta_evts,axis=0)
        mhw_out['intensity_mean'][:n_events] = bn.nanmean(np_ssta_evts,axis=0)
        mhw_out['intensity_cumul'][:n_events] = bn.nansum(np_ssta_evts,axis=0)
        # Finally convert to a ds to send back.
    ds_mhw_out = xr.Dataset({var: ('event', data) for var, data in mhw_out.items()},
                            coords={'event':('event',range(l_return))})
    return ds_mhw_out.to_array()

def calculate_MHWs_metrics(ds, maxEvt=200):
    """
    Use apply_ufunc to vectorize calculations and use both previous functions onto several grid cells at once.
    Inputs:
        - ds: xr.Dataset containing coordinate `time` (np.datetime64 format), and variable `severity`. 
          It can contain more variables, those will simply be ignored. 
          It should contain more coordinates (typically lat and lon or x and y if curvi-linear; also depth),
          else this function is useless and the classic marineheatwaves function might as well be called.
    Outputs:
        - ds_mhw_full: an xr.Dataset containing MHW metrics for each grid cell of the input Dataset.
    """
    # Detect the marine heatwaves
    mhw = xr.apply_ufunc(detectMHWs_fromSeverity, ds.time, ds.severity, kwargs={'l_return':maxEvt},
                         input_core_dims=[['time'],['time']],
                         output_core_dims=[['variable','event']],
                         dask="parallelized",
                         dask_gufunc_kwargs={"output_sizes": {"event": maxEvt, "variable": 5}},
                         vectorize=True)
    # mhw = mhw.compute() # Do I need to compute now? Or can I wait?
    # Convert the output, a DataArray, into a Dataset with variables.
    ds_mhw = mhw.assign_coords({'variable':('variable',['index_start','index_end','date_start','date_end','duration']),
                                'event':('event',range(maxEvt))}).to_dataset(dim='variable')
    # So far, I have padded each grid cell to be able to have a cubic dataset. Drop the extra nan-slices
    ds_mhw = ds_mhw.dropna('event',how='all') 
    # Convert the dates from numpy numbers to human-readable dates
    ds_mhw['date_start'] = ds_mhw.date_start.astype('datetime64[ns]')
    ds_mhw['date_end'] = ds_mhw.date_end.astype('datetime64[ns]')
    # Calculate the number of events
    ds_mhw['n_event'] = ds_mhw.index_start.notnull().sum(dim='event')
    # Now apply the other function to add some more metrics, using the ssta
    ds_mhw_more = xr.apply_ufunc(add_metrics_MHWs, ds.time, ds.ssta, 
                                 ds_mhw.date_start, ds_mhw.date_end, 
                                 kwargs={'l_return':len(ds_mhw.event)},
                                 input_core_dims=[['time'],['time'],['event'],['event']],
                                 output_core_dims=[['variable','event']],
                                 dask="parallelized",
                                 dask_gufunc_kwargs={"output_sizes": {"variable": 5, "event": len(ds_mhw.event)}},
                                 output_dtypes=['float'],
                                 vectorize=True)
    # Convert from a dataArray to a Dataset
    ds_mhw_more = ds_mhw_more.assign_coords({'variable':('variable',
                                                         ['index_peak','date_peak','intensity_max',
                                                          'intensity_mean','intensity_cumul']),
                                             'event':('event',range(len(ds_mhw.event)))}).to_dataset(dim='variable')
    # Merge both datasets
    ds_mhw_more['date_peak'] = ds_mhw_more.date_peak.astype('datetime64[ns]')
    ds_mhw_full = ds_mhw.merge(ds_mhw_more)
    return ds_mhw_full

def smoothedClima_mhw(ds):
    """
    Replicate the climatology calculation used in the marineHeatWave.py algorithm.
    Input:
        - ds: xr.DataArray containing the temperature time series.
    Output:
        - ds_smoothedClim: a xr.DataArray of same size as ds, except for the `time` dimension that became a
        `dayofyear` dimension of size 366. This dataset contains the smoothed seasonal cycle.
    """
    if "time" not in ds.dims:
        print("No 'time' dimensions in the dataset")
        return

    # Remove the preliminary 11-day smoothing to match the Oliver method.
    ds_clim_doy = ds.groupby("time.dayofyear").mean()

    # Stack 3 years of the climatology together for seamless smoothing.
    # Note: .stack() creates coordinates from the multi-index names ('year', 'dayofyear').
    stackedClim = xr.concat([ds_clim_doy, ds_clim_doy, ds_clim_doy], dim='year').stack(time={'year','dayofyear'})

    # --- MODIFICATION: Use the correct coordinate name for sorting ---
    # The coordinate is named 'dayofyear', not 'doy'.
    stackedClim = stackedClim.sortby(['year', 'dayofyear'])

    # Smooth the time series
    smoothedClim = stackedClim.rolling(time=31,
                                       min_periods=1,
                                       center=True).mean()

    # Extract the middle year and rearrange the dimensions to only keep "dayofyear"
    tmpSmooClim = smoothedClim.where(smoothedClim.year==1, drop=True).drop_vars('year')
    ds_smoothedClim = tmpSmooClim.rename({'time':'dayofyear'}).assign_coords(dayofyear=ds_clim_doy.dayofyear.data)
    
    return ds_smoothedClim

def smoothedThresh_mhw(ds, pctile=0.9, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31):
    """
    Replicate the threshold calculation used in the marineHeatWave.py algorithm.
    Note this code is unlikely to work if there are NaNs.
    Input:
        - ds: xr.Dataset containing (at least) `time` (as np.datetime64 format) and any variable over 
              which the climatology should be calculated (e.g. SST, ice concentration, etc.)
        - pctile: Threshold percentile (%) for detection of extreme values (default=0.9).
        - windowHalfWidth: Width of window (one sided) about day-of-year used for the pooling of values
                           and calculation of threshold percentile (default = 5 [days])
        - smoothPercentile: Boolean switch indicating whether to smooth the threshold percentile timeseries
                            with a moving average (default = True)
        - smoothPercentileWidth: Width of moving average window for smoothing threshold (default = 31 days)

    Output:
        - ds_smoothedClim: a xr.Dataset of same size as ds, except for the `time` dimension that became a 
        `dayofyear` dimension of size 366. This dataset contains the threshold, as a percentile, to detect
        marine heatwaves.
    Note that contrarily to the marineheatwave algorithm, the 366th day is not inserted on 
    the 29th of February via interpolation but is calculated as the other days and is inserted on
    the 31st of December.
    """
    if ~np.isin("time",ds.dims):
        print("No 'time' dimensions in the dataset")
        return
    # Generate an empty DataArray to store the threshold results
    ## TODO: Adapt this to any kind of input array, including 3D dataset. Maybe use xr.zeros_like(ds)?
    thresh = xr.DataArray(dims=["lat", "lon", "doy"],
                          coords={'lon':("lon", ds.lon.data),
                                  'lat':("lat", ds.lat.data),
                                  'doy': ("doy", np.arange(1,366+1))})
    # Convert time into dayofyear (doy)
    doy = ds.time.dt.dayofyear
    # ======= DEVELOPING COMMENTS =========================
    # 1. Quantile loop: this is likely a major bottleneck in the script. 
    # Tests showed that the quantile function is the step taking most time in the whole process. Replacing would help.
    # 2. Also, the use of the for loop is of course not ideal. There might be ways to use some convolution to speed up the process.
    # This would require some proper thinking. 
    # 3. Another thing is that the loop is poorly constructed at the moment, with tt=1 coresponding actually at doy=6 by construction
    # It would be better to adjust by maybe using a condition like:
    # `ds.where((doy >= tt - windowHalfWidth) & (doy < tt + windowHalfWidth + 1)`
    # There might also be room for avoiding the if condition if I consistently use modulos for both sides of the condition.
    # But I am not sure about that, I think leap years are again messing everything up.
    # ============= BR, 02/07/2025 ========================
    windowFullWidth = windowHalfWidth * 2 + 1
    # Now loop over the doy of a climatological year
    for tt in np.arange(1,366+1):
        if tt < 366 - windowFullWidth: # If the window is entirely below 366
            # Get the window normally, removing all values outside the time-window
            ds_window = ds.where((doy >= tt) & (doy < tt + windowFullWidth),drop=True)
        else: # If the window includes days above 366
            # Use modulo to wrap around the 1st of January and remove all values outside the time-window
            ds_window = ds.where((doy >= tt) | (doy < (tt + windowFullWidth)%365),drop=True)
        # Now, calculate the percentile for this window, for all years.
        ## TODO: skipna=False is supposed to speed things up dramatically, but is not adapted to irregular time series.
        # Need to adapt this depending on whether there are nans, maybe through user-provided argument?
        qt = ds_window.quantile(pctile,dim='time',skipna=False)
        # Add the percentile to the treshold array for that specific doy
        thresh.loc[{'doy':thresh.doy==tt}] = qt.drop_vars('quantile').expand_dims('doy',axis=-1)
    # Because of the way the loop is constructed, the index tt=1 actually corresponds to the percentile of doy=6.
    # So need to roll this by 5 days to align properly. NOTE: Might be able to remove this by modifying the loop.
    thresh = thresh.roll(doy=windowHalfWidth)
    # Now, to smooth everything, need to concatenate 3 years so that boundaries are accounted for.
    if smoothPercentile:
        stackedThresh = xr.concat([thresh,thresh,thresh],dim='year').stack(time={'year','doy'}).sortby(['year', 'doy']) # added .sortby command 
        # Smooth the time series by using a rolling average.
        smoothedThresh = stackedThresh.rolling(time=31, min_periods=1, center=True).mean()
        # Extract the middle year and rearrange the dimensions to only keep "dayofyear"
        # SmoothThresh = smoothedThresh.where(smoothedThresh.year==1, drop=True).drop('year').rename({'time':'dayofyear'}) # Old method
        SmoothThresh = smoothedThresh.sel(year=1).drop_vars('year').rename({'doy':'dayofyear'}) # This seems to work as well if not better.
    return SmoothThresh