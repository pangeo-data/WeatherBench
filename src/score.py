"""
Functions for evaluating forecasts.
"""
import numpy as np
import xarray as xr

def load_test_data(path, var, years=slice('2017', '2018')):
    """
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window

    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    return ds.sel(time=years)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.

    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    rename_dict = {v: v + '_rmse' for v in rmse} if type(rmse) is xr.Dataset else {rmse.name: rmse.name +'_rmse'}
    return rmse.rename(rename_dict)

def evaluate_iterative_forecast(fc_iter, da_valid):
    rmses = []
    for lead_time in fc_iter.lead_time:
        fc = fc_iter.sel(lead_time=lead_time)
        fc['time'] = fc.time + np.timedelta64(int(lead_time), 'h')
        rmses.append(compute_weighted_rmse(fc, da_valid))
    return xr.concat(rmses, 'lead_time')
    # return xr.DataArray(rmses, dims=['lead_time'], coords={'lead_time': fc_iter.lead_time})


