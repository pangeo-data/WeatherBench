"""
Functions for evaluating forecasts.
"""
import numpy as np
import xarray as xr

def load_test_data(path, var, years=slice('2017', '2018')):
    """
    Load the test dataset. If z return z500, if t return t850.
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window

    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    if var in ['z', 't']:
        if len(ds["level"].dims) > 0:
            try:
                ds = ds.sel(level=500 if var == 'z' else 850) .drop('level')
            except ValueError:
                ds = ds.drop('level')
        else:
            assert ds["level"].values == 500 if var == 'z' else ds["level"].values == 850
    return ds.sel(time=years)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse

def compute_weighted_acc(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    clim = da_true.mean('time')
    try:
        t = np.intersect1d(da_fc.time, da_true.time)
        fa = da_fc.sel(time=t) - clim
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            np.sum(w * fa_prime * a_prime) /
            np.sqrt(
                np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
            )
    )
    return acc

def compute_weighted_mae(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the MAE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        mae: Latitude weighted root mean absolute error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    mae = (np.abs(error) * weights_lat).mean(mean_dims)
    return mae


def evaluate_iterative_forecast(da_fc, da_true, func, mean_dims=xr.ALL_DIMS):
    """
    Compute iterative score (given by func) with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Iterative Forecast. Time coordinate must be initialization time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        score: Latitude weighted score
    """
    rmses = []
    for f in da_fc.lead_time:
        fc = da_fc.sel(lead_time=f)
        fc['time'] = fc.time + np.timedelta64(int(f), 'h')
        rmses.append(func(fc, da_true, mean_dims))
    return xr.concat(rmses, 'lead_time')


