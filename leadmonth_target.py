import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
from datetime import datetime
import glob
import pickle
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

LOCS = np.load('/media/cmlws/Data1/jsp/DLdata/1/test/chl_latlon_mask_1x1.npy')

LON_SIZE = 360
LAT_SIZE = 180

train_path = '/media/cmlws/Data1/jsp/DLdata/cmip6/'
target_train_files = sorted(glob.glob(train_path + 'sfc_chl_Omon_*.nc'))

valid_path = "/media/cmlws/Data1/jsp/DLdata/1/transfer/"
target_valid_files = sorted(glob.glob(valid_path+"sfc_chl_*196501*.nc"))

test_path = "/media/cmlws/Data1/jsp/DLdata/1/test/"
target_test_files = sorted(glob.glob(test_path+"chl_199801-201812_1x1.nc"))


def chk_var(klist): 
    var_name_list = ['eta_t','sshg','ssh','SSH','chl','sfc_chl','chlos','tos','sst','SST']
    var = list(set(klist).intersection(var_name_list))
    return var


def log_transform(arr):
    epsilon=1e-06 
    log_norm = np.log(arr+epsilon)
    return log_norm

def calculate_anomaly(da, groupby_type="time.month"):
    gb = da.groupby(groupby_type)
    clim = gb.mean(dim="time")
    return gb - clim

def calculate_stand_anomaly(da, groupby_type="time.month"):
    gb = da.groupby(groupby_type)
    clim = gb.mean(dim="time")
    clim_s =gb.std(dim="time")
    stand_anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        gb,
        clim,
        clim_s,
    )
    return stand_anomalies

def lme_mask(ds, masknumber):
    # Select lme region by number
    lme_da = ds.mask.where(ds.mask == masknumber)
    # Creating lat lon index 
    lon_ind = np.arange(360)
    lat_ind = np.arange(180)
    # Substitute coordinates with indices for using vectorized access
    lme_da['lon'] = lon_ind
    lme_da['lat'] = lat_ind
    # Get a set of indices by StackExchange
    da_stacked = lme_da.stack(yx=['lat','lon'])
    index = da_stacked[da_stacked.notnull()].indexes['yx']
    mask_locs = xr.DataArray(data=np.array(tuple(index.values)))
    return mask_locs


def weighted_yearly_mean(ds, var, mask_locs):
    """
    weight by latitude 
    """
    # Subset our dataset for our variable
    obs = ds[var].sel()
    dim = list(obs.dims)
    shape =  ['time','lon','lat']
    obs = obs.squeeze(list(set(dim).difference(set(shape))))
    # Select lme grid by mask_locs
    lmeobs = obs[:,mask_locs[:, 0], mask_locs[:, 1]] 
    print(f'# Total number of lme grid: {len(mask_locs)}')
    # lmeobs = log_transform(lmeobs)
    lmeobs = calculate_anomaly(lmeobs)

    # Resample by year 
    resampled = lmeobs.resample(time='AS').mean('time')
    # Creating weights
    #For a rectangular grid the cosine of the latitude is proportional to the grid cell area.
    weights = np.cos(np.deg2rad(resampled.lat))
    weights.name = "weights"
    # Return the weighted average
    lme_weighted = resampled.weighted(weights)
    return lme_weighted.mean('dim_0').isel(time=slice(1, None))#djf

def season_mean_anomaly(ds, var, mask_locs, tm):
    """
    returns 3 month rolling mean anomaly xarray.DataArray.
    Parameters
    ------------------------
    tm = target month 
    tm == 0(DJF) or tm == 1(NDJ)
    """
    obs = ds[var].sel()
    dim = list(obs.dims)
    shape =  ['time','lon','lat']
    obs = obs.squeeze(list(set(dim).difference(set(shape))))
    # Select lme grid by mask_locs
    lmeobs = obs[:,mask_locs[:, 0], mask_locs[:, 1]] 
    print(f'# Total number of lme grid: {len(mask_locs)}')

    lmeobs = log_transform(lmeobs)
    lmeobs = calculate_stand_anomaly(lmeobs)
    lmeobs = lmeobs.fillna(0)
    anom = lmeobs.rolling(time=3, center=True).mean()
    resampled = anom[tm-1::12]
    weights = np.cos(np.deg2rad(resampled.lat))
    weights.name = "weights"
    # Return the weighted average
    lme_weighted = resampled.weighted(weights)

    return lme_weighted.mean('dim_0').dropna(dim='time')

def label_generator(path, mask, tm):
    y_basename = os.path.basename(path)
    y_file_name = os.path.splitext(y_basename)[0]
    data = xr.open_dataset(path).load()
    print(f"{y_file_name} opened")
    k = list(data.keys())
    v = chk_var(k)
    maskds = xr.open_dataset("/media/cmlws/Data1/jsp/DLdata/LME66.mask.nc").load()
    locs = lme_mask(maskds, mask)
    lme = season_mean_anomaly(data, v[0], locs, tm)
    outdir = f"/media/cmlws/Data2/jsp/LMEinput/target_month/{tm}/{mask}/"
    os.makedirs(outdir, exist_ok=True)
    lme.to_netcdf(f"{outdir}{y_file_name}_target{tm}.nc")
    print(f"{y_file_name} LME{mask} target month {tm} saved")
    return lme

def yearly_label_generator(path, mask):
    y_basename = os.path.basename(path)
    y_file_name = os.path.splitext(y_basename)[0]
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    maskds = xr.open_dataset("/media/cmlws/Data1/jsp/DLdata/LME66.mask.nc").load()
    locs = lme_mask(maskds, mask)
    lme = weighted_yearly_mean(data, v[0], locs)
    outdir = f"/media/cmlws/Data2/jsp/LMEinput/anom_yearly_mean/DJF/{mask}/"
    os.makedirs(outdir, exist_ok=True)
    lme.to_netcdf(f"{outdir}{y_file_name}_target{mask}.nc")
    print(f"{y_file_name} LME{mask} target saved")
    return lme

def prepare_tm_labels(path, mask, tm):
    """
    returns target months lme indexes from files
    Parameters
    ------------------------
    path = files path
    tm = target month
    """

    temp_labels = [] #lead months nino index xarray list 
    # For each video.
    for f in (path):
        # Gather all its frames and add a batch dimension.
        label = label_generator(f, mask, tm)
        temp_labels.append(label)
    # frame_features = [el.interpolate_na(dim="lon") for el in temp_frames]    

    labels = np.concatenate(temp_labels, axis=0)
    labels = labels[:,np.newaxis]
    return labels

def prepare_yr_labels(path, mask):
    """
    returns target months lme indexes from files
    Parameters
    ------------------------
    path = files path
    tm = target month
    """

    temp_labels = [] #lead months nino index xarray list 
    # For each video.
    for f in (path):
        # Gather all its frames and add a batch dimension.
        label = yearly_label_generator(f, mask)
        temp_labels.append(label)
    # frame_features = [el.interpolate_na(dim="lon") for el in temp_frames]    

    labels = np.concatenate(temp_labels, axis=0)
    labels = labels[:,np.newaxis]
    return labels

# for lme in np.arange(1,67):
#     for tm in np.arange(1,13):
#         outdir = f"/media/cmlws/Data2/jsp/LMEinput/target_month/{tm}/{lme}/"
#         os.makedirs(outdir, exist_ok=True)
#         print(f"target month {tm} LME {lme} target file processing")
#         train_y = prepare_tm_labels(target_train_files, lme, tm)
#         np.save(f"{outdir}tr_y.npy", train_y)
#         valid_y = prepare_tm_labels(target_valid_files, lme, tm)
#         np.save(f"{outdir}val_y.npy", valid_y)
#         test_y = prepare_tm_labels(target_test_files, lme, tm)
#         np.save(f"{outdir}test_y.npy", test_y)
#         print(f"target month {tm} LME {lme} target file processed")

for lme in np.arange(1,67):
    
    outdir = f"/media/cmlws/Data2/jsp/LMEinput/anom_yearly_mean/DJF/{lme}/"
    os.makedirs(outdir, exist_ok=True)
    print(f"target month LME {lme} target file processing")
    train_y = prepare_yr_labels(target_train_files, lme)
    np.save(f"{outdir}tr_y.npy", train_y)
    valid_y = prepare_yr_labels(target_valid_files, lme)
    np.save(f"{outdir}val_y.npy", valid_y)
    test_y = prepare_yr_labels(target_test_files, lme)
    np.save(f"{outdir}test_y.npy", test_y)
    print(f"target LME {lme} target file processed")