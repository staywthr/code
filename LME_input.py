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


###train
train_path = '/media/cmlws/Data1/jsp/DLdata/cmip6/'
chl_train_files = sorted(glob.glob(train_path + 'sfc_chl_Omon_*.nc'))
sst_train_files = sorted(glob.glob(train_path + 'tos_Omon_*.nc'))
target_train_files = sorted(glob.glob(train_path + 'sfc_chl_Omon_*.nc'))

print(f"Total cmip6 models(chl) for training: {len(chl_train_files)}")
print(f"Total cmip6 models(sst) for training: {len(sst_train_files)}")

####valid
valid_path = "/media/cmlws/Data1/jsp/DLdata/1/transfer/"
chl_valid_files = sorted(glob.glob(valid_path+"sfc_chl_*196501*.nc"))
sst_valid_files = sorted(glob.glob(valid_path+"sst_*196501*.nc"))
target_valid_files = sorted(glob.glob(valid_path+"sfc_chl_*196501*.nc"))

test_path = "/media/cmlws/Data1/jsp/DLdata/1/test/"
chl_test_files = sorted(glob.glob(test_path+"chl_199801-201812_1x1.nc"))
sst_test_files = sorted(glob.glob(test_path+"sst_JAN1998*.nc"))
target_test_files = sorted(glob.glob(test_path+"chl_199801-201812_1x1.nc"))

def chk_var(klist): 
    var_name_list = ['eta_t','sshg','ssh','SSH','chl','sfc_chl','chlos','tos','sst','SST']
    var = list(set(klist).intersection(var_name_list))
    return var

def load_xdata(path):
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    arr = data[v[0]].sel()
    dim = list(arr.dims)
    shape =  ['time','lon','lat']
    if v == ['chl'] and len(arr.dims) == 4:
        arr = arr.squeeze(list(set(dim).difference(set(shape))))
    return arr.transpose('time','lon','lat')



def cleanining_grid(da, locnpy):
    for i in range(len(locnpy)):
        da[:,locnpy[i][0],locnpy[i][1]] = np.nan 
    print(f'# Total number of missing grid: {len(locnpy)}')
    return da

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


def month_anomaly(var_anom, mind):
    xda = var_anom.where(var_anom.coords['month']==mind,drop=True)
    return xda


def season_anomaly(var_anom,ld):
    """
    returns 3 month anomaly xarray.DataArray.
    Parameters
    ------------------------
    ld = lead month 
    ld == 0(DJF) or ld == 1(NDJ)
    """
    if ld == 1:
        anom0 = month_anomaly(var_anom, 11)
        anom1 = month_anomaly(var_anom, 12)
        anom2 = month_anomaly(var_anom, 1)
        ldmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)),anom2.isel(time=slice(1, None))], dim='z', join='override')
    else:
        # ld == 0
        anom0 = month_anomaly(var_anom, 12)
        anom1 = month_anomaly(var_anom, 1)
        anom2 = month_anomaly(var_anom, 2)
        ldmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)),anom2.isel(time=slice(1, None))], dim='z', join='override')

    return ldmn.transpose('time','lon','lat','z')

def season_mean_anomaly(var_anom,ld):
    """
    returns 3 month rolling mean anomaly xarray.DataArray.
    Parameters
    ------------------------
    ld = lead month 
    ld == 0(DJF) or ld == 1(NDJ)
    """
    anom = var_anom.rolling(time=3, center=True).mean()
    ldmn = anom[ld::12]
    ldmn = ldmn.isel(time=slice(0,-1))

    return ldmn.transpose('time','lon','lat')


#xarray 
def log_transform(arr):
    epsilon=1e-06 
    log_norm = np.log(arr+epsilon)
    return log_norm



def prepare_single_frame(path, ld): 
    """
    returns lead months single feature(chl/sst/ssh) xarray from one file
    Parameters
    ------------------------
    path = file path
    ld = lead month
    """
    arr = load_xdata(path)
    if (arr.name == 'chl') or (arr.name == 'chlos') or (arr.name == 'sfc_chl'):
        print("chlorophyll preprocess")
        arr = cleanining_grid(arr, LOCS)
        arr = log_transform(arr)           
    # anom = calculate_anomaly(arr)
    anom = calculate_stand_anomaly(arr)
    ss_anom = season_anomaly(anom, ld)
    # ss_anom = season_mean_anomaly(anom,ld)
    print(f"Lead month {ld} {ss_anom.name} {(path.partition('Omon_')[-1])} Dataset")
    return ss_anom


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
    lmeobs = log_transform(lmeobs)
    lmeobs = calculate_stand_anomaly(lmeobs)
    # Resample by year 
    resampled = lmeobs.resample(time='AS').mean('time')
    # Creating weights
    #For a rectangular grid the cosine of the latitude is proportional to the grid cell area.
    weights = np.cos(np.deg2rad(resampled.lat))
    weights.name = "weights"
    # Return the weighted average
    lme_weighted = resampled.weighted(weights)
    return lme_weighted.mean('dim_0').isel(time=slice(1, None))



def label_generator(path, mask):
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    maskds = xr.open_dataset("/media/cmlws/Data1/jsp/DLdata/LME66.mask.nc").load()
    locs = lme_mask(maskds, mask)
    lme = weighted_yearly_mean(data, v[0], locs)
    return lme


def prepare_ld_sequence(paths, ld, mask):
    """
    returns lead months single feature(chl/sst/ssh) xarray and nino indexes from files
    Parameters
    ------------------------
    path = files path
    ld = lead month
    """
    temp_frames = [] #lead months xarray list 
    temp_labels = [] #lead months nino index xarray list 
    # For each video.
    for path in (paths):
        # Gather all its frames and add a batch dimension.
        frames = prepare_single_frame(path[0],ld)
        label = label_generator(path[1], mask)
        temp_frames.append(frames)
        temp_labels.append(label)
    # frame_features = [el.interpolate_na(dim="lon") for el in temp_frames]    
    frame_features = [el.fillna(0) for el in temp_frames]
    sequence = np.concatenate(frame_features, axis=0)
    labels = np.concatenate(temp_labels, axis=0)
    labels = labels[:,np.newaxis]
    return sequence , labels


ldmn = 0


for lmen in np.arange(1,67): 
    outdir = f"/media/cmlws/Data2/jsp/LMEdata2/{ldmn}/{lmen}/RGB/"
    os.makedirs(outdir, exist_ok=True)       
    print(f"lead month {ldmn} LME {lmen} input file processing")

    chl_train_paths = list(zip(chl_train_files, target_train_files))
    chl_train_x, train_y = prepare_ld_sequence(chl_train_paths, ldmn, lmen)
    np.save(f"{outdir}/chl_tr_x_{lmen}.npy", chl_train_x)
    np.save(f"{outdir}/tr_y_{lmen}.npy", train_y)

    chl_valid_paths = list(zip(chl_valid_files, target_valid_files))
    chl_valid_x, valid_y = prepare_ld_sequence(chl_valid_paths, ldmn, lmen)
    np.save(f"{outdir}/chl_val_x_{lmen}.npy", chl_valid_x)
    np.save(f"{outdir}/val_y_{lmen}.npy", valid_y)

    chl_test_paths = list(zip(chl_test_files, target_test_files))
    chl_test_x, _ = prepare_ld_sequence(chl_test_paths, ldmn, lmen)
    np.save(f"{outdir}/chl_test_x_{lmen}.npy", chl_test_x)


    sst_train_paths = list(zip(sst_train_files, target_train_files))
    sst_train_x, _, = prepare_ld_sequence(sst_train_paths, ldmn, lmen)
    np.save(f"{outdir}/sst_tr_x_{lmen}.npy", sst_train_x)


    sst_valid_paths = list(zip(sst_valid_files, target_valid_files))
    sst_valid_x, _, = prepare_ld_sequence(sst_valid_paths, ldmn, lmen)
    np.save(f"{outdir}/sst_val_x_{lmen}.npy", sst_valid_x)

    sst_test_paths = list(zip(sst_test_files, target_test_files))
    sst_test_x, _ = prepare_ld_sequence(sst_test_paths, ldmn, lmen)
    np.save(f"{outdir}/sst_test_x_{lmen}.npy", sst_test_x)

    #ssmean 
    # sst_train_x = sst_train_x[:,:,:,np.newaxis]
    # chl_train_x = chl_train_x[:,:,:,np.newaxis]
    # sst_valid_x = sst_valid_x[:,:,:,np.newaxis]
    # chl_valid_x = chl_valid_x[:,:,:,np.newaxis]
    # sst_test_x = sst_test_x[:,:,:,np.newaxis]
    # chl_test_x = chl_test_x[:,:,:,np.newaxis]

    train_x = np.append(chl_train_x, sst_train_x, axis=3)
    valid_x = np.append(chl_valid_x, sst_valid_x, axis=3)
    test_x = np.append(chl_test_x, sst_test_x, axis=3)
    np.save(f"{outdir}/tr_x_{lmen}.npy", train_x)
    np.save(f"{outdir}/val_x_{lmen}.npy", valid_x)
    np.save(f"{outdir}/test_x_{lmen}.npy", test_x)
    print(f"lead month {ldmn} LME {lmen} input file processed")