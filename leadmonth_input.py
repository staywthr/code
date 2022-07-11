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


print(f"Total cmip6 models(chl) for training: {len(chl_train_files)}")
print(f"Total cmip6 models(sst) for training: {len(sst_train_files)}")

####valid
valid_path = "/media/cmlws/Data1/jsp/DLdata/1/transfer/"
chl_valid_files = sorted(glob.glob(valid_path+"sfc_chl_*196501*.nc"))
sst_valid_files = sorted(glob.glob(valid_path+"sst_*196501-199712_1x1.nc"))


test_path = "/media/cmlws/Data1/jsp/DLdata/1/test/"
chl_test_files = sorted(glob.glob(test_path+"chl_199801-201812_1x1.nc"))
sst_test_files = sorted(glob.glob(test_path+"sst_JAN1998*.nc"))


def chk_var(klist): 
    var_name_list = ['eta_t','sshg','ssh','SSH','chl','sfc_chl','chlos','tos','sst','SST']
    var = list(set(klist).intersection(var_name_list))
    return var

def load_xdata(path):
    data = xr.open_dataset(path).load()
    x_basename = os.path.basename(path)
    x_file_name = os.path.splitext(x_basename)[0] 
    print(f"{x_file_name} opened")   
    k = list(data.keys())
    v = chk_var(k)
    arr = data[v[0]].sel()
    dim = list(arr.dims)
    shape =  ['time','lon','lat']
    if v == ['chl'] and len(arr.dims) == 4:
        arr = arr.squeeze(list(set(dim).difference(set(shape))))
    else:
        length = arr.dims
        print(f"array has {length} dims")
    return arr.transpose('time','lon','lat')



def cleanining_grid(da, locnpy):
    for i in range(len(locnpy)):
        da[:,locnpy[i][0],locnpy[i][1]] = np.nan 
    print(f'# Total number of missing grid: {len(locnpy)}')
    return da

#xarray 
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


def month_anomaly(var_anom, mind):
    xda = var_anom.where(var_anom.coords['month']==mind,drop=True)
    return xda


def season_anomaly(var_anom, initm):
    """
    returns 3 month anomaly xarray.DataArray.
    Parameters
    ------------------------
    initm = lead month 
    initm == 1(DJF) or initm == 2(JFM)
    """
    if initm == 1:
        anom0 = month_anomaly(var_anom, 12)
        anom1 = month_anomaly(var_anom, 1)
        anom2 = month_anomaly(var_anom, 2)
        initmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(1, None)),anom2.isel(time=slice(1, None))], dim='z', join='override')
    elif initm == 2:
        anom0 = month_anomaly(var_anom, 1)
        anom1 = month_anomaly(var_anom, 2)
        anom2 = month_anomaly(var_anom, 3)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 3:
        anom0 = month_anomaly(var_anom, 2)
        anom1 = month_anomaly(var_anom, 3)
        anom2 = month_anomaly(var_anom, 4)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 4:
        anom0 = month_anomaly(var_anom, 3)
        anom1 = month_anomaly(var_anom, 4)
        anom2 = month_anomaly(var_anom, 5)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 5:
        anom0 = month_anomaly(var_anom, 4)
        anom1 = month_anomaly(var_anom, 5)
        anom2 = month_anomaly(var_anom, 6)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 6:
        anom0 = month_anomaly(var_anom, 5)
        anom1 = month_anomaly(var_anom, 6)
        anom2 = month_anomaly(var_anom, 7)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 7:
        anom0 = month_anomaly(var_anom, 6)
        anom1 = month_anomaly(var_anom, 7)
        anom2 = month_anomaly(var_anom, 8)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 8:
        anom0 = month_anomaly(var_anom, 7)
        anom1 = month_anomaly(var_anom, 8)
        anom2 = month_anomaly(var_anom, 9)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 9:
        anom0 = month_anomaly(var_anom, 8)
        anom1 = month_anomaly(var_anom, 9)
        anom2 = month_anomaly(var_anom, 10)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 10:
        anom0 = month_anomaly(var_anom, 9)
        anom1 = month_anomaly(var_anom, 10)
        anom2 = month_anomaly(var_anom, 11)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override')
    elif initm == 11:
        anom0 = month_anomaly(var_anom, 10)
        anom1 = month_anomaly(var_anom, 11)
        anom2 = month_anomaly(var_anom, 12)
        initmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)),anom2.isel(time=slice(0, None))], dim='z', join='override') 
    else:
        # initm == 12(NDJ)
        anom0 = month_anomaly(var_anom, 11)
        anom1 = month_anomaly(var_anom, 12)
        anom2 = month_anomaly(var_anom, 1)
        initmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)),anom2.isel(time=slice(1, None))], dim='z', join='override')

    return initmn.transpose('time','lon','lat','z')



def prepare_single_frame(path, initm): 
    """
    returns initializtion month single feature(chl/sst/ssh) xarray from one file
    Parameters
    ------------------------
    path = file path
    initm = init month
    """

    arr = load_xdata(path)
    if (arr.name == 'chl') or (arr.name == 'chlos') or (arr.name == 'sfc_chl'):
        print("chlorophyll preprocess")
        arr = cleanining_grid(arr, LOCS)
        arr = log_transform(arr)    
    else:
        print("sst process")       
    # anom = calculate_anomaly(arr)
    anom = calculate_stand_anomaly(arr)
    ss_anom = season_anomaly(anom, initm)
    # ss_anom = season_mean_anomaly(anom,ld)
    outdir = f"/media/cmlws/Data2/jsp/LMEinput/initialization_month/{initm}/"
    os.makedirs(outdir, exist_ok=True)
    x_basename = os.path.basename(path)
    x_file_name = os.path.splitext(x_basename)[0]
    ss_anom.to_netcdf(f"{outdir}{x_file_name}_init{initm}.nc")
    print(f"initial month {initm} {ss_anom.name} {(path.partition('Omon_')[-1])} Dataset")


    return ss_anom


def prepare_ld_sequence(path, initm):
    """
    returns lead months single feature(chl/sst/ssh) xarray and nino indexes from files
    Parameters
    ------------------------
    path = files path
    ld = lead month
    """
    temp_frames = [] #lead months xarray list 
    # For each video.
    for f in (path):
        # Gather all its frames and add a batch dimension.
        frames = prepare_single_frame(f,initm)
        temp_frames.append(frames)
    # frame_features = [el.interpolate_na(dim="lon") for el in temp_frames]    
    frame_features = [el.fillna(0) for el in temp_frames]
    sequence = np.concatenate(frame_features, axis=0)
    return sequence 


for initm in np.arange(1,13):
    outdir = f"/media/cmlws/Data2/jsp/LMEinput/initialization_month/{initm}/"
    os.makedirs(outdir, exist_ok=True)
    chl_train_x = prepare_ld_sequence(chl_train_files, initm)
    np.save(f"{outdir}/chl_tr_x.npy", chl_train_x)
    chl_valid_x = prepare_ld_sequence(chl_valid_files, initm)
    np.save(f"{outdir}/chl_val_x.npy", chl_valid_x)
    chl_test_x = prepare_ld_sequence(chl_test_files, initm)
    np.save(f"{outdir}/chl_test_x.npy", chl_test_x)

    sst_train_x = prepare_ld_sequence(sst_train_files, initm)
    np.save(f"{outdir}/sst_tr_x.npy", sst_train_x)
    sst_valid_x = prepare_ld_sequence(sst_valid_files, initm)
    np.save(f"{outdir}/sst_val_x.npy", sst_valid_x)
    sst_test_x = prepare_ld_sequence(sst_test_files, initm)
    np.save(f"{outdir}/sst_test_x.npy", sst_test_x)
    train_x = np.append(chl_train_x, sst_train_x, axis=3)
    valid_x = np.append(chl_valid_x, sst_valid_x, axis=3)
    test_x = np.append(chl_test_x, sst_test_x, axis=3)
    np.save(f"{outdir}tr_x.npy", train_x)
    np.save(f"{outdir}val_x.npy", valid_x)
    np.save(f"{outdir}test_x.npy", test_x)
