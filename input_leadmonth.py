import glob
import os
import pickle
import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
from contextlib import redirect_stdout



#missing grid 정보를 담고 있는 넘파이 바이너리 파일
LOCS = np.load('/media/cmlws/Data1/jsp/DLdata/remap/test/chl_latlon_mask_5x5.npy')

###train 
train_path = '/media/cmlws/Data1/jsp/DLdata/remap/valid/'
chl_train_files = sorted(glob.glob(train_path + 'remap5_sfc_chl_Omon_*.nc'))
sst_train_files = sorted(glob.glob(train_path + 'remap5_tos_Omon_*.nc'))
target_train_files = sorted(glob.glob(train_path + 'target/tos_Omon_*_target.nc'))

print(f"Total cmip6 models(chl) for training: {len(chl_train_files)}")
print(f"Total cmip6 models(sst) for training: {len(sst_train_files)}")
print(f"Total cmip6 models(target) for training: {len(target_train_files)}")

####valid
valid_path = "/media/cmlws/Data1/jsp/DLdata/remap/transfer/"
chl_valid_files = sorted(glob.glob(valid_path+"sfc_chl_*196501*.nc"))
sst_valid_files = sorted(glob.glob(valid_path+"sst_*196501*.nc"))
ssh_valid_files =  sorted(glob.glob(valid_path+"eta_t_*196501*.nc"))
target_valid_files = sorted(glob.glob(valid_path+"sst_*196501*_target.nc"))

####test
test_path = "/media/cmlws/Data1/jsp/DLdata/remap/test/"
chl_test_files = sorted(glob.glob(test_path+"chl_199801-201812_5x5.nc"))
sst_test_files = sorted(glob.glob(test_path+"sst_JAN1998*.nc"))
ssh_test_files =  sorted(glob.glob(test_path+"ssh_JAN1998*.nc"))
target_test_files = sorted(glob.glob(test_path+"sst_JAN1998*_target.nc"))

#nc파일이 여러 변수를 담고 있을 수도 있음. 확인하고 싶은 변수들과 nc파일의 변수를 확인해서 그 변수 이름만 가져온다.
def chk_var(klist): 
    var_name_list = ['eta_t','sshg','ssh','SSH','chl','sfc_chl','chlos','tos','sst','SST']
    var = list(set(klist).intersection(var_name_list))
    return var

# nc파일을 읽고 nc 파일이 담고 있는 변수들을 확인하고, dimension을 확인한뒤 (z dimension을 필요 없었음) time, lon, lat 모양의 var arr를 리턴한다.
def load_xdata(path):
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    #select var array
    arr = data[v[0]].sel()
    #check dimension
    dim = list(arr.dims)
    #targeted dimension shape
    shape =  ['time','lon','lat']
    #if chl nc file has 4 dimension, squeeze to 3
    if v == ['chl'] and len(arr.dims) == 4:
        arr = arr.squeeze(list(set(dim).difference(set(shape))))
    return arr.transpose('time','lon','lat')



# missing value grid to nan 
def cleanining_grid(da, locnpy):
    for i in range(len(locnpy)):
        da[:,locnpy[i][0],locnpy[i][1]] = np.nan 
    print(f'# Total number of missing grid: {len(locnpy)}')
    return da

# anomaly return
def calculate_anomaly(da, groupby_type="time.month"):
    gb = da.groupby(groupby_type)
    clim = gb.mean(dim="time")
    return gb - clim

# normalized anomaly return
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

# get 1 month anomaly
def month_anomaly(var_anom, mind):
    xda = var_anom.where(var_anom.coords['month']==mind,drop=True)
    return xda


def season_anomaly(var_anom,ld):
    """
    returns 3 month anomaly xarray.DataArray.
    Parameters
    ------------------------
    ld = lead month
    """
    #lead month 1, NDJ anomaly 11월 anomaly, 12월 anomaly, 1월 anomaly를 z dimension에 Concat한다. (RGB 채널처럼) 
    if ld == 1:
        anom0 = month_anomaly(var_anom, 11)
        anom1 = month_anomaly(var_anom, 12)
        anom2 = month_anomaly(var_anom, 1)
        ldmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)),anom2.isel(time=slice(1, None))], dim='z', join='override')
    #DJF anomaly
    elif ld == 12:
        anom0 = month_anomaly(var_anom, 12)
        anom1 = month_anomaly(var_anom, 1)
        anom2 = month_anomaly(var_anom, 2)
        ldmn = xr.concat([anom0.isel(time=slice(0, -2)), anom1.isel(time=slice(1, -1)),anom2.isel(time=slice(1, -1))], dim='z', join='override')
    #lead month 13, NDJ anomaly
    elif ld == 13:
        anom0 = month_anomaly(var_anom, 11)
        anom1 = month_anomaly(var_anom, 12)
        anom2 = month_anomaly(var_anom, 1)
        ldmn = xr.concat([anom0.isel(time=slice(0, -2)), anom1.isel(time=slice(0, -2)),anom2.isel(time=slice(1, -1))], dim='z', join='override')
    elif ld >13:
        anom0 = month_anomaly(var_anom, 25-ld-1)
        anom1 = month_anomaly(var_anom, 25-ld)
        anom2 = month_anomaly(var_anom, 25-ld+1)
        ldmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)), anom2.isel(time=slice(0, -1))], dim='z', join='override')
    else:
        anom0 = month_anomaly(var_anom, 13-ld-1)
        anom1 = month_anomaly(var_anom, 13-ld)
        anom2 = month_anomaly(var_anom, 13-ld+1)
        ldmn = xr.concat([anom0.isel(time=slice(0, None)), anom1.isel(time=slice(0, None)), anom2.isel(time=slice(0, None))], dim='z', join='override')
    return ldmn.transpose('time','lon','lat','z')

#정규분포 형태가 아닌 클로로필을 로그 트랜스폼한다. 
def log_transform(arr):
    epsilon=1e-06 
    log_norm = np.log(arr+epsilon)
    return log_norm

#xarray 
def log10_transform(arr):
    epsilon=1e-06 
    log_norm = np.log10(arr+epsilon)
    return log_norm

# CMIP 6 모델 하나의 seasonal anomaly를 리턴한다 
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
    
    anom = calculate_anomaly(arr)
    # anom = calculate_stand_anomaly(arr)
    ss_anom = season_anomaly(anom, ld)
   
    print(f"Lead month {ld} {ss_anom.name} {(path.partition('Omon_')[-1])} Dataset")
    return ss_anom



def label_generator(path, ld):
    """label을 구한다. 
    1998-2018년도 DJF가 타겟일 때, 
    leadmonth 1인 NDJ Season anomaly는 164개로 숫자가 맞음
    leadmonth 12인 DJF season anomaly는 163개 
    문제와 답의 수를 맞추어주어야 한다. 
    """
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    t_arr = data[v[0]].sel()
    epsilon=1e-06 
    if ld==1:
        label = t_arr.values #164
    elif ld==12:
        label = t_arr.values[1:] #163
    elif ld ==13:
        label = t_arr.values[1:] #163
    # 답이 0이면 문제가 생기므로 작은 epsilon값을 더해준다. 
    elif ld>13:
        label =  np.append(t_arr.values[1:], [0+epsilon]) #164
    else: # 0<ld<12
        label =  np.append(t_arr.values, [0+epsilon]) #165
    return label

# lead month에 따라 모든 CMIP 6 모델들의 season anomaly와 label을 리턴한다.
def prepare_ld_sequence(paths, ld):
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
        label = label_generator(path[1], ld)
        temp_frames.append(frames)
        temp_labels.append(label)
    # frame_features = [el.interpolate_na(dim="lon") for el in temp_frames]    
    frame_features = [el.fillna(0) for el in temp_frames]
    sequence = np.concatenate(frame_features, axis=0)
    labels = np.concatenate(temp_labels, axis=0)
    labels = labels[:,np.newaxis]
    return sequence , labels


# 전처리한 인풋 데이터를 넘파이 바이너리 파일로 저장한다
for ldmn in np.arange(1,24):
        outdir = f"/media/cmlws/Data2/jsp/LMEdata/{ldmn}/"
        os.makedirs(outdir, exist_ok=True)       
        print(f"lead month {ldmn} LME input file processing")

        chl_train_paths = list(zip(chl_train_files, target_train_files))
        chl_train_x, train_y = prepare_ld_sequence(chl_train_paths, ldmn)
        np.save(f"{outdir}/chl_tr_x.npy", chl_train_x)
        np.save(f"{outdir}/tr_y.npy", train_y)

        chl_valid_paths = list(zip(chl_valid_files, target_valid_files))
        chl_valid_x, valid_y = prepare_ld_sequence(chl_valid_paths, ldmn)
        np.save(f"{outdir}/chl_val_x.npy", chl_valid_x)
        np.save(f"{outdir}/val_y.npy", valid_y)

        chl_test_paths = list(zip(chl_test_files, target_test_files))
        chl_test_x, _ = prepare_ld_sequence(chl_test_paths, ldmn)
        np.save(f"{outdir}/chl_test_x.npy", chl_test_x)


        sst_train_paths = list(zip(sst_train_files, target_train_files))
        sst_train_x, _, = prepare_ld_sequence(sst_train_paths, ldmn)
        np.save(f"{outdir}/sst_tr_x.npy", sst_train_x)


        sst_valid_paths = list(zip(sst_valid_files, target_valid_files))
        sst_valid_x, _, = prepare_ld_sequence(sst_valid_paths, ldmn)
        np.save(f"{outdir}/sst_val_x.npy", sst_valid_x)

        sst_test_paths = list(zip(sst_test_files, target_test_files))
        sst_test_x, _ = prepare_ld_sequence(sst_test_paths, ldmn)
        np.save(f"{outdir}/sst_test_x.npy", sst_test_x)

        # sst_train_x = sst_train_x[:,:,:,np.newaxis]
        # chl_train_x = chl_train_x[:,:,:,np.newaxis]
        # sst_valid_x = sst_valid_x[:,:,:,np.newaxis]
        # chl_valid_x = chl_valid_x[:,:,:,np.newaxis]
        # sst_test_x = sst_test_x[:,:,:,np.newaxis]
        # chl_test_x = chl_test_x[:,:,:,np.newaxis]

        train_x = np.append(chl_train_x, sst_train_x, axis=3)
        valid_x = np.append(chl_valid_x, sst_valid_x, axis=3)
        test_x = np.append(chl_test_x, sst_test_x, axis=3)
        np.save(f"{outdir}/tr_x.npy", train_x)
        np.save(f"{outdir}/val_x.npy", valid_x)
        np.save(f"{outdir}/test_x.npy", test_x)
        print(f"lead month {ldmn} LME input file processed")