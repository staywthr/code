
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
import glob
import os
import shutil


def file_moover(filelist, outdir):
    for f in filelist:
        filen = os.path.basename(f)
        # source = f
        # file = np.load(f)[:-1]
        file = np.load(f)
        np.save(f'{outdir}{filen}', file)
        # destination = f'{outdir}{file}'
        # shutil.copyfile(source, destination)
        # print(f'{destination} copied')
        print(f'{filen} copied')

INIM = [1, 12]
LME = [3]
for lmen in np.arange(1,67):
    print(f"LME {lmen} file split")

    for initm in INIM: #or np.arange(2,12)
        print(f"target month {initm} file split")

        xdir = f"/media/cmlws/Data2/jsp/LMEinput/initialization_month/{initm}/"

        
        chl_valid_files = sorted(glob.glob(f"{xdir}chl_val_x.npy"))
        sst_valid_files = sorted(glob.glob(f"{xdir}sst_val_x.npy"))
        valid_files = sorted(glob.glob(f"{xdir}val_x.npy"))
        chl_test_files = sorted(glob.glob(f"{xdir}chl_test_x.npy"))
        sst_test_files = sorted(glob.glob(f"{xdir}sst_test_x.npy"))
        test_files = sorted(glob.glob(f"{xdir}test_x.npy"))


        ydir = f"/media/cmlws/Data2/jsp/LMEinput/anom_yearly_mean/DJF/{lmen}"
        target_valid_files = sorted(glob.glob(f"{ydir}/val_y.npy"))
        target_test_files = sorted(glob.glob(f"{ydir}/test_y.npy"))


        # chl_train_files = sorted(glob.glob(f'{xdir}sfc_chl_Omon_*historical_*_init{initm}.nc'))
        # target_train_files = sorted(glob.glob(f'{ydir}sfc_chl_Omon_*historical_*_target{lmen}.nc'))
        # chl_train_paths = list(zip(chl_train_files, target_train_files))
        # print(f"Total cmip6 models(chl) for training: {len(chl_train_files)}")

        # temp_x_train = [] 
        # temp_x_valid = [] 
        # temp_y_train = []
        # temp_y_valid = []
        # for f in (chl_train_paths):
        #     filename = f[0].partition('Omon_')[-1]
        #     data = xr.open_dataarray(f[0]).load()
        #     target = xr.open_dataarray(f[1]).load()
        #     data = data.fillna(0)
        #     # data = data.isel(time=slice(0, -1)) # initialization month 2~11
        #     x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=0.2, shuffle=False, random_state=1004)
        #     print(f'{filename} chl X_train shape:', x_train.shape)
        #     print(f'{filename} chl X_valid shape:', x_valid.shape)
        #     print('y_train shape:', y_train.shape)
        #     print('y_valid shape:', y_valid.shape)

        #     temp_x_train.append(x_train)
        #     temp_x_valid.append(x_valid)
        #     temp_y_train.append(y_train)
        #     temp_y_valid.append(y_valid)        

        # chl_train_sequence = np.concatenate(temp_x_train, axis=0)
        # chl_valid_sequence = np.concatenate(temp_x_valid, axis=0)
        # train_labels = np.concatenate(temp_y_train, axis=0)
        # valid_labels = np.concatenate(temp_y_valid, axis=0)

        # sst_train_files = sorted(glob.glob(f'{xdir}tos_Omon_*historical_*_init{initm}.nc'))
        # sst_train_paths = list(zip(sst_train_files, target_train_files))
        
        # temp_x_train = [] 
        # temp_x_valid = [] 

        # for f in (sst_train_paths):
        #     filename = f[0].partition('Omon_')[-1]        
        #     data = xr.open_dataarray(f[0]).load()
        #     target = xr.open_dataarray(f[1]).load()
        #     data = data.fillna(0)   
        #     # data = data.isel(time=slice(0, -1)) # initialization month 2~11
        #     x_train, x_valid, _, _ = train_test_split(data, target, test_size=0.2, shuffle=False, random_state=1004)
        #     print(f'{filename} sst X_train shape:', x_train.shape)
        #     print(f'{filename} sst X_valid shape:', x_valid.shape)
        #     temp_x_train.append(x_train)
        #     temp_x_valid.append(x_valid)


        # sst_train_sequence = np.concatenate(temp_x_train, axis=0)
        # sst_valid_sequence = np.concatenate(temp_x_valid, axis=0)


        xoutdir =  f"/media/cmlws/Data2/jsp/LMEinput/split_init_month_yr/{initm}/"
        os.makedirs(xoutdir, exist_ok=True)    
        youtdir =  f"/media/cmlws/Data2/jsp/LMEinput/split_anom_yr_mean/{lmen}/"
        os.makedirs(youtdir, exist_ok=True)  
        # np.save(f'{xoutdir}chl_historical_tr_x.npy', chl_train_sequence)
        # np.save(f'{xoutdir}/chl_historical_val_x.npy', chl_valid_sequence)

        # np.save(f'{youtdir}historical_tr_y.npy', train_labels)
        # np.save(f'{youtdir}historical_val_y.npy', valid_labels)

        # np.save(f'{xoutdir}sst_historical_tr_x.npy', sst_train_sequence)
        # np.save(f'{xoutdir}sst_historical_val_x.npy', sst_valid_sequence)


        # train_x = np.append(chl_train_sequence, sst_train_sequence, axis=3)
        # valid_x = np.append(chl_valid_sequence, sst_valid_sequence, axis=3)

        # np.save(f"{xoutdir}/historical_tr_x.npy", train_x)
        # np.save(f"{xoutdir}/historical_val_x.npy", valid_x)
        
        print(f"target month {initm} LME{lmen} historical split 0.8:0.2 complete")

        # file_moover(chl_valid_files, xoutdir)
        # file_moover(sst_valid_files, xoutdir)
        # file_moover(valid_files, xoutdir)
        # file_moover(chl_test_files, xoutdir)
        # file_moover(sst_test_files, xoutdir)
        # file_moover(test_files, xoutdir)
        file_moover(target_valid_files,youtdir)
        file_moover(target_test_files, youtdir)

        print(f"\ target month {initm} LME{lmen} Reanalysis copy complete")




