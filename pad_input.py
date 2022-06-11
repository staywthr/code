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
    

def paded_npy(npy):
    height, width = npy.shape[1:3]
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]
    # 부족한 길이가 절반으로 안 떨어질 경우 +1
    if np.abs(height-width) % 2 != 0:
        margin[0] += 1
    # 가로, 세로 가운데 부족한 쪽에 margin 추가
    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]
    # color 이미지일 경우 color 채널 margin 추가
    if len(npy.shape) >= 3:
        margin_list.append([0,0])
    # 이미지에 margin 추가
    output_list = []
    for image in npy: 
        output = np.pad(image, margin_list, mode='constant')
        output = output[np.newaxis,:,:,:]
        output_list.append(output)
    paded = np.concatenate(output_list, axis=0)
    return paded
    


for ldmn in np.arange(1):
    for lmen in np.arange(1,67):
        indir = f"/media/cmlws/Data2/jsp/cmip6LMEdata/{ldmn}/{lmen}/historical/"
        print(f"lead month {ldmn} LME {lmen} input file padding processing")

        chl_train_x = np.load(f"{indir}/chl_historical_tr_x.npy")
        paded_chl_train_x  = paded_npy(chl_train_x)
        print(f"chl train x padded!")
        sst_train_x = np.load(f"{indir}/sst_historical_tr_x.npy")
        paded_sst_train_x  = paded_npy(sst_train_x)
        print(f"sst train x padded!")
        train_x = np.load(f"{indir}/historical_tr_x.npy")
        paded_train_x  = paded_npy(train_x)
        print(f"train x padded!")
        train_y = np.load(f"{indir}/historical_tr_y.npy")

        chl_hisval_x = np.load(f"{indir}/chl_historical_val_x.npy")
        paded_chl_hisval_x  = paded_npy(chl_hisval_x)
        print(f"chl historical val x padded!")
        sst_hisval_x = np.load(f"{indir}/sst_historical_val_x.npy")
        paded_sst_hisval_x  = paded_npy(sst_hisval_x)
        print(f"sst historical val x padded!")
        hisval_x = np.load(f"{indir}/historical_val_x.npy")
        paded_hisval_x = paded_npy(hisval_x)
        print(f"historical val x padded!")
        hisval_y = np.load(f"{indir}/historical_val_y.npy")

        chl_valid_x = np.load(f"{indir}/chl_valid_x.npy")
        paded_chl_valid_x  = paded_npy(chl_valid_x)
        print(f"chl val x padded!")
        sst_valid_x = np.load(f"{indir}/sst_valid_x.npy")
        paded_sst_valid_x  = paded_npy(sst_valid_x)
        print(f"sst val x padded!")
        valid_x = np.load(f"{indir}/valid_x.npy")
        paded_valid_x = paded_npy(valid_x)
        print(f"val x padded!")
        valid_y = np.load(f"{indir}/valid_y.npy")

        chl_test_x = np.load(f"{indir}/chl_test_x.npy")
        paded_chl_test_x = paded_npy(chl_test_x)
        print(f"chl test x padded!")
        sst_test_x = np.load(f"{indir}/sst_test_x.npy")
        paded_sst_test_x = paded_npy(sst_test_x)
        print(f"sst test x padded!")
        test_x = np.load(f"{indir}/test_x.npy")
        paded_test_x = paded_npy(test_x)
        print(f"test x padded!")
        test_y = np.load(f"{indir}/test_y.npy")     
       
        outdir = f"/media/cmlws/Data2/jsp/padLMEdata/{ldmn}/{lmen}/historical/"
        os.makedirs(outdir, exist_ok=True)       
        print("out directory made!")
        np.save(f"{outdir}/chl_historical_tr_x.npy", paded_chl_train_x)
        np.save(f"{outdir}/sst_historical_tr_x.npy", paded_sst_train_x)
        np.save(f"{outdir}/historical_tr_x.npy", paded_train_x)
        np.save(f"{outdir}/historical_tr_y.npy", train_y)
        print("hitorical train x and train y saved!")

        np.save(f"{outdir}/chl_historical_val_x.npy", paded_chl_hisval_x)
        np.save(f"{outdir}/sst_historical_val_x.npy", paded_sst_hisval_x)
        np.save(f"{outdir}/historical_val_x.npy", paded_hisval_x)
        np.save(f"{outdir}/historical_val_y.npy", hisval_y)
        print("hitorical valid x and train y saved!")

        np.save(f"{outdir}/chl_valid_x.npy", paded_chl_valid_x)
        np.save(f"{outdir}/sst_valid_x.npy", paded_sst_valid_x)
        np.save(f"{outdir}/valid_x.npy", paded_valid_x)
        np.save(f"{outdir}/valid_y.npy", valid_y)
        print("valid x and train y saved!")

        np.save(f"{outdir}/chl_test_x.npy", paded_chl_test_x)
        np.save(f"{outdir}/sst_test_x.npy", paded_sst_test_x)
        np.save(f"{outdir}/test_x.npy", paded_test_x)
        np.save(f"{outdir}/test_y.npy", test_y)
        print("test x and train y saved!")
        print(f"lead month {ldmn} LME {lmen} input file padding processed")