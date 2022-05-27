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
        indir = f"/media/cmlws/Data2/jsp/LMEdata2/{ldmn}/{lmen}/RGB/"
        print(f"lead month {ldmn} LME {lmen} input file padding processing")

        chl_train_x = np.load(f"{indir}/chl_tr_x_{lmen}.npy")
        paded_chl_train_x  = paded_npy(chl_train_x)
        
        chl_valid_x = np.load(f"{indir}/chl_val_x_{lmen}.npy")
        paded_chl_valid_x  = paded_npy(chl_valid_x)
        
        chl_test_x = np.load(f"{indir}/chl_test_x_{lmen}.npy")
        paded_chl_test_x = paded_npy(chl_test_x)

        sst_train_x = np.load(f"{indir}/sst_tr_x_{lmen}.npy")
        paded_sst_train_x  = paded_npy(sst_train_x)
        
        sst_valid_x = np.load(f"{indir}/sst_val_x_{lmen}.npy")
        paded_sst_valid_x  = paded_npy(sst_valid_x)
        
        sst_test_x = np.load(f"{indir}/sst_test_x_{lmen}.npy")
        paded_sst_test_x = paded_npy(sst_test_x)
        
        train_x = np.load(f"{indir}/tr_x_{lmen}.npy")
        paded_train_x  = paded_npy(train_x)

        valid_x = np.load(f"{indir}/val_x_{lmen}.npy")
        paded_valid_x = paded_npy(valid_x)

        test_x = np.load(f"{indir}/test_x_{lmen}.npy")
        paded_test_x = paded_npy(test_x)

        train_y = np.load(f"{indir}/tr_y_{lmen}.npy")
        valid_y = np.load(f"{indir}/val_y_{lmen}.npy")

        outdir = f"/media/cmlws/Data2/jsp/padLMEdata/{ldmn}/{lmen}/RGB/"
        os.makedirs(outdir, exist_ok=True)       

        np.save(f"{outdir}/chl_tr_x_{lmen}.npy", paded_chl_train_x)
        np.save(f"{outdir}/tr_y_{lmen}.npy", train_y)
        np.save(f"{outdir}/chl_val_x_{lmen}.npy", paded_chl_valid_x)
        np.save(f"{outdir}/val_y_{lmen}.npy", valid_y)
        np.save(f"{outdir}/chl_test_x_{lmen}.npy", paded_chl_test_x)
        np.save(f"{outdir}/sst_tr_x_{lmen}.npy", paded_sst_train_x)
        np.save(f"{outdir}/sst_val_x_{lmen}.npy", paded_sst_valid_x)
        np.save(f"{outdir}/sst_test_x_{lmen}.npy", paded_sst_test_x)
        np.save(f"{outdir}/tr_x_{lmen}.npy", paded_train_x)
        np.save(f"{outdir}/val_x_{lmen}.npy", paded_valid_x)
        np.save(f"{outdir}/test_x_{lmen}.npy", paded_test_x)
        print(f"lead month {ldmn} LME {lmen} input file padding processed")