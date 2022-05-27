import os
import shutil
import numpy as np

ldmn = 0
lmen = 33
for lmen in np.arange(1,67): 
    # indir =  f"/media/cml/Data1/jsp/cmip6LMEdata/{ldmn}/{lmen}/DJF/"
    indir = f"/media/cml/Data1/jsp/cmip6LMEdata/{ldmn}/{lmen}/historical"
    chl_test_x = np.load(f"{indir}/chl_test_x.npy")
    sst_test_x = np.load(f"{indir}/sst_test_x.npy")
    test_x = np.append(chl_test_x, sst_test_x, axis=3)
    
    chl_valid_x = np.load(f"{indir}/chl_valid_x.npy")
    sst_valid_x = np.load(f"{indir}/sst_valid_x.npy")
    valid_x = np.append(chl_valid_x, sst_valid_x, axis=3)
    
    np.save(f'{indir}/test_x.npy', test_x)
    print(f'LME {lmen} test_x has been saved!')
    np.save(f'{indir}/valid_x.npy', valid_x)
    print(f'LME {lmen} valid_x has been saved!')

    # files = os.listdir(indir)
    # for file in files:
    #     if 'test_x' in file:
    #         shutil.copy(indir+file, outdir+file)
    #         print(f'{file} has been copied in new folder!')
    #     if 'valid_x' in file:
    #         shutil.copy(indir+file, outdir+file)
    #         print(f'{file} has been copied in new folder!')
    #     if 'test_y' in file:
    #         shutil.copy(indir+file, outdir+file)
    #         print(f'{file} has been copied in new folder!')
    #     if 'valid_y' in file:
    #         shutil.copy(indir+file, outdir+file)
    #         print(f'{file} has been copied in new folder!')
    # os.rename(f"{indir}/historical_tr_x_{lmen}.npy", f"{indir}/historical_tr_x.npy")
    # os.rename(f"{indir}/historical_val_x_{lmen}.npy", f"{indir}/historical_val_x.npy")