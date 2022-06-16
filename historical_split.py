
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os
import shutil

ldmn = 0
def file_moover(filelist):
    for path in filelist:
        file = os.path.basename(path)
        source = f'{indir}{file}'
        destination = f'{outdir}{file}'
        shutil.copyfile(source, destination)
        print(f'{destination} copied')


for lmen in np.arange(1,67): 
    indir = f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/DJF/"
    chl_train_files = sorted(glob.glob(indir + 'sfc_chl_Omon_*historical_*train_x.npy'))
    sst_train_files = sorted(glob.glob(indir + 'tos_Omon_*historical_*train_x.npy'))
    target_train_files = sorted(glob.glob(indir + '*historical_*train_y.npy'))

    print(f"Total cmip6 models(chl) for training: {len(chl_train_files)}")
    print(f"Total cmip6 models(sst) for training: {len(sst_train_files)}")

    # shuffle = False
    chl_train_paths = list(zip(chl_train_files, target_train_files))

    temp_x_train = [] 
    temp_x_valid = [] 
    temp_y_train = []
    temp_y_valid = []

    for path in (chl_train_paths):
        
        filename = path[0].partition('Omon_')[-1]
        X = np.load(path[0])
        Y = np.load(path[1])    
        x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=1004)
        print(f'{filename} chl X_train shape:', x_train.shape)
        print(f'{filename} chl X_valid shape:', x_valid.shape)
        print('y_train shape:', y_train.shape)
        print('y_valid shape:', y_valid.shape)

        temp_x_train.append(x_train)
        temp_x_valid.append(x_valid)

        temp_y_train.append(y_train)
        temp_y_valid.append(y_valid)

    chl_train_sequence = np.concatenate(temp_x_train, axis=0)
    chl_valid_sequence = np.concatenate(temp_x_valid, axis=0)

    train_labels = np.concatenate(temp_y_train, axis=0)
    valid_labels = np.concatenate(temp_y_valid, axis=0)

    sst_train_paths = list(zip(sst_train_files, target_train_files))
    temp_x_train = [] 
    temp_x_valid = [] 
    for path in (sst_train_paths):
        filename = path[0].partition('Omon_')[-1]        
        X = np.load(path[0])
        Y = np.load(path[1])    
        x_train, x_valid, _, _ = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=1004)
        print(f'{filename} sst X_train shape:', x_train.shape)
        print(f'{filename} sst X_valid shape:', x_valid.shape)
        temp_x_train.append(x_train)
        temp_x_valid.append(x_valid)


    sst_train_sequence = np.concatenate(temp_x_train, axis=0)
    sst_valid_sequence = np.concatenate(temp_x_valid, axis=0)

    outdir =  f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/historical"
    os.makedirs(outdir, exist_ok=True)    

    np.save(f'{outdir}/chl_historical_tr_x.npy', chl_train_sequence)
    np.save(f'{outdir}/chl_historical_val_x.npy', chl_valid_sequence)

    np.save(f'{outdir}/historical_tr_y.npy', train_labels)
    np.save(f'{outdir}/historical_val_y.npy', valid_labels)

    np.save(f'{outdir}/sst_historical_tr_x.npy', sst_train_sequence)
    np.save(f'{outdir}/sst_historical_val_x.npy', sst_valid_sequence)


    train_x = np.append(chl_train_sequence, sst_train_sequence, axis=3)
    valid_x = np.append(chl_valid_sequence, sst_valid_sequence, axis=3)

    np.save(f"{outdir}/historical_tr_x.npy", train_x)
    np.save(f"{outdir}/historical_val_x.npy", valid_x)


    # chl_valid_files = sorted(glob.glob(indir+"chl_valid_x.npy"))
    # sst_valid_files = sorted(glob.glob(indir+"sst_valid_x.npy"))
    # valid_files = sorted(glob.glob(indir+'valid_x.npy'))
    # target_valid_files = sorted(glob.glob(indir+"valid_y.npy"))

    # chl_test_files = sorted(glob.glob(indir+"chl_test_x.npy"))
    # sst_test_files = sorted(glob.glob(indir+"sst_test_x.npy"))
    # test_files = sorted(glob.glob(indir+'test_x.npy'))
    # target_test_files = sorted(glob.glob(indir+"test_y.npy"))



    # file_moover(chl_valid_files)
    # file_moover(sst_valid_files)
    # file_moover(target_valid_files)
    # file_moover(valid_files)
    # file_moover(chl_test_files)
    # file_moover(sst_test_files)
    # file_moover(target_test_files)
    # file_moover(test_files)







    print(f"lead month {ldmn} LME {lmen} input file processed")

