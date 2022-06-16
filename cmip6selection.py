import numpy as np
import glob
import os
import shutil


ldmn = 0
model_name_list = ['GFDL-ESM4']


def select_cmip6File_cp(filelist):
    selection = []
    for path in filelist:

        x_basename = os.path.basename(path)
        x_file_name = os.path.splitext(x_basename)[0]
        x_file_name = x_file_name.partition('Omon_')[-1]  
        for m in model_name_list:
            if m in x_file_name:
                selection.append(x_basename)
                print(f'{x_file_name} has been selected')

    for file in selection: 
        source = f'{indir}{file}'
        destination = f'{outdir}{file}'
        shutil.copyfile(source, destination)



def select_cmip6File_cp_y(filelist):
    selection = []
    for path in filelist:

        x_basename = os.path.basename(path)
        x_file_name = os.path.splitext(x_basename)[0]
        # x_file_name = x_file_name.partition('Omon_')[-1]  #if train_y: doesn't need 
        for m in model_name_list:
            if m in x_file_name:
                selection.append(x_basename)
                print(f'{x_file_name} has been selected')

    for file in selection: 
        source = f'{indir}{file}'
        destination = f'{outdir}{file}'
        shutil.copyfile(source, destination)


def file_moover(filelist):
    for path in filelist:
        file = os.path.basename(path)
        source = f'{indir}{file}'
        destination = f'{outdir}{file}'
        shutil.copyfile(source, destination)
        print(f'{file} copied')


for lmen in range(1,67):
    indir = f"/media/cml/Data1/jsp/cmip6LMEdata/{ldmn}/{lmen}/DJF/"
    print(f'LME {lmen} cmip6 selection start')
    chl_train_files = sorted(glob.glob(indir + 'sfc_chl_Omon_*train_x.npy'))
    sst_train_files = sorted(glob.glob(indir + 'tos_Omon_*train_x.npy'))
    target_train_files = sorted(glob.glob(indir + '*train_y.npy'))
    outdir = f"/media/cml/Data1/jsp/GFDL-ESM4/{ldmn}/{lmen}/DJF/"
    os.makedirs(outdir, exist_ok=True)    

    select_cmip6File_cp(chl_train_files)
    select_cmip6File_cp(sst_train_files)
    select_cmip6File_cp_y(target_train_files)


    chl_valid_files = sorted(glob.glob(indir+"chl_valid_x.npy"))
    sst_valid_files = sorted(glob.glob(indir+"sst_valid_x.npy"))
    valid_files = sorted(glob.glob(indir+'valid_x.npy'))
    target_valid_files = sorted(glob.glob(indir+"valid_y.npy"))

    chl_test_files = sorted(glob.glob(indir+"chl_test_x.npy"))
    sst_test_files = sorted(glob.glob(indir+"sst_test_x.npy"))
    test_files = sorted(glob.glob(indir+'test_x.npy'))
    target_test_files = sorted(glob.glob(indir+"test_y.npy"))

    file_moover(valid_files)
    file_moover(test_files)
    file_moover(chl_valid_files)
    file_moover(sst_valid_files)
    file_moover(target_valid_files)

    file_moover(chl_test_files)
    file_moover(sst_test_files)
    file_moover(target_test_files)



# selection = []
# for path in chl_train_files:

#     x_basename = os.path.basename(path)
#     x_file_name = os.path.splitext(x_basename)[0]
#     x_file_name = x_file_name.partition('Omon_')[-1] 

#     for m in model_name_list:
#         if m in x_file_name:
#             selection.append(x_basename)
#             print(f'{x_file_name} has been selected')

# for file in selection: 
#     source = f'{indir}{file}'
#     destination = f'{outdir}{file}'
#     shutil.copyfile(source, destination)

