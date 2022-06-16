import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os
import shutil

ldmn = 0
def file_moover(filelist, indir, outdir):
    for path in filelist:
        file = os.path.basename(path)
        source = f'{indir}{file}'
        destination = f'{outdir}{file}'
        shutil.copyfile(source, destination)
        print(f'{destination} copied')


for lmen in np.arange(1,67): 
    # indir = f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/DJF/"
    # chl_pi_files = sorted(glob.glob(indir + 'sfc_chl_Omon_*piControl_*train_x.npy'))
    # sst_pi_files = sorted(glob.glob(indir + 'tos_Omon_*piControl_*train_x.npy'))
    # target_pi_files = sorted(glob.glob(indir + '*piControl_*train_y.npy'))


    # indir = f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/historical/"
    # chl_train_files = sorted(glob.glob(indir + 'chl_historical_tr_x.npy'))
    # sst_train_files = sorted(glob.glob(indir + 'sst_historical_tr_x.npy'))
    # target_train_files = sorted(glob.glob(indir + 'historical_tr_y.npy'))



    # # shuffle = False
    # chl_x_paths = list(zip(chl_pi_files, chl_train_files))
    # chl_y_paths = list(zip(target_pi_files , target_train_files))



    # for path in (chl_x_paths):
        
    #     filename = path[0].partition('Omon_')[-1]
    #     pi_x_train = np.load(path[0])
    #     historical_x_train = np.load(path[1])    

    #     print(f'{filename} chl pi X_train shape:', pi_x_train.shape)
    #     print(f'{filename} chl historical X_train shape:', historical_x_train.shape)
        
    #     chl_x_train = np.append(pi_x_train, historical_x_train, axis=0)


    # for path in (chl_y_paths):
        
    #     filename = path[0].partition('Omon_')[-1]
    #     pi_y_train = np.load(path[0])
    #     historical_y_train = np.load(path[1])    

    #     print(f'{filename} chl pi Y_train shape:', pi_y_train.shape)
    #     print(f'{filename} chl historical Y_train shape:', historical_y_train.shape)

    #     y_train = np.append(pi_y_train, historical_y_train, axis=0)

    # print("chlorophyll single model piCon and his training processed")

    # sst_x_paths = list(zip(sst_pi_files, sst_train_files))

    # for path in (sst_x_paths):
    #     filename = path[0].partition('Omon_')[-1]        
    #     pi_x_train = np.load(path[0])
    #     historical_x_train = np.load(path[1])    
    
    #     print(f'{filename} sst pi X_train shape:', pi_x_train.shape)
    #     print(f'{filename} sst historical X_train shape:', historical_x_train.shape)
        
    #     sst_x_train = np.append(pi_x_train, historical_x_train, axis=0)      

    # print("SST single model piCon and his training processed")

    outdir =  f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/piConHis/"
    os.makedirs(outdir, exist_ok=True)    
    print(f"open {outdir}")

    # np.save(f'{outdir}/chl_tr_x.npy', chl_x_train) #pi+his
    # np.save(f'{outdir}/tr_y.npy', y_train) #pi+his
    # np.save(f'{outdir}/sst_tr_x.npy', sst_x_train) #pi+his

    # train_x = np.append(chl_x_train, sst_x_train, axis=3)
    # np.save(f"{outdir}/tr_x.npy", train_x) #pi+his and chl+sst

    # print("file saved")

    # indir = f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/DJF/"
    
    # chl_valid_files = sorted(glob.glob(indir+"chl_valid_x.npy")) #reanalysis
    # sst_valid_files = sorted(glob.glob(indir+"sst_valid_x.npy")) #reanalysis

    # val_x_dir = f"/media/cml/Data1/jsp/cmip6LMEdata/{ldmn}/{lmen}/historical/"
    # valid_files = sorted(glob.glob(val_x_dir+'valid_x.npy')) #reanalysis
    
    # target_valid_files = sorted(glob.glob(indir+"valid_y.npy")) #reanalysis

    # chl_test_files = sorted(glob.glob(indir+"chl_test_x.npy"))
    # sst_test_files = sorted(glob.glob(indir+"sst_test_x.npy"))

    # test_x_dir = f"/media/cml/Data1/jsp/cmip6LMEdata/{ldmn}/{lmen}/historical/"
    # test_files = sorted(glob.glob(test_x_dir+'test_x.npy'))
    
    # target_test_files = sorted(glob.glob(indir+"test_y.npy"))



    # file_moover(chl_valid_files, indir, outdir)
    # file_moover(sst_valid_files, indir, outdir)
    # file_moover(target_valid_files, indir, outdir)
    # file_moover(valid_files, val_x_dir, outdir)

    # file_moover(chl_test_files, indir, outdir)
    # file_moover(sst_test_files, indir, outdir)
    # file_moover(target_test_files, indir, outdir)
    # file_moover(test_files, test_x_dir, outdir)

    # print(f"{indir} valid and test file moved")

    indir = f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/historical/"
    # chl_valid_files = sorted(glob.glob(indir+"chl_historical_val_x.npy"))
    # sst_valid_files = sorted(glob.glob(indir+"sst_historical_val_x.npy"))
    valid_files = sorted(glob.glob(indir+'historical_val_x.npy'))
    # target_valid_files = sorted(glob.glob(indir+"historical_val_y.npy"))


    # file_moover(chl_valid_files, indir, outdir)
    # file_moover(sst_valid_files, indir, outdir)
    # file_moover(target_valid_files, indir, outdir)
    file_moover(valid_files, indir, outdir)
    print(f"{indir} historical valid file moved")

    # indir = f"/media/cml/Data1/jsp/GFDL-CM4/{ldmn}/{lmen}/historical/"
    # chl_train_files = sorted(glob.glob(indir + 'chl_historical_tr_x.npy'))
    # sst_train_files = sorted(glob.glob(indir + 'sst_historical_tr_x.npy'))
    # train_files = sorted(glob.glob(indir+'historical_tr_x.npy'))


    # file_moover(chl_train_files, indir, outdir)
    # file_moover(sst_train_files, indir, outdir)
    # file_moover(train_files, indir, outdir)
    # print(f"{indir} historical train file moved")

    print(f"lead month {ldmn} LME {lmen} input file processed")

