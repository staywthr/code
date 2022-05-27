from matplotlib import colors
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import random
import pandas as pd
import os
import glob
# 재생산성을 위해 시드 고정
np.random.seed(7)
random.seed(7)


test_path = "/media/cmlws/Data1/jsp/DLdata/remap/test/"
target_test_files = sorted(glob.glob(test_path+"sst_JAN1998*_target.nc"))

def chk_var(klist):
    var_name_list = ['eta_t','sshg','ssh','chl','chlos','tos','sst']
    var = list(set(klist).intersection(var_name_list))
    return var

def label_generator(path, ld):
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
    elif ld>13:
        label =  np.append(t_arr.values[1:], [0+epsilon]) #164
    else: # 0<ld<12
        label =  np.append(t_arr.values, [0+epsilon]) #165
    return label

obs = []
for ld in np.arange(1,24):
    temp_labels = []
    for path in (target_test_files):
        label = label_generator(path, ld)
        temp_labels.append(label)
    ld_labels = np.concatenate(temp_labels, axis=0)
    ld_labels = ld_labels[:,np.newaxis]
    obs.append(ld_labels)

# creating directory 
os.makedirs('../output/fig/', exist_ok=True)

def get_cor_list(en, directory):
    cor_list_ld = []
    for ld in np.arange(1,24):
        ob=obs[ld-1][:,0]
        fcst = np.load('../output/'+ directory+'/models/'+str(ld)+'/fcst'+str(ld)+'.npy')
        cor_list_md =[]
        for j in range(en):
            cnn = fcst[j,:,0]
            cor = np.round(np.corrcoef(ob, cnn)[0,1], 2)
            cor_list_md.append(cor)            
        cor_list_ld.append(cor_list_md)

    cor_2d = np.swapaxes(cor_list_ld,0,1)
    df = pd.DataFrame(cor_2d, index=np.arange(en), columns=np.arange(1,24))
    return df

df = get_cor_list(5, 'xrcs_t6vRe_norm')
means1 = df.mean(axis=0)
std1 = df.std(axis=0)
upper_band1 = means1+std1
lower_band1 = means1-std1

df2 = get_cor_list(5, 'xrchl_t6vRe_log1x1')
means2 =df2.mean(axis=0)
std2 = df2.std(axis=0)
upper_band2 = means2+std2
lower_band2 = means2-std2

# df3 = get_cor_list(5, 'xrsst_t6vRe')
# means3 = df3.mean(axis=0)
# std3 = df3.std(axis=0)
# upper_band3 = means3+std3
# lower_band3 = means3-std3

df4 = get_cor_list(5, 'xrchl_t6vRe_hp21')
means4= df4.mean(axis=0)
std4 = df4.std(axis=0)
upper_band4 = means4+std4
lower_band4 = means4-std4

df5 = get_cor_list(5, 'xrchl_t6vRe_log')
means5 = df5.mean(axis=0)
std5 = df5.std(axis=0)
upper_band5 = means5+std5
lower_band5 = means5-std5



df6 = get_cor_list(5, 'xrchl_t6vRe_hp')
means6 = df6.mean(axis=0)
std6 = df6.std(axis=0)
upper_band6 = means6+std6
lower_band6 = means6-std6


df7 = get_cor_list(5, 'xrchl_t5vRe_log')
means7 = df7.mean(axis=0)
std7 = df7.std(axis=0)
upper_band7 = means7+std7
lower_band7 = means7-std7


df8 = get_cor_list(5, 'xrchl_t6vRe_hp20')
means8 = df8.mean(axis=0)
std8 = df8.std(axis=0)
upper_band8 = means8+std8
lower_band8 = means8-std8


#==========================================
# plot
#==========================================
# set figure size
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
# set x (time)
x = np.arange(1,24)
ax.set(xlim=(1, 24), ylim=(0, 1))
plt.xticks(np.arange(0,24))
plt.yticks(np.arange(-0.3,1,0.1))




ax.fill_between(x, upper_band1, lower_band1, color='b', alpha = 0.3)
ax.plot(x, means1, marker='o', color='b', lw=2, label='cmip6 multi-model chl&sst')

ax.fill_between(x, upper_band2, lower_band2, color='orange', alpha = 0.3)
ax.plot(x, means2, marker='o', color='orange', lw=2, label='cmip6 multi-model chl 1x1')

# ax.fill_between(x, upper_band3, lower_band3, color='r', alpha = 0.3)
# ax.plot(x, means3, marker='o', color='r', lw=2, label='cmip6 multi-model sst')

ax.fill_between(x, upper_band4, lower_band4, color='fuchsia', alpha = 0.3)
ax.plot(x, means4, marker='^', color='fuchsia', lw=2, label='cmip6 multi-model chl (tuned for lead month21)')

ax.fill_between(x, upper_band5, lower_band5, color='m', alpha = 0.3)
ax.plot(x, means5, marker='o', color='m', lw=2, label='cmip6 multi-model chl 5x5')

ax.fill_between(x, upper_band6, lower_band6, color='g', alpha = 0.3)
ax.plot(x, means6, marker='^', color='g',lw=2, label='cmip6 multi-model chl(tuned for lead month10)')

# ax.fill_between(x, upper_band7, lower_band7, color='k', alpha = 0.3)
ax.plot(x, means7, marker='o', color='grey',lw=2, label='cmip5 single model chl')

ax.fill_between(x, upper_band8, lower_band8, color='orangered', alpha = 0.3)
ax.plot(x, means8, marker='o', color='orangered', lw=2, label='cmip6 multi-model chl (tuned for lead month20)')

# ax.fill_between(x, upper_band8, lower_band8, color='salmon', alpha = 0.3)
# ax.plot(x, means8, marker='^', color='fuchsia', ls='--', lw=2, label='cmip6 multi-model chl 10x10')

ax.legend(loc = 'best', fontsize='small')

# x-, y-label
ax.set_xlabel('lead month', fontsize=8)
ax.set_ylabel('Correlation skill', fontsize=8)
ax.set_title('Forecast for Nino3.4(DJF)', y=0.99, fontsize=9)

plt.savefig('../output/fig/xrcmip6ens_chl_hp.png', dpi=300)
plt.close()

# cor_list_ld = []
# for i in range(11,23):
#     fcst = np.load('/home/cml/prjt/output/sst_trCmip6_valRe/models/'+str(i)+'/fcst'+str(i)+'.npy')
#     cor_list_md =[]
#     for j in range(5):
#         cnn = fcst[j,:,0]
#         cor = np.round(np.corrcoef(obs2, cnn)[0,1], 2)
#     cor_list_ld.append(cor_list_md)    
# cor_2d = np.swapaxes(cor_list_ld,0,1)

# df = pd.DataFrame(cor_2d, index=np.arange(5))
