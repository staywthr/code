import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr
import numpy as np
from scipy import stats
import pandas as pd
import os
import glob

test_path = "/media/cmlws/Data1/jsp/DLdata/1/test/"
target_test_files = sorted(glob.glob(test_path+"chl_199801-201812_1x1.nc"))

def chk_var(klist): 
    var_name_list = ['eta_t','sshg','ssh','SSH','chl','sfc_chl','chlos','tos','sst','SST']
    var = list(set(klist).intersection(var_name_list))
    return var

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


#xarray 
def log_transform(arr):
    epsilon=1e-06 
    log_norm = np.log(arr+epsilon)
    return log_norm
 


def lme_mask(ds, masknumber):
    # Select lme region by number
    lme_da = ds.mask.where(ds.mask == masknumber)
    # Creating lat lon index 
    lon_ind = np.arange(360)
    lat_ind = np.arange(180)
    # Substitute coordinates with indices for using vectorized access
    lme_da['lon'] = lon_ind
    lme_da['lat'] = lat_ind
    # Get a set of indices by StackExchange
    da_stacked = lme_da.stack(yx=['lat','lon'])
    index = da_stacked[da_stacked.notnull()].indexes['yx']
    mask_locs = xr.DataArray(data=np.array(tuple(index.values)))
    return mask_locs


def weighted_yearly_mean(ds, var, mask_locs):
    """
    weight by latitude 
    """
    # Subset our dataset for our variable
    obs = ds[var].sel()
    dim = list(obs.dims)
    shape =  ['time','lon','lat']
    obs = obs.squeeze(list(set(dim).difference(set(shape))))
    # Select lme grid by mask_locs
    lmeobs = obs[:,mask_locs[:, 0], mask_locs[:, 1]] 
    print(f'# Total number of lme grid: {len(mask_locs)}')
    lmeobs = log_transform(lmeobs)
    lmeobs = calculate_stand_anomaly(lmeobs)
    # Resample by year 
    resampled = lmeobs.resample(time='AS').mean('time')
    # Creating weights
    #For a rectangular grid the cosine of the latitude is proportional to the grid cell area.
    weights = np.cos(np.deg2rad(resampled.lat))
    weights.name = "weights"
    # Return the weighted average
    lme_weighted = resampled.weighted(weights)
    return lme_weighted.mean('dim_0').isel(time=slice(1, None))



def label_generator(path, mask):
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    maskds = xr.open_dataset("/media/cmlws/Data1/jsp/DLdata/LME66.mask.nc").load()
    locs = lme_mask(maskds, mask)
    lme = weighted_yearly_mean(data, v[0], locs)
    return lme

obs = []
for ld in np.arange(2):
    lm_labels = []
    for path in (target_test_files):
        for lmn in np.arange(1,67):
            label = label_generator(path, lmn)
            lm_labels.append(label)
    obs.append(lm_labels)

lme = 3
ld = 0
directory = 'cs_hyper_model'
ob=obs[ld][lme-1]
fcst = np.load(f'/media/cmlws/Data2/jsp/LMEpredict/{directory}/cnn/{ld}/{lme}/fcst{ld}.npy')


df = pd.DataFrame(fcst[:,:,0])

exdf = np.exp(df)
expob = np.exp(ob)
cor_list_md =[]
pv_list_md =[]
for j in range(5):
    cnn = fcst[j,:,0]
    cnn = np.exp(cnn)
    _, _, corr, pval= np.apply_along_axis(lambda y,x: stats.linregress(x,y)[0:4],axis=0,arr=cnn,x=expob)
    cor = np.round(corr, 2)
    pv = np.round(pval, 2)
    cor_list_md.append(cor)     
    pv_list_md.append(pv)  


print(cor_list_md)
print(pv_list_md) 

cordf = pd.DataFrame(dict(r=cor_list_md, p=pv_list_md))
cordf.mean()

R = np.round(cordf.mean().r, 2)
P = np.round(cordf.mean().p, 2)

fig, ax = plt.subplots(figsize=(15,5))
X = np.arange(1999, 2019, 1)
ax.plot (X, exdf.mean(), label='predict')
ax.plot(X, expob, label='obs')
lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

ax.set_xlim([1998.5, 2019.5])
ax.set_xticks(X)
ax.set_xlabel("Year")
ax.set_ylabel("Chlorophyll yearly mean")
ax.set_title("LME3(California Current) Prediction - Kerner Tuned")
textstr = f'r={R} p={P}'
ax.text(0.95, 0.01, textstr,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
plt.tight_layout()
fig.savefig("/media/cmlws/Data2/jsp/FIG/cs_hyper_lme3.png",            
            dpi=300, 
            format='png', 
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
plt.show()