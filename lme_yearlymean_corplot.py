import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr
import numpy as np
from scipy import stats
import pandas as pd
import os
import glob
# plt.style.use ('ggplot')
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

# creating directory 
os.makedirs('/media/cmlws/Data2/jsp/FIG/', exist_ok=True)

def get_cor_list(en, directory):
    cor_list_ld = []
    pv_list_ld = []
    for ld in np.arange(1):
        for lme in np.arange(1,67):
            ob=obs[ld][lme-1]
            fcst = np.load(f'/media/cmlws/Data2/jsp/LMEpredict/{directory}/cnn/{ld}/{lme}/fcst{ld}.npy')
            cor_list_md =[]
            pv_list_md =[]
            for j in range(en):
                cnn = fcst[j,:,0]
                _, _, corr, pval= np.apply_along_axis(lambda y,x: stats.linregress(x,y)[0:4],axis=0,arr=cnn,x=ob)
                cor = np.round(corr, 2)
                pv = np.round(pval, 2)
                cor_list_md.append(cor)     
                pv_list_md.append(pv)       
            cor_list_ld.append(cor_list_md)
            pv_list_ld.append(pv_list_md)
    cor_2d = np.swapaxes(cor_list_ld,0,1)
    rdf = pd.DataFrame(cor_2d, index=np.arange(en), columns=np.arange(1,67))
    means1 = rdf.mean(axis=0)
    pv_2d = np.swapaxes(pv_list_ld,0,1)
    pdf = pd.DataFrame(pv_2d, index=np.arange(en), columns=np.arange(1,67))
    means2 = pdf.mean(axis=0)
    rp = pd.DataFrame(dict(r=means1, p=means2))
    rp['significant'] = rp['p'].apply(lambda x: 'True' if x<0.10 else 'False')
    return rp

sst_df = get_cor_list(5, 'sst_reval')
chl_df = get_cor_list(5, 'chl_reval')
sc_df = get_cor_list(5, 'cs_reval')


sst_t = sst_df['r'].loc[sst_df['p']<0.10]
sst_f = sst_df['r'].loc[sst_df['p']>=0.10]

chl_t = chl_df['r'].loc[chl_df['p']<0.10]
chl_f = chl_df['r'].loc[chl_df['p']>=0.10]

sc_t = sc_df['r'].loc[sc_df['p']<0.10]
sc_f = sc_df['r'].loc[sc_df['p']>=0.10]



fig, ax = plt.subplots(figsize=(15,5))
X = np.arange(2, 67, 2)
Y = np.round(np.arange(-1, 1.1, 0.1),2)
W = 0.2

# plot bars with face color off
sst_bar = ax.bar(sst_t.index, sst_t, width=W, color='r', label='SST', alpha=0.5)
ax.bar(sst_f.index, sst_f, width=W, edgecolor='r', color='None')

# # plot marks with face color off
# ax.bar(chl_t.index+W, chl_t, width=W, color='g', label='Tanh+MSE')
# ax.bar(chl_f.index+W, chl_f, width=W, edgecolor='g', color='None')

# ax.bar(sc_t.index-W, sc_t, width=W, color='b', label='Tanh+MAE')
# ax.bar(sc_f.index-W, sc_f, width=W, edgecolor='b', color='None')

chl_point = ax.scatter(chl_t.index, chl_t, marker='D', color='g', label='CHL')
ax.scatter(chl_f.index, chl_f, marker='D', edgecolor='g', color='None')

cs_point = ax.scatter(sc_t.index, sc_t, marker='o', color='b', label='SST+CHL')
ax.scatter(sc_f.index, sc_f, marker='o', edgecolor='b', color='None')

ax.set_title("Prediction skill - Validation with reanalisys(None - padded)")
ax.set_xlim([0.5, 66.5])
ax.set_ylim([-1.2, 1.2])
ax.set_xticks(X)
ax.set_yticks(Y)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_xlabel("LME")
ax.set_ylabel("Correlation coefficients")
fig.set_dpi(300)

lg1 = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
art_legend1 = plt.gca().add_artist(lg1)
lg2 = ax.legend([chl_point, cs_point, sst_bar, ], 
                    [f'{np.round(chl_t.mean(), 2)}', 
                    f'{np.round(sc_t.mean(), 2)}',
                    f'{np.round(sst_t.mean(), 2)}', ], 
                    title = '$R_{average}$',
                    bbox_to_anchor=(1.05, 0.0),
                    loc='lower left' )
art_legned2 = plt.gca().add_artist(lg2)
lg3 = ax.legend([chl_point, cs_point, sst_bar, ], 
                    [f'{len(chl_t)}', 
                    f'{len(sc_t)}',
                    f'{len(sst_t)}', ], 
                    title = 'N of significant regions',
                    bbox_to_anchor=(1.05, 0.5),
                    loc='center left' )
art_legned3 = plt.gca().add_artist(lg3)     
               
plt.tight_layout()
fig.savefig("/media/cmlws/Data2/jsp/FIG/LMEcs_No_padding.png",            
            dpi=300, 
            format='png', 
            bbox_extra_artists=(lg1,lg2,lg3), 
            bbox_inches='tight')

plt.show()

# sst_df = get_cor_list(5, 'xrsst_Gelu_norm')
# sst2_df = get_cor_list(5, 'xrsst_t6vRe_norm')

# chl_df = get_cor_list(5, 'xrchl_Gelu_norm')
# chl2_df = get_cor_list(5, 'xrchl_t6vRe_norm')

# sc_df = get_cor_list(5, 'xrcs_Gelu_norm')
# sc2_df = get_cor_list(5, 'xrcs_t6vRe_norm')
# # df.to_csv('/media/cmlws/Data1/jsp/LMEpredict/xrsst_t6vRe_norm.csv')

# fig, ax = plt.subplots(figsize=(15,5))
# X = np.arange(2, 67, 2)
# Y = np.round(np.arange(-1, 1.1, 0.1),2)
# W = 0.3
# # cook up some random bar heights -- exact results may vary :-P

# sst_t = sst_df['r'].loc[sst_df['p']<0.10]
# sst_f = sst_df['r'].loc[sst_df['p']>=0.10]

# chl_t = chl_df['r'].loc[chl_df['p']<0.10]
# chl_f = chl_df['r'].loc[chl_df['p']>=0.10]

# sc_t = sc_df['r'].loc[sc_df['p']<0.10]
# sc_f = sc_df['r'].loc[sc_df['p']>=0.10]


# sst2_t = sst2_df['r'].loc[sst2_df['p']<0.10]
# sst2_f = sst2_df['r'].loc[sst2_df['p']>=0.10]

# chl2_t = chl2_df['r'].loc[chl2_df['p']<0.10]
# chl2_f = chl2_df['r'].loc[chl2_df['p']>=0.10]

# sc2_t = sc2_df['r'].loc[sc2_df['p']<0.10]
# sc2_f = sc2_df['r'].loc[sc2_df['p']>=0.10]

# # plot bars with face color off
# ax.bar(sst_t.index, sst_t, width=W, color='r', alpha=0.5, label='Gelu SST')
# # ax.bar(sst_f.index, sst_f, width=W, edgecolor='r', color='None', alpha=0.5)

# ax.bar(sst2_t.index+W, sst2_t, width=W, color='orange', alpha=0.5, label='Tanh SST')
# # ax.bar(sst2_f.index+W, sst2_f, width=W, edgecolor='orange', color='None', alpha=0.5)
# # plot marks with face color off
# ax.scatter(chl_t.index, chl_t, marker='D', color='g', label='Gelu CHL')
# # ax.scatter(chl_f.index, chl_f, marker='D', edgecolor='g', color='None')
# ax.scatter(chl2_t.index, chl2_t, marker='d', color='lime', label='Tanh CHL')
# # ax.scatter(chl2_f.index, chl2_f, marker='d', edgecolor='lime', color='None')

# ax.scatter(sc_t.index, sc_t, marker='P', color='b', label='Gelu SST+CHL')
# # ax.scatter(sc_f.index, sc_f, marker='P', edgecolor='b', color='None')
# ax.scatter(sc2_t.index, sc2_t, marker='X', color='grey', label='Tanh SST+CHL')
# # ax.scatter(sc2_f.index, sc2_f, marker='X', edgecolor='grey', color='None')
# # plt.axhline(y=0.0, color='k',  linestyle='--', linewidth=0.1)
# lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# ax.set_title("Prediction skill")


# # set ticks
# ax.set_xlim([0.5, 66.5])
# ax.set_ylim([-1.2, 1.2])
# ax.set_xticks(X)
# ax.set_yticks(Y)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
# ax.set_xlabel("LME")
# ax.set_ylabel("Correlation coefficients")
# fig.set_dpi(300)
# lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
# fig.savefig("/media/cmlws/Data2/jsp/FIG/LMEscs_30_act_all_pv<0.10_stats.png",            
#             dpi=300, 
#             format='png', 
#             bbox_extra_artists=(lg,), 
#             bbox_inches='tight')
# plt.show()
