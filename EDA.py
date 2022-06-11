import matplotlib.pyplot as plt # Plotting package
import xarray as xr
import cartopy.crs as ccrs # Geospatial data processing package for making maps and anayzing
import cartopy.feature as cfeature
import numpy as np
import glob

LOCS = np.load('/media/cmlws/Data1/jsp/DLdata/1/test/chl_latlon_mask_1x1.npy')


test_path = "/media/cmlws/Data1/jsp/DLdata/1/test/"
chl_test_files = sorted(glob.glob(test_path+"chl_199801-201812_1x1.nc"))
sst_test_files = sorted(glob.glob(test_path+"sst_JAN1998*.nc"))
target_test_files = sorted(glob.glob(test_path+"chl_199801-201812_1x1.nc"))

def chk_var(klist): 
    var_name_list = ['eta_t','sshg','ssh','SSH','chl','sfc_chl','chlos','tos','sst','SST']
    var = list(set(klist).intersection(var_name_list))
    return var

def load_xdata(path):
    data = xr.open_dataset(path).load()
    k = list(data.keys())
    v = chk_var(k)
    arr = data[v[0]].sel()
    dim = list(arr.dims)
    shape =  ['time','lon','lat']
    if v == ['chl'] and len(arr.dims) == 4:
        arr = arr.squeeze(list(set(dim).difference(set(shape))))
    return arr.transpose('time','lon','lat')



def cleanining_grid(da, locnpy):
    for i in range(len(locnpy)):
        da[:,locnpy[i][0],locnpy[i][1]] = np.nan 
    print(f'# Total number of missing grid: {len(locnpy)}')
    return da

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


def month_anomaly(var_anom, mind):
    xda = var_anom.where(var_anom.coords['month']==mind,drop=True)
    return xda


def season_anomaly(var_anom,ld):
    """
    returns 3 month anomaly xarray.DataArray.
    Parameters
    ------------------------
    ld = lead month 
    ld == 0(DJF) or ld == 1(NDJ)
    """
    if ld == 1:
        anom0 = month_anomaly(var_anom, 11)
        anom1 = month_anomaly(var_anom, 12)
        anom2 = month_anomaly(var_anom, 1)
        ldmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)),anom2.isel(time=slice(1, None))], dim='z', join='override')
    else:
        # ld == 0
        anom0 = month_anomaly(var_anom, 12)
        anom1 = month_anomaly(var_anom, 1)
        anom2 = month_anomaly(var_anom, 2)
        ldmn = xr.concat([anom0.isel(time=slice(0, -1)), anom1.isel(time=slice(0, -1)),anom2.isel(time=slice(1, None))], dim='z', join='override')

    return ldmn.transpose('time','lon','lat','z')

def season_mean_anomaly(var_anom,ld):
    """
    returns 3 month rolling mean anomaly xarray.DataArray.
    Parameters
    ------------------------
    ld = lead month 
    ld == 0(DJF) or ld == 1(NDJ)
    """
    anom = var_anom.rolling(time=3, center=True).mean()
    ldmn = anom[ld::12]
    ldmn = ldmn.isel(time=slice(0,-1))

    return ldmn.transpose('time','lon','lat')


#xarray 
def log_transform(arr):
    epsilon=1e-06 
    log_norm = np.log(arr+epsilon)
    return log_norm



def prepare_single_frame(path, ld): 
    """
    returns lead months single feature(chl/sst/ssh) xarray from one file
    Parameters
    ------------------------
    path = file path
    ld = lead month
    """
    arr = load_xdata(path)
    if (arr.name == 'chl') or (arr.name == 'chlos') or (arr.name == 'sfc_chl'):
        print("chlorophyll preprocess")
        arr = cleanining_grid(arr, LOCS)
        arr = log_transform(arr)           
    # anom = calculate_anomaly(arr)
    anom = calculate_stand_anomaly(arr)
    ss_anom = season_anomaly(anom, ld)
    # ss_anom = season_mean_anomaly(anom,ld)
    print(f"Lead month {ld} {ss_anom.name} {(path.partition('Omon_')[-1])} Dataset")
    return ss_anom


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


def prepare_ld_sequence(paths, ld, mask):
    """
    returns lead months single feature(chl/sst/ssh) xarray and nino indexes from files
    Parameters
    ------------------------
    path = files path
    ld = lead month
    """
    temp_frames = [] #lead months xarray list 
    temp_labels = [] #lead months nino index xarray list 
    # For each video.
    for path in (paths):
        # Gather all its frames and add a batch dimension.
        frames = prepare_single_frame(path[0],ld)
        label = label_generator(path[1], mask)
        temp_frames.append(frames)
        temp_labels.append(label)
    # frame_features = [el.interpolate_na(dim="lon") for el in temp_frames]    
    frame_features = [el.fillna(0) for el in temp_frames]
    sequence = np.concatenate(frame_features, axis=0)
    labels = np.concatenate(temp_labels, axis=0)
    labels = labels[:,np.newaxis]
    return sequence , labels


chl_test_paths = list(zip(chl_test_files, target_test_files))

sst_test_paths = list(zip(sst_test_files, target_test_files))

ld = 0 
lmen = 42
paths = chl_test_paths
temp_frames = [] #lead months xarray list 
temp_labels = [] #lead months nino index xarray list 
# For each video.
for path in (paths):
    # Gather all its frames and add a batch dimension.
    frames = prepare_single_frame(path[0],ld)

chl1=frames[0,:,:,0].transpose('lat','lon')
chl2=frames[0,:,:,1].transpose('lat','lon')
chl3=frames[0,:,:,2].transpose('lat','lon')


ld = 0 
lmen = 42
paths = sst_test_paths
temp_frames = [] #lead months xarray list 
temp_labels = [] #lead months nino index xarray list 
# For each video.
for path in (paths):
    # Gather all its frames and add a batch dimension.
    frames = prepare_single_frame(path[0],ld)

sst1=frames[0,:,:,0].transpose('lat','lon')
sst2=frames[0,:,:,1].transpose('lat','lon')
sst3=frames[0,:,:,2].transpose('lat','lon')


fig, axes = plt.subplots(nrows = 6, ncols = 1, figsize = (36, 18))
clevs = np.arange(-10, 10, 1) # Set color levels 
ax1 = plt.subplot(311, projection = ccrs.PlateCarree(central_longitude=180))
a=chl1.plot.contourf(ax=ax1, levels=clevs, cmap='jet', transform=ccrs.PlateCarree())
cbar = plt.colorbar(a, shrink=0.9, orientation="horizontal") 
ax1.set_title('CHl(τ−2)', y=1.12)
ax1.coastlines()
ax1.gridlines(draw_labels=True, linewidth=1, linestyle=':', color='gray')
ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='white'))
ax2 = plt.subplot(312, projection = ccrs.PlateCarree(central_longitude=180))
b = chl2.plot.contourf(ax=ax2, levels=clevs, cmap='jet', transform=ccrs.PlateCarree())
ax2.set_title('CHl(τ−1)', y=1.12)
ax2.coastlines()
ax2.gridlines(draw_labels=True, linewidth=1, linestyle=':', color='gray')
ax2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='white'))
ax3 = plt.subplot(313,  projection = ccrs.PlateCarree(central_longitude=180))
c = chl3.plot.contourf(ax=ax3, levels = clevs, cmap='jet', transform=ccrs.PlateCarree())
ax3.set_title('CHL(τ−0)', y=1.12)
ax3.coastlines()
ax3.gridlines(draw_labels=True, linewidth=1, linestyle=':', color='gray')
ax3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='white'))


plt.subplots_adjust(top=1.3) 
plt.savefig('./month_sst_input.png', dpi=300, transparent=True)