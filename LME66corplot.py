import numpy as np
# Fundamental package for scientific computing
import matplotlib.pyplot as plt # Plotting package
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy as cy
# Geospatial data processing package for making maps and anayzing
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes


#------------ Reading .nc data -------------
var = xr.open_dataset("/media/cmlws/Data1/jsp/DLdata/LME66.mask.nc").load()
#------------- Plotting a Map---------------
fig = plt.figure(figsize=(18,14), dpi=100)
clevs = np.arange(1, 67, 1) # Set color levels
ax1 = plt.subplot(221, projection = ccrs.PlateCarree())
var.mask.plot.contourf(ax=ax1, levels = clevs, cmap='jet', transform=ccrs.PlateCarree(),
cbar_kwargs={'extendrect': 'True', 'orientation': 'horizontal', 'pad': 0.06, 'aspect': 30})
ax1.set_title('LME',fontsize=17)
ax1.coastlines()
ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='black'))
# ax1.gridlines(draw_labels=True, linewidth=1, linestyle=':', color='gray', x_inline=False)


# plt.style.use ('ggplot')
maskds = xr.open_dataset("/media/cmlws/Data1/jsp/DLdata/LME66.mask.nc").load()
lons = maskds['lon']
lats = maskds['lat']
lme_da64 = maskds.mask.where(maskds.mask == 64)
lme_da3 = maskds.mask.where(maskds.mask==3)
#------------- Plotting a Map---------------
fig = plt.figure(figsize=(15,5), dpi=100)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
clevs = np.arange(-0.2, 0.2, 0.001) # Set color levels

ax.coastlines()
gl = ax.gridlines()
gl.xlabels_top = False
gl.ylabels_right = False
# lme_da64.plot(ax=ax, cmap=plt.get_cmap('Blues'),transform=ccrs.PlateCarree())
# lme_da3.plot(ax=ax, cmap=plt.get_cmap('Reds'),transform=ccrs.PlateCarree())
ax.plot(lons, lats, maskds.mask, transform=ccrs.PlateCarree())
# ----------- Save plot ------------
fig.tight_layout()

plt.show()
