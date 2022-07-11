import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import glob

indir = f"/media/cmlws/Data2/jsp/cmip6LMEdata/0/66/historical"
chl_train_x = np.load(f"{indir}/chl_historical_tr_x.npy")