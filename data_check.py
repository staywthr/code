import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import glob

indir = f"/media/cmlws/Data2/jsp/padLMEdata/0/66/RGB"
chl_train_x = np.load(f"{indir}/chl_tr_x_66.npy")