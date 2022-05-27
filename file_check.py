
import numpy as np
import glob
import pandas as pd
from scipy import stats
ldmn = 0



model = []
for lmen in range(1, 67):
    indir = f"/media/cml/Data1/jsp/cmip6LMEdata/{ldmn}/{lmen}/DJF/"
    target_train_files = sorted(glob.glob(indir + '*train_y.npy'))


    cmip6_y = []
    for i in range(15):
        cmip6yi = np.load(target_train_files[i])
        cmip6_y.append(cmip6yi)

    cmipy = np.concatenate(cmip6_y, axis=1)

    df = pd.DataFrame(cmipy)
    df2 = df.T
   

    df3 = df2[(np.abs(stats.zscore(df2)) < 3).all(axis=1)]
    """For each column(year), it first computes the Z-score of each value in the column,  relative to the column mean and standard deviation.
    It then takes the absolute Z-score because the direction does not matter, only if it  is below the threshold.
    all(axis=1) ensures that for each row, all column satisfy the constraint.
    Finally, the result of this condition is used to index the dataframe.
    https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe
    """
    model_list = df3.index 
    model.append(model_list)




lmen = 62
indir = f"/media/cml/Data1/jsp/cmip6LMEdata/{ldmn}/{lmen}/DJF/"
target_train_files = sorted(glob.glob(indir + '*train_y.npy'))
cmip6_y = []
for i in range(15):
    cmip6yi = np.load(target_train_files[i])
    cmip6_y.append(cmip6yi)
cmipy = np.concatenate(cmip6_y, axis=1)
df = pd.DataFrame(cmipy)
df2 = df.T
df3 = df2[(np.abs(stats.zscore(df2)) < 3).all(axis=1)]

import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(15,5))
# X = np.arange(1851, 2015, 1)
# ax.plot (X, df, label='log chl yearly mean')
# lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

# ax.set_xlim([1861.5, 2015.5])
# ax.set_xticks(X)
# ax.set_xlabel("Year")
# ax.set_title("LME3(California Current) Prediction")