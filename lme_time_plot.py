import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr
import numpy as np
from scipy import stats
import pandas as pd
import os
import glob

lme = 55
ld = 0
directory = 'cs_his+reval_base'
fcst = np.load(f'/media/cmlws/Data2/jsp/LMEpredict/{directory}/cnn/{ld}/{lme}/fcst{ld}.npy')
ob = np.load(f'/media/cmlws/Data2/jsp/cmip6LMEdata/{ld}/{lme}/historical/test_y.npy')
df = pd.DataFrame(fcst[:,:,0])

exdf = np.exp(df)
expob = np.exp(ob[:,0])
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
ax.set_title("LME55(Beaufort Sea) Prediction - CNN")
textstr = f'r={R} p={P}'
ax.text(0.95, 0.01, textstr,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)
plt.tight_layout()

fig.savefig("/media/cmlws/Data2/jsp/FIG/cnn_lme55.png",            
            dpi=300, 
            format='png', 
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
plt.show()