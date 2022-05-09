#!/usr/bin/env python
# coding: utf-8

# In[ ]:


targetfolder = '0509'
import os
if not os.path.exists(targetfolder): 
    os.mkdir(targetfolder)
from calc import *
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
import pandas as pd
import time
from copy import deepcopy as copy
plt.rcParams['font.size'] = 10.0
plt.rcParams['font.sans-serif'] = 'Arial'
startTs = pd.read_csv('CorrectedTpPresent.csv', header=0) #best-fit-to-Phanero present Tp's for each beta
estimate_start_Tp = interpolate.interp1d(x=startTs.b0, y=startTs.Tp_AbbottMatch) #(assuming RK18 curve)
nruns=1
pars = generate_parameters(nruns)
tol=1.0e-15


# In[ ]:


tmax = 4.0
timestep = 0.001
GrowthCurves = interpolate_growth_curve(timestep, tmax)
GrowthCurves.index = np.round(GrowthCurves.index, 3)
HPhistories = fast_median_HPE_budget(GrowthCurves)
b0s = np.round(np.arange(-0.15, 0.05+tol, 0.001), 3) #rounding=essential
Tps = dict(zip(b0s, estimate_start_Tp(b0s)))
midpoints = (HPhistories+(HPhistories.diff()*0.5).iloc[1:].set_index(HPhistories.index[:-1])).dropna()
Ea, Qt = pars['Ea'].mu, pars['Qtot'].mu - midpoints.crust[0]
curves, statistics = {}, {}


# In[ ]:


curve = 'C03'
curves[curve] = {}
for ct, b in enumerate(b0s):
    curves[curve][b] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                 Tp=Tps[b], Ea=Ea, Qt=Qt, HP=midpoints[curve])
curves[curve] = pd.DataFrame(curves[curve], index=midpoints.index)
scores = Z_scores(curves[curve])
scores['LAE'] = scores.iloc[:,0:3].abs().T.mean()
odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
odds.columns = odds.columns+' odds'
statistics[curve] = pd.concat([scores, odds], axis=1)


# In[ ]:


curve = 'RK18'
curves[curve] = {}
for ct, b in enumerate(b0s):
    curves[curve][b] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                 Tp=Tps[b], Ea=Ea, Qt=Qt, HP=midpoints[curve])
curves[curve] = pd.DataFrame(curves[curve], index=midpoints.index)
scores = Z_scores(curves[curve])
scores['LAE'] = scores.iloc[:,0:3].abs().T.mean()
odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
odds.columns = odds.columns+' odds'
statistics[curve] = pd.concat([scores, odds], axis=1)


# In[ ]:


[curves[i].to_csv(targetfolder+'/trajects_for_betas_'+i+'.csv') for i in curves]
[statistics[i].to_csv(targetfolder+'/stats_for_betas_'+i+'.csv') for i in statistics]


# In[ ]:


nxn=2.9
fig, ax = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              figsize=(nxn*2.1, nxn*1.3), constrained_layout=True)
upper, lower, side = ax['upper left'], ax['lower left'], ax['right']
c = {'RK18': 'r', 'C03': 'b'}
for curve in curves:
    statistics[curve]['RMSE'].plot(ax=upper, c=c[curve], ylim=(0,4), ylabel='Z-score',  xlim=(-0.15,0.05))
    statistics[curve]['RMSE odds'].plot(ax=lower, c=c[curve], ylabel='Odds Ratio', ylim=(0,1), xlim=(-0.15,0.05))
    best = curves[curve][statistics[curve].RMSE.idxmin()]
    best.plot(ax=side, c=c[curve], ylabel=r'Mantle $T_p$ (K)', xlabel='Time (Ga)', label=curve+r', $\beta$ = '+str(best.name))
    upper.text(x=0.04, y=0.9*4, s='a'), lower.text(x=0.04, y=0.9, s='b'), side.text(x=3.8, y=2080, s='c')
    upper.set_xticks([]), lower.set_xlabel(r'$\beta$')
    side.yaxis.tick_right(), side.yaxis.set_label_position('right')
    upper.axvline(0, lw=0.5, ls='dotted', c='grey'), lower.axvline(0, lw=0.5, ls='dotted', c='grey')
plot_gaussian_target(side)
side.legend(loc='upper left')
plt.savefig(targetfolder+'/'+'odds_and_trajects.pdf')


# In[ ]:




