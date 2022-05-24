#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from calc import *
from scipy import interpolate
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10.0
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams["savefig.facecolor"] = 'w'
plt.rcParams["savefig.edgecolor"] = 'w'
tol=1.0e-15
nruns=1
startTs = pd.read_csv('CorrectedTpPresent.csv', header=0)
estimate_start_Tp = interpolate.interp1d(x=startTs.b0, y=startTs.Tp_AbbottMatch)
pars = generate_parameters(nruns)


# In[ ]:


# ---------- INPUTS ---------- #
targetfolder = '0523'
timestep = 0.001
tmax = 4.0
b0s = np.round(np.arange(-0.2, 0.05+tol, 0.001), 3) #recommend rounding
# ---------- INPUTS ---------- #


# In[ ]:


if not os.path.exists(targetfolder): os.mkdir(targetfolder)
GrowthCurves = interpolate_growth_curve(timestep, tmax)
GrowthCurves.index = np.round(GrowthCurves.index, 3)
HPhistories = fast_median_HPE_budget(GrowthCurves)
midpoints = (HPhistories+(HPhistories.diff()*0.5).
             iloc[1:].set_index(HPhistories.index[:-1])).dropna()
Ea, Qt = pars['Ea'].mu, pars['Qtot'].mu - midpoints.crust[0]
Tps = dict(zip(b0s, estimate_start_Tp(b0s)))


# In[ ]:


curves, statistics = {}, {}
for curve in ['C03', 'RK18']:
    curves[curve] = {}
    for ct, b in enumerate(b0s):
        curves[curve][b] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                     Tp=Tps[b], Ea=Ea, Qt=Qt, HP=midpoints[curve])
    curves[curve] = pd.DataFrame(curves[curve], index=midpoints.index)
    scores = Z_scores(curves[curve])
    odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
    odds.columns = odds.columns+' odds'
    statistics[curve] = pd.concat([scores, odds], axis=1)
[curves[i].to_csv(targetfolder+'/SingleBetaFitAssessment_trajects_'+i+'.csv') for i in curves]
[statistics[i].to_csv(targetfolder+'/SingleBetaFitAssessment_stats_'+i+'.csv') for i in statistics]
print('finished running model!')


# In[ ]:


overall = pd.DataFrame({'C03 RMSE': statistics['C03']['RMSE'],
 'C03 RMSE Archean-only': (((statistics['C03'].Tp_latearc)**2 + 
                  (statistics['C03'].Tp_midarc)**2)*0.5)**0.5,
 'RK18 RMSE': statistics['RK18']['RMSE'],
 'RK18 RMSE Archean-only': (((statistics['RK18'].Tp_latearc)**2 + 
                   (statistics['RK18'].Tp_midarc)**2)*0.5)**0.5})
print('2-sigma ranges for beta:')
for col in overall.columns:
    betas = overall[col][overall[col]<=2].index
    print(betas[0], 'to', betas[-1], '\t', col)


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
    upper.text(x=0.035, y=0.85*4, s='a', fontsize=14), lower.text(x=0.035, y=0.85, s='b', fontsize=14), side.text(x=3.7, y=2070, s='c', fontsize=14)
    upper.set_xticks([]), lower.set_xlabel(r'$\beta$')
    side.yaxis.tick_right(), side.yaxis.set_label_position('right')
plot_gaussian_target(side)
side.legend(loc='upper left')
plt.savefig(targetfolder+'/SingleBetaFitAssessment_plot_stats_and_trajects.pdf', bbox_inches='tight')
plt.savefig(targetfolder+'/SingleBetaFitAssessment_plot_stats_and_trajects.png', bbox_inches='tight', dpi=200)

