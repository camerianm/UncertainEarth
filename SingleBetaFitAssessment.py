#!/usr/bin/env python
# coding: utf-8

# In[ ]:


targetfolder = '0502'
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


# In[ ]:


growthcurve = 'C03'
timestep, GrowthCurves, times, timerange = generate_time_evolution(4.0)
tmax = timerange[-1]
HPE_budgets = strict_median_heat_budget(nruns, timestep, [0.0, tmax])
tol=1.0e-15
b0s = np.round(np.arange(-0.15, 0.3+tol, 0.001), 4)
Tps = estimate_start_Tp(b0s)
every = 1
Ea, Qt = pars['Ea'].mu, pars['Qtot'].mu - HPE_budgets[4].iat[0,0]
crustfrac = GrowthCurves[growthcurve]
HP = (HPE_budgets[3] - (HPE_budgets[4].mul(crustfrac, axis=0)).dropna()).T.drop_duplicates().T[0]
results = dict()
for ct, b in enumerate(b0s):
    results[b] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                            Tp=Tps[ct], Ea=Ea, Qt=Qt, HP=HP, every=every)
results = pd.DataFrame(results)
scores = Z_scores(results)
scores['LAE'] = np.abs(scores[scores.columns[0:3]]).T.mean()
odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
odds.columns = odds.columns+' odds'
C03_summary = copy(pd.concat([scores[['RMSE', 'LAE']], odds[['RMSE odds', 'LAE odds']]], axis=1))


# In[ ]:


growthcurve = 'RK18'
timestep, GrowthCurves, times, timerange = generate_time_evolution(4.0)
tmax = timerange[-1]
HPE_budgets = strict_median_heat_budget(nruns, timestep, [0.0, tmax])
tol=1.0e-15
b0s = b0s
Tps = estimate_start_Tp(b0s)
every = 1
Ea, Qt = pars['Ea'].mu, pars['Qtot'].mu - HPE_budgets[4].iat[0,0]
crustfrac = GrowthCurves[growthcurve]
HP = (HPE_budgets[3] - (HPE_budgets[4].mul(crustfrac, axis=0)).dropna()).T.drop_duplicates().T[0]
results2 = dict()
for ct, b in enumerate(b0s):
    results2[b] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                            Tp=Tps[ct], Ea=Ea, Qt=Qt, HP=HP, every=every)
results2 = pd.DataFrame(results2)
scores = Z_scores(results2)
scores['LAE'] = np.abs(scores[scores.columns[0:3]]).T.mean()
odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
odds.columns = odds.columns+' odds'
RK18_summary = copy(pd.concat([scores[['RMSE', 'LAE']], odds[['RMSE odds', 'LAE odds']]], axis=1))


# In[ ]:


best_C03_RMSE = C03_summary.RMSE.sort_values().index[0]
best_C03_LAE = C03_summary.LAE.sort_values().index[0]
best_RK18_RMSE = RK18_summary.RMSE.sort_values().index[0]
best_RK18_LAE = RK18_summary.LAE.sort_values().index[0]


# In[ ]:


best_RK18 = results2[[best_RK18_RMSE, best_RK18_LAE]]
best_RK18.columns = [' RMSE', ' LAE']
best_C03 = results[[best_C03_RMSE, best_C03_LAE]]
best_C03.columns = [' RMSE', ' LAE']
best_runs = {'C03': best_C03, 'RK18': best_RK18}

fig, ax = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              figsize=(nxn*2.4, nxn*1.2), constrained_layout=True)
summaries = {'RK18': RK18_summary, 'C03': C03_summary}
c = {'RK18': 'r', 'C03': 'b'}
nxn=2.9
#fig, ax = plt.subplots(2,1, figsize=(nxn, nxn*2), sharex=True)
n=0
for curve in summaries.keys():
    summaries[curve]['RMSE'].plot(ylim=(0,4), xlim=(-0.15,0.05), label=curve+' RMSE', ax=ax['upper left'], ylabel='Z-score', c=c[curve])
    summaries[curve]['LAE'].plot(ylim=(0,4), xlim=(-0.15,0.05), label=curve+' LAE', ax=ax['upper left'], ylabel='Z-score', c=c[curve], ls='dotted')
    #ax['upper left'].legend()
    ax['upper left'].set_xticks([])
    summaries[curve]['RMSE odds'].plot(ylim=(0,4), xlim=(-0.15,0.05), label=curve+' RMSE', ax=ax['lower left'], ylabel='Odds Ratio', xlabel=r'$\beta$', c=c[curve])
    summaries[curve]['LAE odds'].plot(ylim=(0,1), xlim=(-0.15,0.05), label=curve+' LAE', ax=ax['lower left'], ylabel='Odds Ratio', c=c[curve], ls='dotted')
    best_runs[curve][' RMSE'].plot(ax=ax['right'], label=curve+' RMSE', c=c[curve], ylabel=r'Mantle $T_p$ (K)')
    best_runs[curve][' LAE'].plot(ax=ax['right'], label=curve+' LAE', c=c[curve], ls='dotted', xlabel='Time (Ga)')
    ax['right'].legend(loc='upper left')
    plot_gaussian_target(ax['right'])
    ax['right'].yaxis.tick_right()
    ax['right'].yaxis.set_label_position('right')

plt.savefig(targetfolder+'/'+'odds_and_trajects.pdf')


# In[ ]:


results.to_csv(targetfolder+'/trajects_for_betas_C03.csv')
results2.to_csv(targetfolder+'/trajects_for_betas_RK18.csv')

