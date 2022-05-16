#!/usr/bin/env python
# coding: utf-8

# Import modules

# In[ ]:


import os
if not os.path.exists(targetfolder): os.mkdir(targetfolder)
from calc import *
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
import pandas as pd
import numpy as np
from copy import deepcopy
plt.rcParams['font.size'] = 10.0
plt.rcParams['font.sans-serif'] = 'Arial'
tol = 1.0e-15
c = {'RK18': 'r', 'C03': 'b'}


# Set model parameters

# In[ ]:


# Where should outputs go?
targetfolder = 'OUTPUT/0515'
# How many samples in any given random sample?
nruns = 1000
# Which crustal growth model to start with?
curve = 'RK18'
# How far back in time to run the model, in Ga?
tmax = 4.0
# With what timestep resolution, in Ga? (0.005 fast; 0.001 OK)
timestep = 0.005
# What raw/processed output to save?
toprint = {'trajectories': False, 'statistics': True}


# Import files from which to interpolate model parameters.

# In[ ]:


startTs = pd.read_csv('CorrectedTpPresent.csv', header=0) #best-fit-to-Phanero present Tp's for each beta
estimate_start_Tp = interpolate.interp1d(x=startTs.b0, y=startTs.Tp_AbbottMatch) #(assuming RK18 curve)
GrowthCurves = interpolate_growth_curve(timestep, tmax)
GrowthCurves.index = np.round(GrowthCurves.index, 3)
pars = generate_parameters(nruns)


# First, which $\beta$ fits best assuming NO uncertainty?

# In[ ]:


# Heat production from published means
HPhistory = fast_median_HPE_budget(GrowthCurves)
# Midpoint method used for time integration of heat production
midpoints = (HPhistory+(HPhistory.diff()*0.5).iloc[1:].set_index(HPhistory.index[:-1])).dropna()
# Between midpoints and instantaneous... which to use?
HPtouse = midpoints
# Establish beta range to test
b0s = np.round(np.arange(-0.15, 0.05+tol, 0.001), 3)
# ID starting temperatures which produce best phanerozoic results for each beta
Tps = dict(zip(b0s, estimate_start_Tp(b0s)))
# Assume mean values of other model parameters
Ea, Qt = pars['Ea'].mu, pars['Qtot'].mu - HPhistory.crust[0]
# Get ready to receive model outputs
curves, statistics = {curve: pd.DataFrame(index=HPtouse.index)}, {curve: None}

for ct, b in enumerate(b0s):
    curves[curve][b] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                 Tp=Tps[b], Ea=Ea, Qt=Qt, HP=HPtouse[curve])

scores = Z_scores(curves[curve])
odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
odds.columns = odds.columns+' odds'
statistics[curve] = deepcopy(pd.concat([scores, odds], axis=1))
print('Best-fitting beta:', statistics[curve]['RMSE'].idxmin())
print('Highest beta within 95%CI:', statistics[curve][statistics[curve].RMSE<2].index[-1])


# In[ ]:


del scores
del odds
if toprint['trajectories']:
    curves[curve].to_csv(targetfolder+'/beta_mean_trajectories_'+curve+'.csv')
if toprint['statistics']: 
    curves[curve].to_csv(targetfolder+'/beta_mean_statistics_'+curve+'.csv')
try: [print(i) for i in os.listdir(targetfolder) if '.csv' in i]
except: pass


# Now, consider uncertainties in each model parameter, using that ideal beta.

# In[ ]:


Tps = dict(zip(b0s, estimate_start_Tp(b0s)))
b = statistics[curve]['RMSE'].idxmin()
Tp_mu = Tps[b]
Tps = Tp_mu+mu_sig(nruns,0,10).dist
Qms = pars['Qtot'].dist - HPhistory.crust[0]
ensemble_df = pd.DataFrame(index=HPtouse.index, columns=list(range(nruns)))
ensembles = {'Tp': deepcopy(ensemble_df),
            'Ea': deepcopy(ensemble_df),
            'Qtot': deepcopy(ensemble_df),
            'Ht': deepcopy(ensemble_df)}

for n, Tp_vary in enumerate(Tps):
    ensembles['Tp'][n] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                 Tp=Tp_vary, Ea=pars['Ea'].mu, Qt=Qt, HP=HPtouse[curve])
for n, Ea_vary in enumerate(pars['Ea'].dist):
    ensembles['Ea'][n] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                 Tp=Tp_mu, Ea=Ea_vary, Qt=Qt, HP=HPtouse[curve])
for n, Qt in enumerate(Qms):
    ensembles['Qtot'][n] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                 Tp=Tp_mu, Ea=pars['Ea'].mu, Qt=Qt, HP=HPtouse[curve])

HPhistories = generate_HP_distributions(nruns, GrowthCurves)
HP_mantle = HPhistories['midpoints'][curve]
HP_crust = HPhistories['instantaneous']['Crust'].iloc[0]
Qms_from_Ht = pars['Qtot'].mu - HP_crust
for n, HP in HP_mantle.items():
    ensembles['Ht'][n] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                 Tp=Tp_mu, Ea=pars['Ea'].mu, Qt=Qms_from_Ht[n], HP=HP)

ensemble_statistics = {}
for param, df in ensembles.items():
    scores = Z_scores(df)
    odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
    odds.columns = odds.columns+' odds'
    ensemble_statistics[param] = deepcopy(pd.concat([scores, odds], axis=1))

for param, df in ensembles.items():
    if toprint['trajectories']: df.to_csv(targetfolder+'/trajectories_'+curve+'_'+param+'_'+str(b)+'.csv')
    if toprint['statistics']: df.to_csv(targetfolder+'/statistics_'+curve+'_'+param+'_'+str(b)+'.csv')

try: [print(i) for i in os.listdir(targetfolder) if '.csv' in i]
except: pass


# Visualize these distributions.

# In[ ]:


fig, ax = plt.subplots(2,4, figsize=(10,5), sharey='row', sharex=False, tight_layout=True)
row0, row1, label0, label1 = ax[0], ax[1], 'abcd', 'efgh'
bins=np.arange(0,1,0.05)
row0[0].set_ylabel(r'Mantle T$_p$'), row0[0].set_yticks([1600,1700,1800,1900,2000,2100])
#row0[n].set_xlabel('Time (Ga)'), 
for n, i in enumerate(['Tp', 'Ea', 'Qtot', 'Ht']):
    estats = ensembles[i].T.quantile(percentiles).T
    lines = estats.values.transpose()
    row0[n].fill_between(x=estats.index, y1=lines[0], 
                       y2=lines[-1], color=c[curve], alpha=0.1)
    row0[n].fill_between(x=estats.index, y1=lines[1], 
                       y2=lines[-2], color=c[curve], alpha=0.2)
    estats[estats.columns[2]].plot(ax=row0[n], c=c[curve])
    plot_gaussian_target(row0[n])
    row0[n].text(x=3.7,y=2050,s=label0[n], fontsize=12)
    row1[n].text(x=0.92,y=510,s=label1[n], fontsize=12) 
    ensemble_statistics[i]['RMSE odds'].plot.hist(ec='w', ax=row1[n], bins=bins, color=c[curve])
    row0[n].set_xlabel('Time (Ga)')

for i in row1:
    i.set_xlabel('Odds Ratio')
    i.set_xticks([0,0.25,0.5,0.75,1])
    i.set_xlim(0,1)
    i.set_ylim(0, 570)
    
plt.savefig(targetfolder+'/'+curve+'_param_sensitivity.png', dpi=200)
plt.savefig(targetfolder+'/'+curve+'_param_sensitivity.pdf')


# Consider the effect of these uncertainties all together.

# In[ ]:


combined = pd.DataFrame(columns=HP_mantle.columns, index=HP_mantle.index)
for n in combined.columns:
    combined[n] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0, 
                 Tp=Tps[n], Ea=pars['Ea'].dist[n], Qt=Qms_from_Ht[n], HP=HP_mantle[n])
scores = Z_scores(combined)
odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
odds.columns = odds.columns+' odds'
ensemble_statistics['All'] = deepcopy(pd.concat([scores, odds], axis=1))


# In[ ]:


fig, ax = plt.subplots(2,1, tight_layout=True, figsize=(3.5,6), sharex='none', sharey='none')
plot_percentiles_over_time(df=combined, ax=ax[0], c=c[curve], name='All', both=True)
plot_gaussian_target(ax[0])
ensemble_statistics['All']['RMSE odds'].plot.hist(ax=ax[1], ec='w', xlim=(0,1), color=c[curve])
ax[1].set_xlabel('Odds Ratio')
ax[0].set_xlabel('Time (Ga)')
ax[0].set_ylabel(r'Mantle T$_p$')

plt.savefig(targetfolder+'/'+curve+'_param_sensitivity_ALL.png', dpi=200)
plt.savefig(targetfolder+'/'+curve+'_param_sensitivity_ALL.pdf')

if toprint['trajectories']: df.to_csv(targetfolder+'/trajectories_'+curve+'_all_'+str(b)+'.csv')
if toprint['statistics']: df.to_csv(targetfolder+'/statistics_'+curve+'_all_'+str(b)+'.csv')

try: [print(i) for i in os.listdir(targetfolder) if '.csv' in i]
except: pass


# Finally, consider role of changing beta in mean distribution.

# In[ ]:


Ea = pars['Ea'].mu
Qt = pars['Qtot'].mu-HPhistory.crust.iloc[0]
HP = HPhistory[curve]
chgts = [0.5, 0.9, 1.3, 1.7]
b0s = np.arange(-0.15, 0.3+tol, 0.01)
n=0
entries = {}
for b0 in b0s:
    Tp = estimate_start_Tp(b0)
    for b1 in b0s:
        for chgt in chgts:
            entry[n] = pd.Series({'b0':b0, 'b1':b1, 'chgt':chgt})
            trajec = pd.DataFrame(fast_evolve_singlemodel_twobeta(b0=b0, b1=b1, chgt=chgt, 
                 Tp=Tp, Ea=Ea, Qt=Qt, HP=HP), index=HP.index)
            scores = Z_scores(trajec)
            odds = scores.apply(stats.norm.pdf)/(stats.norm.pdf(0))
            odds.columns = odds.columns+' odds'
            statistics = (pd.concat([scores.iloc[0], odds.iloc[0], entry[n]], axis=0))
            entries[n] = statistics
            n=n+1
        
entries = pd.DataFrame(entries).T


# In[ ]:


timegroups = entries.groupby(by='chgt')
fig, ax = plt.subplots(1, len(timegroups), gridspec_kw={'wspace': 0},
                       figsize=(2*len(timegroups),2), sharey=True)
n=0
b0_labels = np.round(b0s,3)
ax[0].set_ylabel(r'$\beta_{past}$')
for chgt, group in timegroups:
    result = group.pivot(index='b0', columns='b1', values='RMSE odds').T.values
    ax[n].pcolormesh(result, cmap='gist_stern_r', vmin=0, vmax=1)
    ax[n].set_aspect('equal'), 
    ax[n].set_xticks([]), ax[n].set_yticks([])
    ax[n].set_title(str(chgt)+'Ga')
    ax[n].set_xlabel(r'$\beta_{present}$')
    n=n+1
for ax in fig.get_axes(): 
    #ax.xlabel(r'$\beta_present')
    ax.label_outer()
entries.to_csv(targetfolder+'/Grid_b0b1chgt'+curve+'.csv')
plt.savefig(targetfolder+'/Grid_b0b1chgt'+curve+'.pdf')
plt.savefig(targetfolder+'/Grid_b0b1chgt'+curve+'.png', dpi=200)

