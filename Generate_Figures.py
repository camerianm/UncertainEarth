#!/usr/bin/env python
# coding: utf-8

# ### Import modules

# In[ ]:


import numpy as np
import pandas as pd
import os
from calc import *
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

plt.rcParams['font.size'] = 10.0
plt.rcParams['font.sans-serif'] = 'Arial'


# ### Net crustal growth curves

# In[ ]:


nxn = 2.9
GrowthCurves = pd.read_csv('GrowthCurves.csv', header=0, index_col=0)
GrowthCurves.columns = GrowthCurves.columns.map(Curves)
GrowthCurves.plot(xlabel='Time (Ga)', ylabel='Fraction of Continental Crust', title=None,#'Crustal Growth Models', 
                  cmap='bwr_r', figsize=(nxn, nxn), xlim=(0, GrowthCurves.index[-1]), ylim=0, legend=True)
plt.xlim(0,4.5)
plt.tight_layout()
plt.savefig('OUTPUT/GrowthModels.pdf')


# ### Mantle heat-production implications from crustal growth

# In[ ]:


timestep, GrowthCurves, times, timerange = generate_time_evolution(tmax)
HPE_budgets = generate_HPE_budgets(1000, timestep, [0.0, tmax])
mHP = {}
for curve, crustfrac in GrowthCurves.items():
    mHP[curve] = HPE_budgets[3] - (HPE_budgets[4].mul(crustfrac, axis=0).dropna(how='all'))
fig, ax = plt.subplots(figsize=(nxn, nxn))
c = 'rb'
curve_stats = dict()
for i, curve in enumerate(GrowthCurves):
    curve_stats[curve] = mHP[curve].T.quantile(percentiles).T
    curve_stats[curve].columns = [r'-2$\sigma$', r'-1$\sigma$', '0', r'+1$\sigma$', r'+2$\sigma$']
    plt.fill_between(y1=curve_stats[curve][r'-2$\sigma$'], y2=curve_stats[curve][r'+2$\sigma$'], 
                     x=curve_stats[curve].index, color=c[i], alpha=0.2)
    curve_stats[curve]['0'].plot(color=c[i], label=Curves[curve])
plt.legend()
plt.xlim(0,4)
plt.ylim(5,75)
plt.ylabel('Mantle heat production (TW)')
plt.xlabel('Time (Ga)')
plt.tight_layout()
plt.savefig('OUTPUT/GrowthModels_to_HeatProduction.pdf')


# ### Sample: mean cooling pathways

# In[ ]:


trajectories = pd.read_csv('OUTPUT/Sample_Trajectories_reduced.csv', header=0, index_col=0)
trajectories.columns = trajectories.columns.str.replace('Beta',  'β')
trajectories = trajectories[trajectories.columns[::-1]]
fig, ax = plt.subplots(figsize=(nxn, nxn))
trajectories.plot(xlim=(0,4), ax=ax, xlabel='Time (Ga)', ylabel=r'Mantle $T_p$ (K)')
plot_gaussian_target(ax)
plt.tight_layout()
plt.savefig('OUTPUT/C03_beta_single_comparison.pdf')


# ### Compare possible scaling-law transitions

# In[ ]:


C03 = pd.read_csv('OUTPUT/C03_reduced.csv', header=0, index_col=0).sort_values(by=['b1', 'b0'], ascending=False) 
C03groups = C03.groupby(by='chgt')
rowcols = sorted(C03.b1.unique())

n=0
nxn=4
tol=1.0e-15
panels='abcd'
subplotno = [[0,0],[0,1],
             [1,0],[1,1]]
ticks = np.arange(-0.15, 0.3+tol, 0.05)
ticklabels = pd.Series(ticks).round(2).astype(str)
textloc = dict(zip([0.5, 0.8, 1.1, 1.4], [[0.05,0.85], [0.05,0.85], [0.11, 0.85], [0.24, 0.85]]))
fig, ax = plt.subplots(2,2, sharex=False, sharey=False, figsize=(nxn,nxn), constrained_layout=True)
for i, data in C03groups:
    if i not in textloc.keys():
        continue
    else:
        pivoted = pd.pivot_table(data, values='ZRMSE', index='b0', columns='b1', fill_value=None, dropna=False).T
        pivoted = pivoted.reindex(index=rowcols, columns=rowcols, fill_value=np.nan).fillna(10)
        panel = ax[subplotno[n][0]][subplotno[n][1]]
        im = panel.pcolormesh(pivoted.values, snap=True, vmin=0, vmax=4, cmap='gist_stern', shading='gouraud')
        panel.set_aspect('equal')
        if subplotno[n][0]==0: panel.set_xticks([])
        else: 
            panel.set_xticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_xticklabels(ticklabels, rotation=90)
            panel.set_xlabel(r'$\beta_{present}$')
        if subplotno[n][1]==1: panel.set_yticks([])
        else:
            panel.set_yticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_yticklabels(ticklabels)
            panel.set_ylabel(r'$\beta_{past}$')
        panel.text(x=len(pivoted.columns)*0.85, y=len(pivoted.columns)*0.85, s=panels[n]) #+' (t='+str(i)+')')
        panel.text(x=len(pivoted.columns)*textloc[i][0], y=len(pivoted.columns)*textloc[i][1], s=r'$t_{\Delta\beta}$='+str(i))
        n=n+1
        im.set_edgecolor('none')
cb_ax = fig.add_axes([1.02, 0.175, 0.05, 0.805])
plt.colorbar(im, ax=cb_ax, cax=cb_ax)
plt.savefig('OUTPUT/C03_stats.pdf', bbox_inches='tight')

