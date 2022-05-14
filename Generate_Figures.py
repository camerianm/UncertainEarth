#!/usr/bin/env python
# coding: utf-8

# ### Import modules

# In[ ]:


targetfolder = 'FIGURES'
import os
import numpy as np
import pandas as pd
from calc import *
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
if not os.path.exists(targetfolder): os.mkdir(targetfolder)
plt.rcParams['font.size'] = 10.0
plt.rcParams['font.sans-serif'] = 'Arial'


# ### Net crustal growth curves

# In[ ]:


c = {'RK18':'r','C03':'b'}
nxn = 2.5
GrowthCurves = pd.read_csv('GrowthCurves.csv', header=0, index_col=0)
#GrowthCurves.columns = GrowthCurves.columns.map(Curves)
GrowthCurves.plot(xlabel='Time (Ga)', ylabel='Fraction of Continental Crust', title=None,
                  cmap='bwr_r', figsize=(nxn, nxn), xlim=(0, GrowthCurves.index[-1]+0.05), ylim=0, legend=True)
plt.xticks([0,1,2,3,4])
plt.savefig(targetfolder+'/GrowthModels.pdf', bbox_inches='tight', facecolor='w', edgecolor='w')
plt.savefig(targetfolder+'/GrowthModels.png', bbox_inches='tight', facecolor='w', edgecolor='w', dpi=200)


# ### Mantle heat-production implications from crustal growth

# In[ ]:


tmax = 4.0
timestep = 0.05
nruns = 10000
GrowthCurves = interpolate_growth_curve(timestep, tmax)
GrowthCurves.index = np.round(GrowthCurves.index, 3)
HPHistory_Mean = fast_median_HPE_budget(GrowthCurves)
HPHistory_Ensembles = generate_HP_distributions(nruns=nruns, Curves=GrowthCurves)['instantaneous']

fig, ax = plt.subplots(figsize=(nxn, nxn))
for curve in c.keys(): #middle col = same as HPHistory_Mean[curve].plot(ax=ax, c=c[curve], label=None)
    plot_percentiles_over_time(HPHistory_Ensembles[curve], ax=ax, c=c[curve], name=curve, both=False)
plt.legend(loc='upper left')
plt.yticks(np.arange(10,80,10))
plt.ylabel('Mantle heat produced (TW)')
plt.savefig(targetfolder+'/GrowthModels_Production.pdf', bbox_inches='tight', facecolor='w', edgecolor='w')
plt.savefig(targetfolder+'/GrowthModels_Production.png', bbox_inches='tight', facecolor='w', edgecolor='w', dpi=200)


# ### Sample mean cooling pathways

# In[ ]:


nxn = 3.5
trajectories = pd.read_csv('OUTPUT/Sample_Trajectories_reduced.csv', header=0, index_col=0)
trajectories.columns = trajectories.columns.str.replace('Beta',  'β')
trajectories = trajectories[trajectories.columns[::-1]]
fig, ax = plt.subplots(figsize=(nxn*1.5, nxn))
alphas = [1.0, 1.0, 1.0]
lss = ['dotted', 'dashed', '-']
n = 0
for col, item in trajectories.items():
    item.plot(c='b', alpha=alphas[n], ax=ax, xlim=(0,4), xlabel='Time (Ga)', ylabel=r'Mantle $T_p$ (K)', ls=lss[n])
    n=n+1 #this loop replaced: trajectories.plot(xlim=(0,4), ax=ax, xlabel='Time (Ga)', ylabel=r'Mantle $T_p$ (K)')
plot_gaussian_target(ax)
plt.legend(loc='upper left')
plt.savefig(targetfolder+'/C03_beta_single_comparison.pdf', bbox_inches='tight', facecolor='w', edgecolor='w')
plt.savefig(targetfolder+'/C03_beta_single_comparison.png', bbox_inches='tight', facecolor='w', edgecolor='w', dpi=200)


# ### Compare possible scaling-law transitions

# In[ ]:


curve = 'C03' #'C03' or 'RK18'
gridtype = 'Odds Ratio' #r'Z$_{RMSE}$' or 'Odds Ratio'
C03 = pd.read_csv('OUTPUT/'+curve+'_reduced.csv', header=0, index_col=0).sort_values(by=['b1', 'b0'], ascending=False) 
C03groups = C03.groupby(by='chgt')
rowcols = sorted(C03.b1.unique())

n=0
nxn=3.5
tol=1.0e-15
panels='abcd'
subplotno = [[0,0],[0,1],
             [1,0],[1,1]]
ticks = np.arange(-0.15, 0.3+tol, 0.05)
ticklabels = pd.Series(ticks).round(2).astype(str)
textloc = dict(zip([0.5, 0.9, 1.3, 1.7], [[0.05,0.85], [0.05,0.85], [0.15, 0.85], [0.28, 0.85]]))
fig, ax = plt.subplots(2,2, sharex=False, sharey=False, figsize=(nxn,nxn), constrained_layout=True)
for i, data in C03groups:
    if i not in textloc.keys():
        continue
    else:
        pivoted = pd.pivot_table(data, values='ZRMSE', index='b0', columns='b1', fill_value=None, dropna=False).T
        pivoted = pivoted.reindex(index=rowcols, columns=rowcols, fill_value=np.nan).fillna(10)
        if 'dds' in gridtype:  #as in, 'odds ratio' or similar
            pivoted = pivoted.apply(stats.norm.pdf)/stats.norm.pdf(0)
            vmin, vmax, cmap = 0, 1, 'gist_stern_r'
        else: vmin, vmax, cmap = 0, 4, 'gist_stern'
        panel = ax[subplotno[n][0]][subplotno[n][1]]
        im = panel.pcolormesh(pivoted.values, snap=True, vmin=vmin, vmax=vmax, cmap=cmap, shading='flat')
        panel.set_aspect('equal')
        if subplotno[n][0]==0: 
            panel.set_xticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_xticklabels([])
        else: 
            panel.set_xticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_xticklabels(ticklabels, rotation=90)
            panel.set_xlabel(r'$\beta_{present}$')
        if subplotno[n][1]==1: 
            panel.set_yticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_yticklabels([])
        else:
            panel.set_yticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_yticklabels(ticklabels)
            panel.set_ylabel(r'$\beta_{past}$')
        panel.text(x=len(pivoted.columns)*0.9, y=len(pivoted.columns)*0.9, s=panels[n])
        panel.text(x=len(pivoted.columns)*textloc[i][0], y=len(pivoted.columns)*textloc[i][1], 
                   s=r'$t_{\Delta\beta}$='+str(i)+'Ga')
        n=n+1
cb_ax = fig.add_axes([1.02, 0.20, 0.05, 0.70])
plt.colorbar(im, ax=cb_ax, cax=cb_ax)
cb_ax.set_title(gridtype.replace(' ','\n'), fontsize=10, pad=6, loc='center')
plt.savefig(targetfolder+'/Grid '+gridtype+'_'+curve+'.pdf', bbox_inches='tight', facecolor='w', edgecolor='w')
plt.savefig(targetfolder+'/Grid '+gridtype+'_'+curve+'.png', bbox_inches='tight', facecolor='w', edgecolor='w', dpi=200)

