#!/usr/bin/env python
# coding: utf-8

# ### Import modules
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
nxn = 2.9
GrowthCurves = pd.read_csv('GrowthCurves.csv', header=0, index_col=0)
GrowthCurves.columns = GrowthCurves.columns.map(Curves)
GrowthCurves.plot(xlabel='Time (Ga)', ylabel='Fraction of Continental Crust', title=None,#'Crustal Growth Models', 
                  cmap='bwr_r', figsize=(nxn, nxn), xlim=(0, GrowthCurves.index[-1]), ylim=0, legend=True)
plt.xlim(0,4.5)
plt.tight_layout()
plt.savefig('OUTPUT/GrowthModels.pdf')


# ### Mantle heat-production implications from crustal growth
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
trajectories = pd.read_csv('OUTPUT/Sample_Trajectories_reduced.csv', header=0, index_col=0)
trajectories.columns = trajectories.columns.str.replace('Beta',  'β')
fig, ax = plt.subplots(figsize=(2*nxn, nxn))
trajectories.plot(xlim=(0,4), ax=ax, xlabel='Time (Ga)', ylabel=r'Mantle $T_p$ (K)')
plot_gaussian_target(ax)
plt.tight_layout()
plt.savefig('OUTPUT/C03_beta_single_comparison.pdf')


# ### Compare possible scaling-law transitions
RK18 = pd.read_csv('OUTPUT/RK18_reduced.csv', header=0, index_col=0).sort_values(by=['b1', 'b0'], ascending=False)
RK18groups = RK18.groupby(by='chgt')
fig, ax = plt.subplots(2,4, figsize=(nxn*2.1,nxn*1.2))
xt = np.linspace(-0.15, 0.3, 10)
panelno = ['ABCD','EFGH']
n, row = 0, 0
for i, group in RK18groups:
    if i in [0.5, 0.8, 1.1, 1.4]:
            if n>0:
                xticks = xt[1:]
            else: xticks=xt
            group.plot(kind='scatter', s=5, marker='s', edgecolors='none', x='b0', y='b1', sharex=True, sharey=True,
                       c='ZRMSE', xticks=xticks, xlabel=None, ylabel=r'Past $\beta$', ax=ax[row][n],
                       colorbar=False, yticks=xt, rot=0, cmap='gist_stern', vmin=0, vmax=4, xlim=(xt[0], xt[-1]), ylim=(xt[0], xt[-1]))
            ax[row][n].set_aspect('equal')
            ax[row][n].set_title(str(i)+' Ga', pad=1)
            ax[row][n].text(s=panelno[row][n], y=0.22, x=0.22)
            n=n+1
C03 = pd.read_csv('OUTPUT/C03_reduced.csv', header=0, index_col=0).sort_values(by=['b1', 'b0'], ascending=False)
C03groups = C03.groupby(by='chgt')                
n, row = 0, 1
for i, group in C03groups:
    if i in [0.5, 0.8, 1.1, 1.4]:
            if n>0:
                xticks = xt[1:]
            else: xticks=xt
            group.plot(kind='scatter', s=5, marker='s', edgecolors='none', x='b0', y='b1', sharex=True, sharey=True,
                       c='ZRMSE', xticks=xticks, xlabel=r'Present $\beta$', ylabel=r'Past $\beta$', ax=ax[row][n],
                       colorbar=False, yticks=xt[:-1], rot=90, cmap='gist_stern', vmin=0, vmax=4, xlim=(xt[0], xt[-1]), ylim=(xt[0], xt[-1]))
            ax[row][n].set_aspect('equal')
            ax[row][n].text(s=panelno[row][n], y=0.22, x=0.22)
            n=n+1            
plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig('OUTPUT/RK18_C03_ChangeTime_Comparison.pdf')


# ### And the color bar for this:
fig, ax = plt.subplots(figsize=(1,2))
norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
cmap = matplotlib.cm.gist_stern
cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
cb1.set_label('RMSE Z-score')
plt.tight_layout()
plt.savefig('OUTPUT/RK18_C03_ChangeTime_Comparison_colorbar.pdf')

