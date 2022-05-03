#!/usr/bin/env python
# coding: utf-8

# ### Import  modules, make folder

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
plt.rcParams['font.size'] = 10.0
plt.rcParams['font.sans-serif'] = 'Arial'


# ### Single $\beta$ case: common parameters

# In[ ]:


nruns = 500
b = -0.07
startTs = pd.read_csv('CorrectedTpPresent.csv', header=0) #best-fit-to-Phanero present Tp's for each beta
estimate_start_Tp = interpolate.interp1d(x=startTs.b0, y=startTs.Tp_AbbottMatch) #(assuming RK18 curve)
timestep, GrowthCurves, times, timerange = generate_time_evolution(4.0)
tmax = timerange[-1]
HPE_budgets1 = strict_median_heat_budget(nruns, timestep, [0.0, tmax]) #or replace w/:generate_HPE_budgets
HPE_budgets2 = generate_HPE_budgets(nruns, timestep, [0.0, tmax]) #or replace w/:generate_HPE_budgets


# #### Sensitivity test part I: statistical variation in $E_A$, $T_p$, $Q_{total}$ without $H(t)$ variation

# In[ ]:


HPE_budgets = HPE_budgets1
pars = generate_parameters(nruns)
mantle_HP = {}
for curvename, crustfrac in GrowthCurves.items():
    mantle_HP[curvename] = HPE_budgets[3] - (HPE_budgets[4].mul(crustfrac, axis=0).dropna(how='all'))
to_vary = ['Ea', 'Qtot', 'Tp']
scenarios = {}
for param in to_vary:
    vary_me = [param]# ['Ea', 'Tp', 'Qtot']
    cases = {'Tp_uncert': nruns * [0]}
    for name, par in pars.items():
        if name in vary_me: cases[name] = par.dist
        else: cases[name] = nruns*[par.mu]
    if 'Tp' in vary_me: cases['Tp_uncert'] = mu_sig(nruns, 0, Abbott(3)['Tp_phanero'].sig).dist
    cases['Qm'] = pd.Series(cases['Qtot']) - HPE_budgets[4].iloc[0]
    cases = pd.DataFrame(cases).drop_duplicates()
    cases['Tp'] = estimate_start_Tp(b) + cases['Tp_uncert']
    scenarios[param] = evolve_model_onebeta(b, mantle_HP, cases)
    print(param, ' done')


# #### Sensitivity test part II: statistical variation in $H(t)$

# In[ ]:


HPE_budgets = HPE_budgets2
mantle_HP = {}
for curvename, crustfrac in GrowthCurves.items():
    mantle_HP[curvename] = HPE_budgets[3] - (HPE_budgets[4].mul(crustfrac, axis=0).dropna(how='all'))
to_vary = ['Ht']
for param in to_vary:
    vary_me = [param]# ['Ea', 'Tp', 'Qtot']
    cases = {'Tp_uncert': nruns * [0]}
    for name, par in generate_parameters(nruns).items():
        if name in vary_me: cases[name] = par.dist
        else: cases[name] = nruns*[par.mu]
    if 'Tp' in vary_me: cases['Tp_uncert'] = mu_sig(nruns, 0, Abbott(3)['Tp_phanero'].sig).dist
    cases['Qm'] = pd.Series(cases['Qtot']) - HPE_budgets[4].iloc[0]
    cases = pd.DataFrame(cases).drop_duplicates()
    cases['Tp'] = estimate_start_Tp(b) + cases['Tp_uncert']
    scenarios[param] = evolve_model_onebeta(b, mantle_HP, cases)
print(param, ' done')


# ### Get summary statistics for these

# In[ ]:


growthcurve = 'C03'
summary_stats = {}
nxn=2.5
a=0.2
npanels = len(scenarios.keys())
for name, curves in scenarios.items():
    summary_stats[name] = curves[growthcurve].T.quantile(percentiles).T #scenarios[name]['C03_stats']
    summary_stats[name].to_csv(targetfolder+'/'+growthcurve+'_'+name+'.csv')


# ### Plot RMSE Z-scores against their trajectory ensembles
# #### Plot with axis labels

# In[ ]:


nxn = 2.1
fig, ax = plt.subplots(2, npanels, figsize=(nxn*0.9*npanels, 2*nxn))
n=0
c='b'
Zs = {}
boxno = 'ABCD'
for name, s in summary_stats.items():
    if n==0: 
        ylabel=r'Mantle $T_p$ (K)'
    else: 
        ylabel=None
        ax[0][n].set_yticklabels([])
    ax[0][n].fill_between(y1=s[s.columns[0]], y2=s[s.columns[-1]], x=s.index, ec='b', color=c, alpha=a)
    ax[0][n].fill_between(y1=s[s.columns[1]], y2=s[s.columns[-2]], x=s.index, ec='b', color=c, alpha=a)
    s[s.columns[2]].plot(c=c, label=None, ax=ax[0][n], lw=1, xticks=[0,1,2,3,4],
                         xlabel='Time (Ga)', ylabel=ylabel)
    ax[0][n].text(s=boxno[n], y=2040, x=0.2)
    #ax[0][n].xaxis.tick_top()
    ax[0][n].xaxis.set_label_position('bottom')
    plot_gaussian_target(ax[0][n])
    n=n+1

n=0
boxno='EFGH'
for i, thing in scenarios.items():
    if n==0:
        ylabel='Kernel Density'
    else: 
        ylabel=None
        ax[1][n].set_yticklabels([])
    Zs[i] = dict()
    str_i = r'$'+i[0]+'_{'+i[1:]+'}$'
    for curve, df in thing.items():
        Zs[i][curve] = Z_scores(df)
        if curve==growthcurve:
            Zs[i][curve]['RMSE'].plot.kde(ax=ax[1][n], xlim=(0,3), ylim=(0,5), label=i+' '+curve, c=c)
            ax[1][n].set_xlabel('Z-score')
            ax[1][n].set_ylabel(ylabel)
            #ax[n].text(s=str_i, x=2.3, y=-1.75) #.25)
            ax[1][n].text(s=boxno[n], x=0.15, y=3.2)
            n=n+1
#plt.legend(loc='right', bbox_to_anchor=(1.9,0.5))
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.tight_layout()
plt.savefig(targetfolder+'/'+growthcurve+'scenarios_test_scores_COMBINED_RMSE_Ea_Qtot_Tp_Ht.pdf')
        #k.loc[k.index[0::5]].to_csv('OUTPUT/'+curve+'_'+i+'.csv')


# #### Same plots, with X-axis labels removed

# In[ ]:


nxn = 2.1
c='b'
fig, ax = plt.subplots(2, npanels, figsize=(nxn*0.9*npanels, 2*nxn))
n=0
boxno = 'ABCD'
for name, s in summary_stats.items():
    if n==0: 
        ylabel=r'Mantle $T_p$ (K)'
    else: 
        ylabel=None
        ax[0][n].set_yticklabels([])
    ax[0][n].fill_between(y1=s[s.columns[0]], y2=s[s.columns[-1]], x=s.index, ec=c, color=c, alpha=a)
    ax[0][n].fill_between(y1=s[s.columns[1]], y2=s[s.columns[-2]], x=s.index, ec=c, color=c, alpha=a)
    s[s.columns[2]].plot(c=c, label=None, ax=ax[0][n], lw=1, xticks=[0,1,2,3,4],
                         xlabel=None, ylabel=ylabel)
    ax[0][n].text(s=boxno[n], y=2040, x=0.2)
    ax[0][n].xaxis.set_label_position('bottom')
    plot_gaussian_target(ax[0][n])
    n=n+1

n=0
boxno='EFGH'
for i, thing in scenarios.items():
    if n==0:
        ylabel='Kernel Density'
    else: 
        ylabel=None
        ax[1][n].set_yticklabels([])
    Zs[i] = dict()
    str_i = r'$'+i[0]+'_{'+i[1:]+'}$'
    for curve, df in thing.items():
        Zs[i][curve] = Z_scores(df)
        if curve=='C03':
            Zs[i][curve]['RMSE'].plot.kde(ax=ax[1][n], xlim=(0,3), ylim=(0,5), label=i+' '+curve, c=c)
            ax[1][n].set_ylabel(ylabel)
            ax[1][n].text(s=boxno[n], x=0.15, y=3.2)
            n=n+1
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.tight_layout()
plt.savefig(targetfolder+'/'+growthcurve+'_scenarios_test_scores_noxlabels_COMBINED_RMSE_Ea_Qtot_Tp_Ht.pdf')
        #k.loc[k.index[0::5]].to_csv('OUTPUT/'+curve+'_'+i+'.csv')


# ### Explore role of variable $\beta$ given average parameters otherwise

# In[ ]:


tol=1.0e-15
betas = np.round(np.arange(-0.15, 0.3+tol, 0.01), 2)
betas = ([[i, j] for i in betas for j in betas])
b0s, b1s = np.array(betas).transpose()
Tps = estimate_start_Tp(b0s)
chgts = np.round(np.arange(0.4, 2.0+tol, 0.2), 2)
scenarios = pd.DataFrame({'b0': b0s, 'b1': b1s, 'Tp': Tps})
results = dict(zip(chgts, len(chgts)*[None]))
every = 5
Ea, Qt = pars['Ea'].mu, pars['Qtot'].mu - HPE_budgets1[4].iat[0,0]
crustfrac = GrowthCurves[growthcurve]
HP = (HPE_budgets1[3] - (HPE_budgets1[4].mul(crustfrac, axis=0)).dropna()).T.drop_duplicates().T[0]
summary = dict()
for chgt in results.keys():
    results[chgt] = dict()
    for n,p in scenarios.T.items():
        results[chgt][n] = fast_evolve_singlemodel_twobeta(b0=p.b0, b1=p.b1, chgt=chgt, 
                            Tp=p.Tp, Ea=Ea, Qt=Qt, HP=HP, every=every)
    results[chgt] = pd.DataFrame(results[chgt])
    interp = interpolate_temps_at(df=results[chgt])
    Zs = Z_score_from_interpolation(interp)
    summary[chgt] = pd.concat([scenarios, interp, Zs], axis=1)
    summary[chgt] = summary[chgt].replace(np.nan, np.inf)
    summary[chgt].to_csv(targetfolder+'/'+str(chgt)+growthcurve+'b0b1.csv')


# ### Plot these cases' fit with respect to Abbott et al 1994

# In[ ]:


subplotno = [[0,0],[0,1],#[0,2],
             [1,0],[1,1]]#,#[1,2],
             #[2,0],[2,1],[2,2]]
n=0
panels='ABCDEFGHI'
ticks = np.arange(-0.15, 0.3+tol, 0.05)
ticklabels = pd.Series(ticks).round(2).astype(str)
fig, ax = plt.subplots(2,2, sharex=False, sharey=False, figsize=(3,3))
for i in summary.keys():
    if i not in [0.4, 0.8, 1.2, 1.6]:
        continue
    else:
        data = summary[i]
        pivoted = pd.pivot_table(data, values='Z_{RMSE}', index='b0', columns='b1', fill_value=None, dropna=False).T
        pivoted = pivoted.mask(pivoted>100, other=100, inplace=False)
        panel = ax[subplotno[n][0]][subplotno[n][1]]
        im = panel.pcolormesh(pivoted.values, snap=True, vmin=0, vmax=4, cmap='gist_stern', shading='flat')
        panel.set_aspect('equal')
        if subplotno[n][0]==0: panel.set_xticks([])
        else: 
            panel.set_xticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_xticklabels(ticklabels, rotation=90)
        if subplotno[n][1]==1: panel.set_yticks([])
        else:
            panel.set_yticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_yticklabels(ticklabels)
        panel.text(x=len(pivoted.columns)*0.85, y=len(pivoted.columns)*0.85, s=panels[n]) #+' (t='+str(i)+')')
        panel.text(x=len(pivoted.columns)*0.45, y=len(pivoted.columns)*0.08, s=r'$t_{chg}$='+str(i))
        n=n+1
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.15)
cb_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
cbar = fig.colorbar(im, cax=cb_ax)

plt.savefig(targetfolder+'/'+growthcurve+'_stats.pdf')


# ### How does this map to odds ratio WRT ideal case?

# In[ ]:


subplotno = [[0,0],[0,1],#[0,2],
             [1,0],[1,1]]#,#[1,2],
             #[2,0],[2,1],[2,2]]
n=0
panels='ABCDEFGHI'
ticks = np.arange(-0.15, 0.3+tol, 0.05)
ticklabels = pd.Series(ticks).round(2).astype(str)
fig, ax = plt.subplots(2,2, sharex=False, sharey=False, figsize=(3,3))
for i in summary.keys():
    if i not in [0.4, 0.8, 1.2, 1.6]:
        continue
    else:
        data = summary[i]
        pivoted = pd.pivot_table(data, values='Z_{RMSE}', index='b0', columns='b1', fill_value=None, dropna=False).T
        pivoted = pivoted.mask(pivoted>100, other=100, inplace=False)
        odds = stats.norm.pdf(pivoted.values)/stats.norm.pdf(0)
        odds = pd.DataFrame(odds, columns=pivoted.index, index=pivoted.columns)
        panel = ax[subplotno[n][0]][subplotno[n][1]]
        im = panel.pcolormesh(odds.values, snap=True, vmin=0, vmax=1, cmap='gist_stern_r', shading='flat')
        panel.set_aspect('equal')
        if subplotno[n][0]==0: panel.set_xticks([])
        else: 
            panel.set_xticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_xticklabels(ticklabels, rotation=90)
        if subplotno[n][1]==1: panel.set_yticks([])
        else:
            panel.set_yticks(np.linspace(0,len(pivoted.index), 10))
            panel.set_yticklabels(ticklabels)
        panel.text(x=len(pivoted.columns)*0.85, y=len(pivoted.columns)*0.85, s=panels[n]) #+' (t='+str(i)+')')
        panel.text(x=len(pivoted.columns)*0.45, y=len(pivoted.columns)*0.08, s=r'$t_{chg}$='+str(i))
        n=n+1
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.15)
cb_ax = fig.add_axes([1.0, 0.05, 0.05, 0.9])
cbar = fig.colorbar(im, cax=cb_ax)
plt.savefig(targetfolder+'/'+growthcurve+'_odds_stats.pdf')

