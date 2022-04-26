targetfolder = '0425'
import os
if not os.path.exists(targetfolder): 
    os.mkdir(targetfolder)
from calc import *
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

nruns = 10
startTs = pd.read_csv('CorrectedTpPresent.csv', header=0) #best-fit-to-Phanero present Tp's for each beta
estimate_start_Tp = interpolate.interp1d(x=startTs.b0, y=startTs.Tp_AbbottMatch) #(assuming RK18 curve)
timestep, GrowthCurves, times, timerange = generate_time_evolution(4.0)
tmax = timerange[-1]
HPE_budgets = strict_median_heat_budget(nruns, timestep, [0.0, tmax]) #or replace w/:generate_HPE_budgets
mantle_HP = {}
for curvename, crustfrac in GrowthCurves.items():
    mantle_HP[curvename] = HPE_budgets[3] - (HPE_budgets[4].mul(crustfrac, axis=0).dropna(how='all'))

vary_me = ['Ea', 'Tp', 'Qtot']
cases = {'Tp_uncert': nruns * [0]}
for name, par in generate_parameters(nruns).items():
    if name in vary_me: cases[name] = par.dist
    else: cases[name] = nruns*[par.mu]
if 'Tp' in vary_me: cases['Tp_uncert'] = mu_sig(nruns, 0, Abbott(3)['Tp_phanero'].sig).dist
cases['Qm'] = pd.Series(cases['Qtot']) - HPE_budgets[4].iloc[0]
cases = pd.DataFrame(cases).drop_duplicates()

b = -0.1
cases['Tp'] = estimate_start_Tp(b) + cases['Tp_uncert']
batches = evolve_model_onebeta(b, mantle_HP, cases)

#nxn = 2.9 #default matplotlib plot size
#fig, ax = plt.subplots(figsize=(nxn,nxn))
#[batches[i].T.mean().plot(legend=False, ax=ax) for i in batches]
#plot_gaussian_target(ax=ax)
#plt.tight_layout()
#plt.savefig(targetfolder+'/test.pdf')
print('done!')