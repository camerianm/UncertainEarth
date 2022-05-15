targetfolder = 'OUTPUT'
import os
if not os.path.exists(targetfolder): 
    os.mkdir(targetfolder)
from calc import *
from scipy import interpolate
from scipy import stats
import pandas as pd
tol = 1.0e-15
import numpy as np

nruns = 100
b = -0.093 #RK18
model='RK18' #or 'C03'
tmax, timestep = 4.0, 0.005

startTs = pd.read_csv('CorrectedTpPresent.csv', header=0) #best-fit-to-Phanero present Tp's for each beta
estimate_start_Tp = interpolate.interp1d(x=startTs.b0, y=startTs.Tp_AbbottMatch) #(assuming RK18 curve)
GrowthCurves = interpolate_growth_curve(timestep, tmax)
GrowthCurves.index = np.round(GrowthCurves.index, 3)
pars = generate_parameters(nruns)
HPhistories = generate_HP_distributions(nruns, GrowthCurves)
HPmodel = HPhistories['midpoints'][model]
cases = {'Ea': pars['Ea'].dist, 'Qtot': pars['Qtot'].dist,
         'Qm': (pars['Qtot'].dist - HPhistories['instantaneous']['Crust'].iloc[0]).values,
         'Tp': estimate_start_Tp(b)+mu_sig(nruns,0,10).dist}

trajects = pd.DataFrame(columns=HPmodel.columns, index=HPmodel.index)
for i in list(range(nruns)):
    trajects[i] = fast_evolve_singlemodel_twobeta(b0=b, b1=b, chgt=5.0,
                  Qt=cases['Qm'][i], Ea=cases['Ea'][i], Tp=cases['Tp'][i],
                   HP=HPmodel[i], every=1)
interpolated_temps = interpolate_temps_at(df=trajects)
interpolated_Zs = Z_score_from_interpolation(interpolated_temps)
interpolated_odds = interpolated_Zs.apply(stats.norm.pdf)/stats.norm.pdf(0)
trajects_stats = (trajects.T.quantile(percentiles)).T
trajects_stats.to_csv(targetfolder+'/trajects_'+model+'_'+str(round(b,5))+'.csv')

print(model+' '+str(round(b,5))+' done!')
print('Z-score means:')
print(interpolated_Zs.median())
print('Odds ratio means:')
print(interpolated_odds.median())

