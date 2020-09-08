import time
import calc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

nruns = 100
timestep=1.0e-3  # e.g. if 1.0e-2, open calc and set decimals=2
tmin, tmax = 0.0, 3.5
targetP=1.5
lowerrow=0
startpoint = 'midarc'
endpoint = 'phanero'
constants = ['Ea_list', 'betatransition', 'Qm_list', 'Tp_present', 'Tp_phanero', 'Tp_latearc', 'Tp_midarc', 't_midarc', 't_latearc', 't_phanero']
curves = ['Linear', 'Sqrt', 'Fifth', 'HalfBy2.0', 'HalfBy2.5', 'HalfBy3.0', 'HalfBy3.5']
print('Using only mean values for the following:\n', [i for i in constants])
cases = calc.case_defs(nruns=nruns, constants=constants)
print(cases.describe())
print('Generating crustal and bulk heat budgets.')
Isos, bse_budgets, cr_budgets, bse_Ht_max, crust_Ht_max = calc.HPE_budgets(nruns=nruns, timestep=timestep, trange=[tmin, tmax])
GrowthCurves = calc.growth_models(crust_Ht_max=crust_Ht_max, nruns=nruns, timestep=timestep).loc[list(crust_Ht_max.index), :]
UM, LM, WM = calc.PTXgrids()
Pslice = calc.slice_PTXgrid(grid=UM, targetP=targetP, lowerrow=lowerrow)
pz = pd.read_csv('PREM.csv', header=0, usecols=[8,9])
z_fP = interpolate.interp1d(pz['Pressure(GPa)'], pz['Depth(km)'])
dT_fTp = 0.0 #0.35 * z_fP(targetP)
rho_fT = interpolate.interp1d(Pslice['Tp'], Pslice['rho'])
alpha_fT = interpolate.interp1d(Pslice['Tp'], Pslice['alpha'])
Cp_fT = interpolate.interp1d(Pslice['Tp'], Pslice['Cp'])
cases['rho_ref'] = rho_fT(cases['Tp_present']+dT_fTp)
cases['Cp_ref'] = Cp_fT(cases['Tp_present']+dT_fTp)
cases['alpha_ref'] = alpha_fT(cases['Tp_present']+dT_fTp)
cases['deltaT_ref'] = cases['Tp_present']-dT_fTp-300.0
runs = cases.transpose()

for curve in curves:
	mantle_Ht_max = calc.impose_growth_models_on_mantle(models=[curve],
		GrowthCurves=GrowthCurves, bse_df=bse_Ht_max, crust_df=crust_Ht_max)
	Tlist = len(runs.columns)*['NaN']
	start = time.time()
	for r in runs:
		t_start = runs[r]['t_'+startpoint] #calc.round_up(runs[r]['t_'+startpoint])
		b=0.1
		if runs[r]['t_'+startpoint]<runs[r]['betatransition']:
			 b=0.3
		H_start = mantle_Ht_max[r].loc[calc.round_up(t_start, )]
		Tp = runs[r]['Tp_'+startpoint]
		factors = [#(rho_fT(Tp)/runs[r]['rho_ref'])**(2*b),
				#((Cp_fT(Tp) * alpha_fT(Tp)) / (runs[r]['Cp_ref'] * runs[r]['alpha_ref']))**(b),
				((Tp-dT_fTp-300.0) / (runs[r]['deltaT_ref']))**(b+1),
				np.exp(((runs[r]['Ea_list'])/calc.R_idealgas)*((1/Tp)-(1/(runs[r]['Tp_present'] + dT_fTp))))**(-1*b)]
		Q_start = runs[r]['Qm_list'] * np.prod(factors)
		dT = -1 * ((t_start - calc.round_down(t_start))*calc.seconds*1.0e12*(Q_start-H_start)/(Cp_fT(Tp)*calc.M_mant+calc.Cpcore))
		Hts = mantle_Ht_max[r].loc[calc.round_down(t_start):calc.round_up(runs[r]['t_latearc']):-1]
		for t in Hts.index:
			if t>runs[r]['t_'+endpoint]:
				Tp = Tp + timestep*dT
				if t<runs[r]['betatransition']:
					b=0.3
				factors = [#(rho_fT(Tp)/runs[r]['rho_ref'])**(2*b),
							#((Cp_fT(Tp) * alpha_fT(Tp)) / (runs[r]['Cp_ref'] * runs[r]['alpha_ref']))**(b),
							((Tp-dT_fTp-300.0) / (runs[r]['deltaT_ref']))**(b+1),
							np.exp(((runs[r]['Ea_list'])/calc.R_idealgas)*((1/Tp)-(1/(runs[r]['Tp_present'] + dT_fTp))))**(-1*b)]
				Qt = runs[r]['Qm_list'] * np.prod(factors)
				dT = -1 * (calc.seconds*1.0e12*(Qt-Hts[t])/(Cp_fT(Tp)*calc.M_mant+calc.Cpcore))
		Tlist[r]=Tp+dT*(calc.round_up(runs[r]['t_'+endpoint]) - runs[r]['t_'+endpoint])
	cases['Tend'] = Tlist
	print(curve, '\n', calc.CI_cols(cases)['Tend']/calc.CI_cols(cases)['Tp_'+endpoint])
	#print(curve, '\t', time.time()-start, 's')
