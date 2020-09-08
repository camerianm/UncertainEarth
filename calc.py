import pandas as pd
import numpy as np
import math
#import scipy.stats as stats
#from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib import rcParams as rcParams
rcParams['figure.dpi'] = 200

decimals = 3
tmin, tmax = 2.5, 3.9
Tpmin, Tpmax = 1600, 2100
R_idealgas = 8.314
M_mant = 4.043e24
M_crus = 2.35e22
seconds = 3.1536e16
Cpcore = 7.0e27 - M_mant*1250.0
percentiles=[0.023, 0.159, 0.5, 0.841, 0.977]
mins = ['C2/c,wt%', 'Wus,wt%', 'Pv,wt%', 'Sp,wt%', 'O,wt%', 'Wad,wt%', 'Ring,wt%', 'Opx,wt%', 'Cpx,wt%', 'Aki,wt%', 'Gt_maj,wt%', 'Ppv,wt%', 'CF,wt%', 'st,wt%', 'ca-pv,wt%']
percentilenames=['2.3%', '15.9%', '50%', '84.1%', '97.7%']
R_idealgas = 8.314
minT, maxT = 1600, 2100
tmin, tmax = 2.5, 3.9
cloud_alpha = lambda nruns: 1./np.log(nruns)

def round_up(n, decimals=decimals):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
def round_down(n, decimals=decimals):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

# Setting up RE: Q(T)

def case_defs(nruns: int, constants: list) -> pd.DataFrame:
    """
    Establishes empirically-derived uncertainties to accommodate/explore.
    :nruns: integer
    :return: pandas DataFrame
    """
    cases = pd.DataFrame({'Tp_present': list(np.random.normal(1610., 35.0, nruns)),  # Method: Katsura et al 2010
                         'Tp_phanero' : list(np.random.normal(1653., 10.0, nruns)),  # Method: Abbott et al 1994
                         'Tp_latearc': list(np.random.normal(1840.15, 18.0, nruns)), # Method: Abbott et al 1994
                         'Tp_midarc': list(np.random.normal(1891., 55.0, nruns)),    # Method: Abbott et al 1994
                         't_phanero': list(np.random.normal(0.302, 0.033, nruns)),   # Method: Abbott et al 1994
                         't_latearc' : list(np.random.normal(2.756, 0.028, nruns)),  # Method: Abbott et al 1994
                         't_midarc': list(np.random.normal(3.344, 0.114, nruns)),    # Method: Abbott et al 1994
                         'Ea_list': list(np.random.normal(362.0e3 , 60.0e3, nruns)), # Method: Jain 2018? or Korenaga 08?
                         'Qm_list' : list(np.random.uniform(36.5, 4.5, nruns)),  # Method: Korenaga 2008, mantle heat flux DOI:10.1029/2007RG000241
                         #'Cw_list' : list(np.random.normal(1.05, 0.02, nruns)), # Water content ratio, Archean/Present
                         #'pCw_list' : list(np.random.normal(0.84, 0.26, nruns)), # Water content exponent + uncertainty
                         #'Gs_list' : list(np.random.normal(2, 1, nruns)),  # Grain size ratio Archean/Present - span from se.copernicus.org/articles/11/959/2020/
                         #'pGs_list' : list(np.random.normal(1.74, 0.12, nruns)), #Grain size exponent + uncertainty
                         #'act_vol': list((0.01**3) * pd.Series(np.random.normal(6.75, 13.23, nruns))) #6.75, 13.23, nruns)))
                          'betatransition' : list(np.random.uniform(2.3, 3.5, nruns)),
                         })
    for i in constants:
        cases[i] = len(cases.index) * [cases[i].mean()]
    return(cases)

def PTXgrids() -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Accesses and filters ExoPlex-derived P-T-X grids for Earthlike composition.
	Inputs for grid were from Unterborn et al. 2016, doi.org/10.3847/0004-637X/819/1/32
	:return: list of 3 pandas DataFrames - upper-mantle, lower-mantle, whole-mantle
	"""
	df = pd.read_csv('STE_UM_LM.csv', header=0, names=['gridID', 'UpperLower', 'Tp', 'Pbar', 'P', 'rho', 'v0', 'vp', 'vs', 'alpha', 'Cp', 
														'C2/c,wt%', 'Wus,wt%', 'Pv,wt%', 'Sp,wt%', 'O,wt%', 'Wad,wt%', 'Ring,wt%', 'Opx,wt%', 
														'Cpx,wt%', 'Aki,wt%', 'Gt_maj,wt%', 'Ppv,wt%', 'CF,wt%', 'st,wt%', 'ca-pv,wt%'])
	df = df[(df['alpha']>0) & (df['Cp']>0)] #Remove unphysical values - only a few, at very high T
	df = df.drop([i for i in mins if df[i].sum()<0.0001], axis=1) #[[i for i in df if ('%' in i) and (df[i].sum()>0)]]
	WM = df
	LM, UM = [x for _, x in df.groupby('UpperLower')] # Separate upper and lower mantle grids
	fields = ['alpha', 'Cp', 'rho'] # What we want to read in
	UM = UM.drop(['UpperLower'], axis=1)
	LM = LM.drop(['UpperLower'], axis=1)
	return(UM, LM, WM)

def slice_PTXgrid(grid: pd.DataFrame, targetP=1.0, lowerrow=0) -> pd.DataFrame:
	"""
	Slices upper or lower mantle P-T-X grid at the relevant reference pressure using linear interpolation.
	:grid: pandas DataFrame, either UM or LM - whichever inludes the reference pressure and its highest expected temperature
	:targetP: float, default 1.0 (GPa)
	:lowerrow: integer, default 0 - lowerrow and upperrow (not specified) would bound targetP in grid
	:return: pandas DataFrame
	"""
	upperrow = lowerrow + 1
	df=grid
	lower, upper = df['P'].unique()[lowerrow], df['P'].unique()[upperrow]
	if (upper<targetP) or (lower>targetP):  # Help user ID the rows between which targetP would hypothetically be found
		print('Is targetP ', targetP, 'above ', lower, ' and below', upper, '?')
		print('If not, adjust to approx rows in DataFrame. Ballpark:')
		for i,j in enumerate(df.P.unique()):
			if (i % 5) == 0:
				print(i, j)
	fraclower = (targetP - upper)/(lower - upper)
	lowP = pd.DataFrame(fraclower*df[df['P']==lower]).reset_index(drop=True)
	highP = pd.DataFrame((1-fraclower)*df[df['P']==upper]).reset_index(drop=True)
	Pslice = (highP+lowP).dropna()
	minsum = Pslice[mins].transpose().sum()
	for i in mins:
		Pslice[i] = 100.0 * Pslice[i]/minsum
	Pslice = Pslice[[i for i in Pslice if Pslice[i].sum()>0]] # remove minerals irrelevant for pressure or chemistry
	return(Pslice)

def superimpose_Archean_temperatures(cases: pd.DataFrame):
    """
    Add paleo-temperature distributions to the current time vs temperature plot object
    """
    plt.scatter(x=cases['t_latearc'], y=cases['Tp_latearc'], c='coral', alpha=(1./np.log(0.5*len(cases['t_latearc']))), label='Late Archean')
    plt.scatter(x=cases['t_midarc'], y=cases['Tp_midarc'], c='maroon', alpha=(1./np.log(0.5*len(cases['t_latearc']))), label='Mid Archean')
    plt.xlabel('Age (Ga)', fontsize=15)
    plt.ylabel('$T_p$ (K)', fontsize=15)
    plt.title('Mid- and Late-Archean mantle thermal states', fontsize=16)
    plt.ylim(Tpmin, Tpmax)
    plt.xlim(tmin, tmax)
    return

def geotherms() -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Import upper-mantle solidus, liquidus, and upper+lower bounds of 1623K adiabat, for sanity check.
	Solidus_Katz.csv and Liquidus_Katz.csv from Katz et al 2003, doi.org/10.1029/2002GC000433
	1673Adiabat.csv and 1573Adiabat.csv from Rohrbach & Schmidt 2011, doi.org/10.1038/nature09899
	Values are approximate; obtained from plots using automeris.io/WebPlotDigitizer/
	:return: list of 4 pandas DataFrames: solidus, liquidus, high_adiabat, low_adiabat
	"""
	solidus = pd.read_csv('Solidus_Katz.csv', header=0)
	liquidus = pd.read_csv('Liquidus_Katz.csv', header=0)
	high_adiabat = pd.read_csv('1673Adiabat.csv', header=0)
	low_adiabat = pd.read_csv('1573Adiabat.csv', header=0)
	return(solidus, liquidus, high_adiabat, low_adiabat)

def linear_geotherm(Tp=1625., gradient=0.35, maxP=20.0):
	"""
	Calculate temperature vs pressure according to linear geotherm - last item is at maxP
	"""
	geotherm = pd.DataFrame({'TemperatureK': Tp+np.linspace(0.1, maxP, 50)*(gradient*100/3),
		'PressureGPa': np.linspace(0.1, maxP, 50)})
	return(geotherm)

def geotherm_plot(maxP=20.0, contours=False, WM=None, Tp=1625.0, gradient=0.35):
	"""
	Plot geotherm with respect to relevant properties from PerPlex output
	"""
	solidus, liquidus, high_adiabat, low_adiabat = geotherms()
	linear_adiabat = linear_geotherm(Tp, gradient, maxP)
	if (WM==None):
		UM, LM, WM = PTXgrids()  # allows user to selectively replace if using different grid.
	CI_to_show=[0.023, 0.977]
	fields = ['alpha', 'Cp', 'rho']
	df=WM
	for index, value in enumerate(fields):
	    Z = df.pivot_table(index='Tp', columns='P', values=value).T.values
	    X_unique = np.sort(df['Tp'].unique())
	    Y_unique = np.sort(df['P'].unique())
	    X, Y = np.meshgrid(X_unique, Y_unique)
	    plt.pcolormesh(X, Y, Z, shading='gouraud', cmap='rainbow', 
	                   vmin=df[df['P']<maxP][value].describe(percentiles=CI_to_show)[[4]],
	                   vmax=df[df['P']<maxP][value].describe(percentiles=CI_to_show)[[6]])
	    plt.ylabel('Pressure (GPa)')
	    plt.xlabel('Temperature (K)')
	    plt.title(value)
	    plt.gca().invert_yaxis()
	    plt.colorbar(extend='both').set_label(value)
	    plt.tight_layout()
	    plt.plot(solidus['TemperatureK'], solidus['PressureGPa'], c='k', lw=3, label='Solidus')
	    plt.plot(liquidus['TemperatureK'], liquidus['PressureGPa'], c='orange', lw=3, label='Liquidus')
	    plt.plot(high_adiabat['TemperatureK'], high_adiabat['PressureGPa'], c='white', lw=3, alpha=0.8, label='1573K/1300C')
	    plt.plot(low_adiabat['TemperatureK'], low_adiabat['PressureGPa'], c='white', lw=4, alpha=0.8, label='1673K/1400C')
	    plt.plot(linear_adiabat['TemperatureK'], linear_adiabat['PressureGPa'],
	             label=str(gradient)+'K/km', c='k', lw=2, ls='--')
	    plt.legend(loc="lower right")
	    if maxP>136:
	        plt.yscale('log')
	        plt.legend(loc="upper right")
	    if contours==True:
	        levels=list(df[value].describe(percentiles=percentiles)[4:9])
	        plot = plt.contour(X,Y,Z,levels,colors='k')
	        plt.setp(plot.collections, path_effects=[pe.withStroke(linewidth=4, foreground="white")])
	        plt.clabel(plot,fmt='%f',fontsize=12.0)
	    plt.xlim(1400, 2500)
	    plt.ylim(maxP, 0)
	    plt.show()
	return()

# Setting up RE: H(t)

def HPE_budgets(nruns : int, timestep : float, trange : list):
    Isos = pd.read_csv('HPEs.csv', header=0, index_col=0)
    bse_budgets, cr_budgets = pd.DataFrame(), pd.DataFrame()
    for i in Isos.index:
        bse_budgets[i] = pd.Series(np.random.normal(Isos['Xbse_Elem_Mean'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_mant+M_crus),
                                Isos['Xbse_Elem_Std'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_mant+M_crus), nruns)) * 1.0e-12
        cr_budgets[i] = pd.Series(np.random.normal(Isos['Xcr_Elem_Mean'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_crus),
                                Isos['Xcr_Elem_Std'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_crus), nruns)) * 1.0e-12
    bse_budgets, cr_budgets = bse_budgets.transpose(), cr_budgets.transpose()  # Each column is a present-day starting case, in TW.
    times = np.arange(trange[0], trange[1]+timestep, timestep).round(decimals)
    df_t = pd.DataFrame(nruns * [list(times)], columns=(times))
    bse_Ht_max = pd.DataFrame(0.0, columns=df_t.columns, index=df_t.index)
    cr_Ht_max = pd.DataFrame(0.0, columns=df_t.columns, index=df_t.index)
    for isotope in Isos.index:
        m=((np.exp(Isos['Lambda'][isotope]*df_t)).mul(bse_budgets.loc[isotope], axis=0))
        c=((np.exp(Isos['Lambda'][isotope]*df_t)).mul(cr_budgets.loc[isotope], axis=0))
        bse_Ht_max = bse_Ht_max.add(m, fill_value=0.0)
        cr_Ht_max = cr_Ht_max.add(c, fill_value=0.0)
    return(Isos, bse_budgets, cr_budgets, bse_Ht_max.transpose(), cr_Ht_max.transpose())

def growth_models(crust_Ht_max : pd.DataFrame, nruns : int, timestep : float):
    times = np.arange(0, 4.5+timestep, timestep).round(decimals)
    timefrac = pd.Series(1.00-(times/times.max()), index=times)
    GrowthCurves = pd.DataFrame({'Linear': timefrac,
                                 'Sqrt': timefrac ** 0.5,
                                'Fifth': timefrac ** 0.2,
                                'HalfBy2.0': 1 - 1/(1+np.exp(4*(2.0-times))),
                                'HalfBy2.5': 1 - 1/(1+np.exp(4*(2.5-times))),
                                'HalfBy3.0': 1 - 1/(1+np.exp(4*(3.0-times))),
                                'HalfBy3.5': 1 - 1/(1+np.exp(4*(3.5-times)))},
                                index = times)
    return(GrowthCurves)

def stat_sample_growth_models(bse_Ht_max=None, crust_Ht_max=None, GrowthCurves=None, nruns=100, timestep=0.001, showplot=True):
    if (type(bse_Ht_max)==None) or (type(crust_Ht_max)==None):
        temp1, temp2, temp3, bse_Ht_max, crust_Ht_max = calc.HPE_budgets(nruns=nruns, timestep=0.1)
    if (type(GrowthCurves)==None):
        GrowthCurves = growth_models(nruns=nruns, timestep=timestep)
    bse_sample = bse_Ht_max.transpose().describe(percentiles=percentiles).transpose().drop(
        ['count', 'mean', 'std', 'min', 'max'], axis=1)  # Get statistics for each time snapshot
    crust_sample = crust_Ht_max.transpose().describe(percentiles=percentiles).transpose().drop(
        ['count', 'mean', 'std', 'min', 'max'], axis=1)  # Get statistics for each time snapshot# Get statistics for each time snapshot
    if showplot==True:
        GrowthCurves.plot(cmap='rainbow', title='Proposed continental growth curves')
        bse_sample.plot(cmap='rainbow', title='Max Bulk Silicate Earth heat production')
        crust_sample.plot(cmap='rainbow', title='Max Continental Crust heat production')
    plt.show()
    return(bse_sample, crust_sample)

def plot_archean_crust_uncert(GrowthCurves: pd.DataFrame, crust_sample: pd.DataFrame, cases=None):
    assign_colors = cm.get_cmap('rainbow', len(GrowthCurves.columns))
    my_cm = dict(zip(list(GrowthCurves.columns), np.linspace(0, 1, len(GrowthCurves.columns))))
    for curve in GrowthCurves:
        showplot=plt.plot(GrowthCurves.index, crust_sample.mul(GrowthCurves[curve], axis=0), 
                           c=assign_colors(my_cm[curve]), label=curve)
    mylegend = {y:x for x,y in dict(zip(plt.gca().get_legend_handles_labels()[1], plt.gca().get_legend_handles_labels()[0])).items()}
    plt.xlim(tmin, tmax)
    plt.xlabel('Age (Ga)')
    plt.ylabel('Q (TW)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=mylegend.values(), handles=mylegend.keys())
    plt.title('Archean crustal heat production, by growth model')
    plt.show()
    return(plt.show())

def impose_growth_models_on_mantle(GrowthCurves: pd.DataFrame, bse_df: pd.DataFrame, crust_df: pd.DataFrame, models=None):
    """
    Apply growth curve to crustal and BSE heat budgets.
    :bse_df: pandas DataFrame, either bse_sample or bse_Ht_max
    :crust_df: pandas DataFrame, either crust_sample or crust_Ht_max
    :models: for now, list, ONE of ['Linear', 'Sqrt', 'Fifth', 'HalfBy2.0', 'HalfBy2.5', 'HalfBy3.0', 'HalfBy3.5']
    :return: either mantle_Ht_max (DataFrame, distribution) or MantleHeat (derived from statistical summary)
    """
    MantleHeat = pd.DataFrame()
    mantle_Ht_max = pd.DataFrame()
    curve_snapshot = {}
    meancases, upperlower = [], []
    for curve in models:
        if bse_df.columns[1]!=1: # if statistical snapshot, not 500 scenarios
            for i in bse_df:
                for j in crust_df:
                    MantleHeat[curve+' BSE:'+i+', Cr:'+j] = bse_df[i] - crust_df[j] * GrowthCurves[curve]
            meancases.append(curve+' BSE:50%, Cr:50%')
            upperlower.append(curve+' BSE:97.7%, Cr:2.3%')
        else:
            for i in bse_df:
                mantle_Ht_max[i] = bse_df[i] - crust_df[i] * GrowthCurves[curve]
    if len(MantleHeat.columns)>0:
        output=MantleHeat
    else:
        output=mantle_Ht_max
    return(output)

def CI_rows(df: pd.DataFrame) -> pd.DataFrame:
    return(df.transpose().describe(percentiles=percentiles).transpose().drop(['count', 'mean', 'std', 'min', 'max'], axis=1))

def CI_cols(df: pd.DataFrame) -> pd.DataFrame:
    return(df.describe(percentiles=percentiles).drop(['count', 'mean', 'std', 'min', 'max'], axis=0))

'''
def plot_
showplot=True
if showplot==True:
    ax=MantleHeat[meancases].plot(colormap='rainbow', legend=True) #, title='Mantle HP, assuming MEAN abundances of HPEs')
    MantleHeat[upperlower].plot(colormap='rainbow', legend=True, ax=ax, ls='--') #title='Mantle HP, assuming MEAN abundances of HPEs')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(cases.t_latearc.min(), cases.t_midarc.max())
    ax2 = MantleHeat.plot(c='k', legend=False, alpha=0.1, label='_nolegend_')
    plt.xlim(cases.t_latearc.min(), cases.t_midarc.max())

mantle_Ht_max = pd.DataFrame()
for curve in GrowthCurves:
    for i in bse_Ht_max:
        mantle_Ht_max[i] = bse_Ht_max[i] - crust_Ht_max[i] * GrowthCurves[curve]
    mantle_Ht_max.plot(alpha=0.1, legend=False, c='k')
    plt.xlim(cases.t_latearc.min(), cases.t_midarc.max())

'''
