import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from scipy import interpolate

decimals = 3
timestep=np.round(0.001,decimals) #time increment of model, used as index during scoring
tmax_evolution = 4.0
tmin, tmax = 0.0, 4.0
Tpmin, Tpmax = 1600, 2100
R_idealgas = 8.314
seconds = 3.1536e16 #how many seconds in 1Ga
M_crus = 2.35e22
M_mant = 4.043e24
C = 7.0e27
Cpcore = C - M_mant*1250.0
percentiles=list(np.round(stats.norm.cdf([-2, -1, 0, 1, 2]), 5))
percentilenames=['2.3%', '15.9%', '50%', '84.1%', '97.7%']
Curves = {'RK18': 'Rosas & Korenaga 2018', 'C03': 'Campbell 2003'}
getodds = lambda Zseries: Zseries.apply(stats.norm.pdf)/(stats.norm.pdf(0))

def generate_parameters(nruns):
    return({'Ea': mu_sig(nruns, 362.0e3, 60.0e3),
        'Qtot': mu_sig(nruns, 46.7, 1.0)})

class mu_sig:
    def __init__(self, n: int, mu: float, sig: float):
        np.random.seed(5)
        self.mu = mu
        self.sig = sig
        self.dist = np.random.normal(self.mu, self.sig, n)
        self.Zdist = (self.dist - self.mu)/self.sig
        self.normed_odds = stats.norm.pdf(self.Zdist)/stats.norm.pdf(0)

def Abbott(nruns: int):     # Abbott et al 1994
    return({'Tp_phanero': mu_sig(nruns, 1653., 10.0), 
    'Tp_latearc': mu_sig(nruns, 1840.15, 18.0),
    'Tp_midarc': mu_sig(nruns, 1891., 55.0),
    't_phanero': mu_sig(nruns, 0.302, 0.033),
    't_latearc': mu_sig(nruns, 2.756, 0.028),
    't_midarc': mu_sig(nruns, 3.344, 0.114)})

A94 = Abbott(3)
def Z_scores(df):
    times = [A94['t_phanero'].mu, A94['t_latearc'].mu, A94['t_midarc'].mu]
    temps = df.loc[times].T
    temps.columns = ['Tp_phanero', 'Tp_latearc', 'Tp_midarc']
    Zs = pd.DataFrame(index=temps.index)
    for i in temps.columns:
        Zs[i] = (temps[i] - A94[i].mu) / A94[i].sig
    Zs.fillna(np.inf, inplace=True)
    Zs['RMSE'] = (Zs**2).T.sum()**0.5
    return(Zs)

def plot_gaussian_target(ax):
    fc = ['black', 'dimgray', 'darkgray', 'lightgray'] #['black', 'gold', 'crimson', 'dodgerblue']
    alpha = 0.5
    xmeans = [A94['t_'+time].mu for time in ['phanero', 'latearc', 'midarc']]
    ymeans = [A94['Tp_'+time].mu for time in ['phanero', 'latearc', 'midarc']]
    ax.scatter(xmeans, ymeans, marker='x', c=fc[0], label=r'A94 Means $\pm$ 3 $\sigma$', zorder=10)
    x, y = 't', 'Tp'
    for n, time in enumerate(['_phanero', '_latearc', '_midarc']):
        xmu, xsig, ymu, ysig = A94[x+time].mu, A94[x+time].sig, A94[y+time].mu, A94[y+time].sig
        for sig in [3, 2, 1]:
            ax.add_patch(Ellipse((xmu, ymu), width=sig*2*xsig, height=sig*2*ysig, facecolor=fc[sig], alpha=alpha))
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(Tpmin, Tpmax)
    return(None)

def growth_models(timestep : float):
    """
    Generate representative crustal growth curves over Earth history.
    :timestep: float, either 0.001 or 0.01 - ensure "decimals" in calc is changed according to precision
    :return: pandas DataFrame
    """
    GrowthCurves = pd.read_csv('GrowthCurves.csv', header=0, index_col=0)
    GrowthCurves.index = np.round(GrowthCurves.index, decimals)
    times = np.arange(0, 4.5+timestep, timestep).round(decimals)
    return(GrowthCurves.loc[times])

def generate_time_evolution(tmax_Ga: float):
    """
    :tmax_Ga: float, max model run time
    :return: timestep, GrowthCurves, times, timerange
    """
    timestep=np.round(0.001,decimals) #time increment of model, used as index during scoring
    GrowthCurves = growth_models(timestep=timestep) #import growth curves; chg their time increment to ours
    GrowthCurves.index = np.round(GrowthCurves.index,decimals)
    times = GrowthCurves.index
    timerange = times[:int(tmax_evolution * 1000)+5]
    return(timestep, GrowthCurves, times, timerange)

def generate_HPE_budgets(nruns : int, timestep : float, trange : list):
    fn = 'HPEs.csv'
    np.random.seed(5)  #for reproducibility
    direction = 1 
    bse_budgets, cr_budgets = pd.DataFrame(), pd.DataFrame()
    Isos = pd.read_csv(fn, header=0, index_col=0)
    if nruns==1: #single heat budget
        for i in Isos.index:
            bse_budgets[i] = pd.Series((Isos['Xbse_Elem_Mean'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_mant+M_crus))) * 1.0e-12
            cr_budgets[i] = pd.Series(Isos['Xcr_Elem_Mean'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_crus)) * 1.0e-12
    else:
        for i in Isos.index:
            bse_budgets[i] = pd.Series(np.random.normal(Isos['Xbse_Elem_Mean'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_mant+M_crus),
                                    Isos['Xbse_Elem_Std'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_mant+M_crus), nruns)) * 1.0e-12
            cr_budgets[i] = pd.Series(np.random.normal(Isos['Xcr_Elem_Mean'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_crus),
                                    Isos['Xcr_Elem_Std'][i]*Isos['IsoFrac'][i]*Isos['PowerPerKg'][i]*(M_crus), nruns)) * 1.0e-12
    bse_budgets, cr_budgets = bse_budgets.transpose(), cr_budgets.transpose()  # Each column is a present-day starting case, in TW.
    times = np.arange(trange[0], trange[1]+timestep, timestep).round(3)
    df_t = pd.DataFrame(nruns * [list(times)], columns=(times))
    bse_Ht_max = pd.DataFrame(0.0, columns=df_t.columns, index=df_t.index)
    cr_Ht_max = pd.DataFrame(0.0, columns=df_t.columns, index=df_t.index)
    for isotope in Isos.index:
        m=((np.exp(Isos['Lambda'][isotope]*(df_t+direction*0.5*timestep))).mul(bse_budgets.loc[isotope], axis=0)) #MIDPOINT METHOD
        c=((np.exp(Isos['Lambda'][isotope]*(df_t+direction*0.5*timestep))).mul(cr_budgets.loc[isotope], axis=0)) #MIDPOINT METHOD
        bse_Ht_max = bse_Ht_max.add(m, fill_value=0.0)
        cr_Ht_max = cr_Ht_max.add(c, fill_value=0.0)
    return(Isos, bse_budgets, cr_budgets, bse_Ht_max.transpose(), cr_Ht_max.transpose())

def evolve_model_onebeta(b:float, mantle_HP, cases):
    batches = dict()
    STNC = -1 * seconds * 1.0e12 * timestep / C
    for curve, HP in mantle_HP.items():
        batches[curve] = pd.DataFrame(index=HP.index)
        for r, p in cases.T.items():
            trajec=list()
            Qt, Ea, Tp = p['Qm'], p['Ea'], p['Tp']
            denom = np.prod([(Tp)**(b+1), np.exp(Ea/(R_idealgas*(Tp)))**(-1*b)])
            numer = denom
            HPi = HP[r].values
            for Ht in HPi:
                dT = (Ht - Qt) * STNC
                denom = np.prod([Tp**(b+1), np.exp(Ea/(R_idealgas*(Tp)))**(-1*b)])
                Tp = Tp + dT
                numer = np.prod([Tp**(b+1), np.exp(Ea/(R_idealgas*(Tp)))**(-1*b)])
                Qt = Qt * numer/denom
                trajec.append(Tp)
            batches[curve][r]=trajec
    return(batches)

def fast_evolve_singlemodel_twobeta(b0:float, b1:float, chgt:float, Qt:float, Ea:float, Tp:float, HP:pd.Series, every=1):
    """
    Shortcut to getting model results.
    :b0: present-day beta
    :b1: past beta
    :chgt: time (Ga BP) at which beta is assumed to have transitioned - i.e. b=b0 for t<chgt.
    :Qt: present-day mantle heat flux (TW)
    :Ea: activation energy of viscosity
    :Tp: present-day mantle potential temperature (K)
    :HP: heat-production (TW) time series (index in Ga)
    :every: consider every Nth timestamp in HP.index to be a timestep. IMPORTANT: if chgt isn't an exact multiple of the timestep (e.g. chgt=1.004, every=5, and HP.index=[0.000, 0.001, ...]) the change is assumed to occur as soon as t>chgt.
    """
    tsteps = HP.index[0::every]
    tstep = tsteps[1]-tsteps[0]
    STNC = -1 * seconds * 1.0e12 * tstep / C
    trajec=[Tp]   #??
    b = b0
    denom = np.prod([(Tp)**(b+1), np.exp(Ea/(R_idealgas*(Tp)))**(-1*b)])
    numer = denom
    for time, Ht in HP.loc[tsteps].items():
        dT = (Ht - Qt) * STNC
        denom = np.prod([Tp**(b+1), np.exp(Ea/(R_idealgas*(Tp)))**(-1*b)])
        Tp = Tp + dT
        numer = np.prod([Tp**(b+1), np.exp(Ea/(R_idealgas*(Tp)))**(-1*b)])
        Qt = Qt * numer/denom
        trajec.append(Tp)
        if time>=chgt:
            b = b1
    return(dict(zip(tsteps, trajec)))

def imitate_distribution(dist: pd.DataFrame, recipient: pd.DataFrame):
    for i in dist.columns:
        recipient[i] = recipient[recipient.columns[0]]
    return(recipient)

def strict_median_heat_budget(nruns, timestep, trange : list):
    HPE_budgets = generate_HPE_budgets(nruns, timestep, [0.0, tmax])
    strict_median_heat_budget = generate_HPE_budgets(1, timestep, [0.0, tmax])
    return([HPE_budgets[0]]+[imitate_distribution(HPE_budgets[i], strict_median_heat_budget[i]) for i in [1, 2, 3, 4]])



