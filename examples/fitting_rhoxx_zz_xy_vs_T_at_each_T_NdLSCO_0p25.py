import numpy as np
from scipy.constants import Boltzmann, hbar
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from lmfit import minimize, Parameters, fit_report

from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

sample_name = r"Nd-LSCO $p$ = 0.24"

# init_member = {
#     "bandname": "LargePocket",
#     "a": 3.74767,
#     "b": 3.74767,
#     "c": 13.2,
#     "t": 190,
#     "tp": -0.14,
#     "tpp": 0.07,
#     "tz": 0.07,
#     "tz2": 0,
#     "tz3": 0,
#     "tz4": 0,
#     "mu": -0.826,
#     "fixdoping": 0.24,
#     "numberOfKz": 7,
#     "mesh_ds": 1 / 20,
#     "Ntime": 500,
#     "T" : 0,
#     "Bamp": 0,
#     "Btheta_min": 0,
#     "Btheta_max": 90,
#     "Btheta_step": 5,
#     "Bphi_array": [0, 45],
#     "gamma_0": 15.1,
#     "gamma_k": 71.5,
#     "gamma_dos_max": 0,
#     "power": 11.5,
#     "factor_arcs": 1,
#     "seed": 72,
#     "data_T": 25,
#     "data_p": 0.24,
# }


# ## For FIT
# ranges_dict = {
#     # "t": [180,220],
#     # "tp": [-0.2,-0.1],
#     # "tpp": [-0.10,0.04],
#     # "tppp": [-0.1,0.1],
#     # "tpppp": [-0.05,0.05],
#     # "tz": [0,0.2],
#     # "tz2": [-0.07,0.07],
#     # "tz3": [-0.07,0.07],
#     # "tz4": [0,0.2],
#     # "mu": [-1.8,-1.0],
#     # "gamma_0": [5,20],
#     # "gamma_k": [0,150],
#     # "power":[1, 20],
#     # "gamma_dos_max": [0.1, 200],
#     # "factor_arcs" : [1, 300],
# }


## Initial parameters
gamma0_ini = 20  # in THZ
gamma0_vary = True
gammak_ini = 20  # in THZ
gammak_vary = True
power_ini = 10
power_vary = False

## Graph values
Bamp = 35  # in Telsa

T_array = np.arange(2.5, 42, 5)

## Fit >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Initialize the BandStructure Object
bandObject = BandStructure(bandname="LargePocket",
                           a=3.74767, b=3.74767, c=13.2,
                           t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
                           mu=-0.826,
                           numberOfKz=7, mesh_ds=1/20)

# ADMR paper parameters
bandObject = BandStructure(bandname="LargePocket",
                           a=3.74767, b=3.74767, c=13.2,
                           t=190, tp=-0.154, tpp=0.074, tz=0.076, tz2=0.00,
                           mu=-0.930,
                           numberOfKz=7, mesh_ds=1/20)

bandObject.discretize_FS()
bandObject.dos_k_func()
bandObject.doping()
print("p = " + str(bandObject.p))

## Interpolate data over theta of simulation
data = np.loadtxt(
    "data/NdLSCO_0p25/rho_a_vs_T_NdLSCO_0p24_Daou_2009.dat", dtype="float", comments="#")
x = data[:, 0]
y = data[:, 1] * 1e-8 # Ohm.m
rhoxx_data = np.interp(T_array, x, y)

data = np.loadtxt(
    "data/NdLSCO_0p25/rho_c_vs_T_NdLSCO_0p24_H_45T_Daou_et_al_Nat_phys_2009.dat", dtype="float", comments="#")
x = data[:, 0]
y = data[:, 1] * 1e-5 # Ohm.m
rhozz_data = np.interp(T_array, x, y)

data = np.loadtxt("data/NdLSCO_0p25/RH_vs_T_NdLSCO_0p24_Daou_2009.dat",
                  dtype="float",
                  comments="#")
x = data[:, 0]
y = data[:, 1] * 1e-9 * Bamp # Ohm.m
rhoxy_data = np.interp(T_array, x, y)

## Empty arrays
gamma0_array = np.empty_like(T_array, dtype=np.float64)
gammak_array = np.empty_like(T_array, dtype=np.float64)
power_array = np.empty_like(T_array, dtype=np.float64)
rhoxx_fit = np.empty_like(T_array, dtype=np.float64)
rhozz_fit = np.empty_like(T_array, dtype=np.float64)
rhoxy_fit = np.empty_like(T_array, dtype=np.float64)

## Function residual ########
def residualFunc(pars, bandObject, rhoxx_point, rhozz_point, rhoxy_point):

    gamma0 = pars["gamma0"].value
    gammak = pars["gammak"].value
    power  = pars["power"].value

    condObject = Conductivity(bandObject, Bamp=Bamp, gamma_0=gamma0, gamma_k=gammak, power=power)
    condObject.solveMovementFunc()
    condObject.chambersFunc(0, 0)
    rhoxx_fit = 1 / condObject.sigma[0, 0]
    condObject.chambersFunc(2, 2)
    rhozz_fit = 1 / condObject.sigma[2, 2]
    condObject.chambersFunc(0, 1)
    rhoxy_fit = condObject.sigma[0, 1] / (condObject.sigma[0, 0]**2 + condObject.sigma[0, 1]**2)

    diffxx = rhoxx_point - rhoxx_fit
    diffzz = rhozz_point - rhozz_fit
    diffxy = rhoxy_point - rhoxy_fit

    return np.array([diffxx, diffzz, diffxy])


for i, T in enumerate(T_array):
    ## Initialize
    pars = Parameters()
    pars.add("gamma0", value=gamma0_ini, vary=gamma0_vary, min=0)
    pars.add("gammak", value=gammak_ini, vary=gammak_vary, min=0)
    pars.add("power", value=power_ini, vary=power_vary, min=0)

    ## Run fit algorithm
    out = minimize(residualFunc, pars, args=(bandObject, rhoxx_data[i], rhozz_data[i], rhoxy_data[i]))

    ## Display fit report
    print("T = " + str(T) + " K")
    print(fit_report(out.params))

    ## Export final parameters from the fit
    gamma0_array[i] = out.params["gamma0"].value
    gammak_array[i] = out.params["gammak"].value
    power_array[i]  = out.params["power"].value

    condObject = Conductivity(bandObject, Bamp=Bamp, gamma_0=gamma0_array[i], gamma_k=gammak_array[i], power=power_array[i])
    condObject.solveMovementFunc()
    condObject.chambersFunc(0, 0)
    rhoxx_fit[i] = 1 / condObject.sigma[0, 0]
    condObject.chambersFunc(2, 2)
    rhozz_fit[i] = 1 / condObject.sigma[2, 2]
    condObject.chambersFunc(0, 1)
    rhoxy_fit[i] = condObject.sigma[0, 1] / (condObject.sigma[0, 0]**2 + condObject.sigma[0, 1]**2)


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
## Figures ////////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

#///// RC Parameters //////#
mpl.rcdefaults()
mpl.rcParams['font.size'] = 24. # change the size of the font in every figure
mpl.rcParams['font.family'] = 'Arial' # font Arial in every figure
mpl.rcParams['axes.labelsize'] = 24.
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6
mpl.rcParams['axes.linewidth'] = 0.6 # thickness of the axes lines
mpl.rcParams['pdf.fonttype'] = 3  # Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
                                    # editing the text in illustrator





fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8))
fig.subplots_adjust(left = 0.10, right = 0.96, bottom = 0.16, top = 0.95, wspace=0.45) # adjust the box of axes regarding the figure size

## gamma_0 ////////////////////////////////////////////////////////////////////#

line = axes[0].plot(T_array, gamma0_array)
plt.setp(line, ls ="", c = '#0000ff', lw = 3, marker = "s", mfc = '#0000ff', ms = 7, mec = '#0000ff', mew= 0)

#############################################
axes[0].set_xlim(0, np.max(T_array))
# axes.set_ylim(0, 1.25*np.max(rhoxx_array*1e8))
axes[0].set_ylim(bottom=0)
axes[0].tick_params(axis='x', which='major', pad=7)
axes[0].tick_params(axis='y', which='major', pad=8)
axes[0].set_xlabel(r"$T$ ( K )", labelpad=8)
axes[0].set_ylabel(r"$\gamma_{\rm 0}$ ( THz )", labelpad=8)
#############################################

## Set ticks space and minor ticks space ############
xtics = 20 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes[0].xaxis.set_major_locator(MultipleLocator(xtics))
axes[0].xaxis.set_major_formatter(majorFormatter)
axes[0].xaxis.set_minor_locator(MultipleLocator(mxtics))
#///////////////////////////////////////////////////////////////////////////////


## gamma_k ////////////////////////////////////////////////////////////////////#
line = axes[1].plot(T_array, gammak_array)
plt.setp(line, ls ="", c = '#ff0000', lw = 3, marker = "s", mfc = '#ff0000', ms = 7, mec = '#ff0000', mew= 0)

#############################################
axes[1].set_xlim(0, np.max(T_array))
# axes[1].set_ylim(0, 1.25*np.max(rhozz_array*1e8))
axes[1].set_ylim(bottom=0)
axes[1].tick_params(axis='x', which='major', pad=7)
axes[1].tick_params(axis='y', which='major', pad=8)
axes[1].set_xlabel(r"$T$ ( K )", labelpad=8)
axes[1].set_ylabel(r"$\gamma_{\rm k}$ ( THz )", labelpad=8)
#############################################

## Set ticks space and minor ticks space ############
xtics = 20 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes[1].xaxis.set_major_locator(MultipleLocator(xtics))
axes[1].xaxis.set_major_formatter(majorFormatter)
axes[1].xaxis.set_minor_locator(MultipleLocator(mxtics))
#///////////////////////////////////////////////////////////////////////////////

## power //////////////////////////////////////////////////////////////////////#
line = axes[2].plot(T_array, power_array)
plt.setp(line, ls ="", c = '#3cc44d', lw = 3, marker = "s", mfc = '#3cc44d', ms = 7, mec = '#3cc44d', mew= 0)

#############################################
axes[2].set_xlim(0, np.max(T_array))
# axes[2].set_ylim(0, 1.25*np.max(rhozz_array*1e8))
axes[2].set_ylim(bottom=0)
axes[2].tick_params(axis='x', which='major', pad=7)
axes[2].tick_params(axis='y', which='major', pad=8)
axes[2].set_xlabel(r"$T$ ( K )", labelpad=8)
axes[2].set_ylabel(r"power", labelpad=8)
#############################################

## Set ticks space and minor ticks space ############
xtics = 20 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes[2].xaxis.set_major_locator(MultipleLocator(xtics))
axes[2].xaxis.set_major_formatter(majorFormatter)
axes[2].xaxis.set_minor_locator(MultipleLocator(mxtics))
#///////////////////////////////////////////////////////////////////////////////







fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8))
fig.subplots_adjust(left = 0.10, right = 0.96, bottom = 0.16, top = 0.95, wspace=0.45) # adjust the box of axes regarding the figure size

## rho_xx ///////////////////////////////////////////////////////////////////////#

line = axes[0].plot(T_array, rhoxx_data * 1e8, label=r"data")
plt.setp(line, ls ="-", c = '#0000ff', lw = 3, marker = "", mfc = '#0000ff', ms = 7, mec = '#0000ff', mew= 0)

line = axes[0].plot(T_array, rhoxx_fit * 1e8, label=r"fit")
plt.setp(line, ls ="", c = '#000000', lw = 1, marker = "o", mfc = '#000000', ms = 7, mec = '#000000', mew= 0)

#############################################
axes[0].set_xlim(0, np.max(T_array))
# axes.set_ylim(0, 1.25*np.max(rhoxx_array*1e8))
axes[0].set_ylim(bottom=0)
axes[0].tick_params(axis='x', which='major', pad=7)
axes[0].tick_params(axis='y', which='major', pad=8)
axes[0].set_xlabel(r"$T$ ( K )", labelpad=8)
axes[0].set_ylabel(r"$\rho_{\rm xx}$ ( $\mu\Omega$ cm )", labelpad=8)
#############################################

axes[0].legend(loc=4, fontsize=14, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 20 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes[0].xaxis.set_major_locator(MultipleLocator(xtics))
axes[0].xaxis.set_major_formatter(majorFormatter)
axes[0].xaxis.set_minor_locator(MultipleLocator(mxtics))
#///////////////////////////////////////////////////////////////////////////////


## rho_zz ///////////////////////////////////////////////////////////////////////#
line = axes[1].plot(T_array, rhozz_data * 1e5, label=r"data")
plt.setp(line, ls ="-", c = '#ff0000', lw = 3, marker = "", mfc = '#ff0000', ms = 7, mec = '#ff0000', mew= 0)

line = axes[1].plot(T_array, rhozz_fit * 1e5, label=r"fit")
plt.setp(line, ls ="", c = '#000000', lw = 1, marker = "o", mfc = '#000000', ms = 7, mec = '#000000', mew= 0)

#############################################
axes[1].set_xlim(0, np.max(T_array))
# axes[1].set_ylim(0, 1.25*np.max(rhozz_array*1e8))
axes[1].set_ylim(bottom=0)
axes[1].tick_params(axis='x', which='major', pad=7)
axes[1].tick_params(axis='y', which='major', pad=8)
axes[1].set_xlabel(r"$T$ ( K )", labelpad=8)
axes[1].set_ylabel(r"$\rho_{\rm zz}$ ( m$\Omega$ cm )", labelpad=8)
#############################################

axes[1].legend(loc=4, fontsize=14, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 20 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes[1].xaxis.set_major_locator(MultipleLocator(xtics))
axes[1].xaxis.set_major_formatter(majorFormatter)
axes[1].xaxis.set_minor_locator(MultipleLocator(mxtics))
#///////////////////////////////////////////////////////////////////////////////

## rho_xy /////////////////////////////////////////////////////////////////////#
line = axes[2].plot(T_array, rhoxy_data / Bamp * 1e9, label=r"data")
plt.setp(line, ls ="-", c = '#3cc44d', lw = 3, marker = "", mfc = '#3cc44d', ms = 7, mec = '#3cc44d', mew= 0)

line = axes[2].plot(T_array, rhoxy_fit / Bamp * 1e9, label=r"fit")
plt.setp(line, ls ="", c = '#000000', lw = 1, marker = "o", mfc = '#000000', ms = 7, mec = '#000000', mew= 0)

#############################################
axes[2].set_xlim(0, np.max(T_array))
# axes[2].set_ylim(0, 1.25*np.max(rhozz_array*1e8))
axes[2].set_ylim(bottom=0)
axes[2].tick_params(axis='x', which='major', pad=7)
axes[2].tick_params(axis='y', which='major', pad=8)
axes[2].set_xlabel(r"$T$ ( K )", labelpad=8)
axes[2].set_ylabel(r"$R_{\rm H}$ ( mm$^3$ / C )", labelpad=8)
#############################################

axes[2].legend(loc=4, fontsize=14, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 20 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes[2].xaxis.set_major_locator(MultipleLocator(xtics))
axes[2].xaxis.set_major_formatter(majorFormatter)
axes[2].xaxis.set_minor_locator(MultipleLocator(mxtics))
#///////////////////////////////////////////////////////////////////////////////







plt.show()