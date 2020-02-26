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
A_ini = 20  # in THZ
A_vary = True

B_ini = 0.5 # in THZ / K
B_vary = True

## Graph values
Bamp = 0  # in Telsa

T_array = np.arange(2.5, 50, 1)

## Fit >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Initialize the BandStructure Object
bandObject = BandStructure(bandname="LargePocket",
                           a=3.74767, b=3.74767, c=13.2,
                           t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
                           mu=-0.826,
                           numberOfKz=7, mesh_ds=1/20)

bandObject.discretize_FS()
bandObject.dos_k_func()
bandObject.doping()


## Interpolate data over theta of simulation
data = np.loadtxt(
    "data/NdLSCO_0p25/rho_a_vs_T_NdLSCO_0p24_Daou_2009.dat", dtype="float", comments="#")
x = data[:, 0]
y = data[:, 1] * 1e-8 # Omh.m
rhoxx_data = np.interp(T_array, x, y)


## Function residual ########
def residualFunc(pars, bandObject, T_array, rhoxx_data):

    A = pars["A"].value
    B = pars["B"].value

    print("A = ", A)
    print("B = ", B)

    start_total_time = time.time()
    bandObject.discretize_FS()
    bandObject.dos_k_func()
    bandObject.doping()

    rhoxx_fit = np.empty_like(T_array, dtype=np.float64)
    for i, T in enumerate(T_array):
        condObject = Conductivity(bandObject, Bamp=Bamp, gamma_0=A+B*T, gamma_k=0, power=2)
        # condObject.solveMovementFunc()
        condObject.chambersFunc(0, 0)
        rhoxx_fit[i] = 1 / condObject.sigma[0, 0]

    print("Generate one curve rho_xx time : %.6s seconds" % (time.time() - start_total_time))

    diff = rhoxx_data - rhoxx_fit

    return diff


## Initialize
pars = Parameters()
pars.add("A", value=A_ini, vary=A_vary, min=0)
pars.add("B", value=B_ini, vary=B_vary, min=0)

## Run fit algorithm
out = minimize(residualFunc, pars, args=(bandObject, T_array, rhoxx_data))

## Display fit report
print(fit_report(out.params))

## Export final parameters from the fit
A = out.params["A"].value
B = out.params["B"].value

T_fit = np.arange(0, np.max(T_array), 1)
rhoxx_fit = np.empty_like(T_fit, dtype=np.float64)
for i, T in enumerate(T_fit):
        condObject = Conductivity(bandObject, Bamp=Bamp, gamma_0=A+B*T, gamma_k=0, power=2)
        # condObject.solveMovementFunc()
        condObject.chambersFunc(0, 0)
        rhoxx_fit[i] = 1 / condObject.sigma[0, 0]

print("alpha = " + str(np.round(B*1e12*hbar/Boltzmann, 2)))

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


## rho_xx ///////////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

line = axes.plot(T_array, rhoxx_data * 1e8, label=r"data")
plt.setp(line, ls ="-", c = '#ff0000', lw = 3, marker = "", mfc = '#ff0000', ms = 7, mec = '#ff0000', mew= 0)

line = axes.plot(T_fit, rhoxx_fit * 1e8, label=r"fit")
plt.setp(line, ls ="-", c = '#000000', lw = 1, marker = "", mfc = '#000000', ms = 7, mec = '#000000', mew= 0)

#############################################
axes.set_xlim(0, np.max(T_array))
# axes.set_ylim(0, 1.25*np.max(rhoxx_array*1e8))
axes.set_ylim(bottom=0)
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$T$ ( K )", labelpad=8)
axes.set_ylabel(r"$\rho_{\rm xx}$ ( $\mu\Omega$ cm )", labelpad=8)
#############################################

plt.legend(loc=4, fontsize=14, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 10 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
#///////////////////////////////////////////////////////////////////////////////


plt.show()