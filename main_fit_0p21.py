import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from lmfit import minimize, Parameters, fit_report

from bandstructure import BandStructure
from conductivity import Conductivity
from admr import ADMR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

sample_name = r"Nd-LSCO $p$ = 0.21"

## Initial parameters
gamma_0_ini = 15  # in THZ
gamma_0_vary = True

gamma_dos_ini = 0  # in THz
gamma_dos_vary = False

gamma_k_ini = 70  # in THz
gamma_k_vary = True

power_ini = 12
power_vary = False

mu_ini = -0.78
mu_vary = False

## Graph values
T = 25  # in Kelvin
Bamp = 45  # in Telsa

Btheta_array = np.arange(0, 95, 5)

## Fit >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Initialize the BandStructure Object
bandObject = BandStructure(mu=mu_ini)

## Interpolate data over theta of simulation
data = np.loadtxt(
    "data_NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_0.dat", dtype="float", comments="#")
x = data[:, 0]
y = data[:, 2]
rzz_0 = np.interp(Btheta_array, x, y)
data = np.loadtxt(
    "data_NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_15.dat", dtype="float", comments="#")
x = data[:, 0]
y = data[:, 2]
rzz_15 = np.interp(Btheta_array, x, y)
data = np.loadtxt(
    "data_NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_30.dat", dtype="float", comments="#")
x = data[:, 0]
y = data[:, 2]
rzz_30 = np.interp(Btheta_array, x, y)
data = np.loadtxt(
    "data_NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_45.dat", dtype="float", comments="#")
x = data[:, 0]
y = data[:, 2]
rzz_45 = np.interp(Btheta_array, x, y)


## Function residual ########
def residualFunc(pars, bandObject, rzz_0, rzz_15, rzz_30, rzz_45):

    gamma_0 = pars["gamma_0"].value
    gamma_dos = pars["gamma_dos"].value
    gamma_k = pars["gamma_k"].value
    power = pars["power"].value
    mu = pars["mu"].value

    print("gamma_0 = ", gamma_0)
    print("gamma_dos = ", gamma_dos)
    print("gamma_k = ", gamma_k)
    print("power = ", power)
    print("mu = ", mu)

    power = int(power)
    if power % 2 == 1:
        power += 1
    start_total_time = time.time()
    bandObject.mu = mu
    bandObject.discretize_FS()
    bandObject.densityOfState()
    bandObject.doping()
    condObject = Conductivity(bandObject, Bamp=45, gamma_0=gamma_0,
                              gamma_k=gamma_k, power=power, gamma_dos=gamma_dos)
    ADMRObject = ADMR([condObject])
    ADMRObject.Btheta_array = Btheta_array
    ADMRObject.runADMR()
    print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

    diff_0 = rzz_0 - ADMRObject.rzz_array[0, :]
    diff_15 = rzz_15 - ADMRObject.rzz_array[1, :]
    diff_30 = rzz_30 - ADMRObject.rzz_array[2, :]
    diff_45 = rzz_45 - ADMRObject.rzz_array[3, :]

    return np.concatenate((diff_0, diff_15, diff_30, diff_45))


## Initialize
pars = Parameters()
pars.add("gamma_0", value=gamma_0_ini, vary=gamma_0_vary, min=0)
pars.add("gamma_dos",      value=gamma_dos_ini, vary=gamma_dos_vary, min=0)
pars.add("gamma_k", value=gamma_k_ini, vary=gamma_k_vary, min=0)
pars.add("power",   value=power_ini, vary=power_vary, min=2)
pars.add("mu",      value=mu_ini, vary=mu_vary)

## Run fit algorithm
out = minimize(residualFunc, pars, args=(
    bandObject, rzz_0, rzz_15, rzz_30, rzz_45))

## Display fit report
print(fit_report(out.params))

## Export final parameters from the fit
gamma_0 = out.params["gamma_0"].value
gamma_dos = out.params["gamma_dos"].value
gamma_k = out.params["gamma_k"].value
power = out.params["power"].value
mu = out.params["mu"].value

## Compute ADMR with final parameters from the fit
bandObject.mu = mu
bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()
condObject = Conductivity(bandObject, Bamp=45, gamma_0=gamma_0,
                          gamma_k=gamma_k, power=power, gamma_dos=gamma_dos)
ADMRObject = ADMR([condObject])
ADMRObject.Btheta_array = Btheta_array
ADMRObject.runADMR()
ADMRObject.fileADMR(folder="data_NdLSCO_0p21")

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


####################################################
## Plot Parameters #################################

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure

fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

#############################################
fig.text(0.79,0.86, sample_name, ha = "right")

fig.text(0.84,0.89, r"$T$ = " + str(T) + " K", ha = "left")
fig.text(0.84,0.82, r"$H$ = " + str(Bamp) + " T", ha = "left")
#############################################

#############################################
axes.set_xlim(0,90)   # limit for xaxis
axes.set_ylim(0.990,1.008) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)
#############################################

## Allow to shift the label ticks up or down with set_pad
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

colors = ['k', '#3B528B', 'r', '#C7E500']
line = axes.plot(Btheta_array, rzz_0)
plt.setp(line, ls ="", c = colors[0], lw = 3, marker = "o", mfc = colors[0], ms = 7, mec = colors[0], mew= 0)
line = axes.plot(Btheta_array, rzz_15)
plt.setp(line, ls ="", c = colors[1], lw = 3, marker = "o", mfc = colors[1], ms = 7, mec = colors[1], mew= 0)
line = axes.plot(Btheta_array, rzz_30)
plt.setp(line, ls ="", c = colors[2], lw = 3, marker = "o", mfc = colors[2], ms = 7, mec = colors[2], mew= 0)
line = axes.plot(Btheta_array, rzz_45)
plt.setp(line, ls ="", c = colors[3], lw = 3, marker = "o", mfc = colors[3], ms = 7, mec = colors[3], mew= 0)

line = axes.plot(Btheta_array, ADMRObject.rzz_array[0,:], label = r"$\phi$ = 0")
plt.setp(line, ls ="-", c = colors[0], lw = 3, marker = "", mfc = colors[0], ms = 7, mec = colors[0], mew= 0)
line = axes.plot(Btheta_array, ADMRObject.rzz_array[1,:], label = r"$\phi$ = 15")
plt.setp(line, ls ="-", c = colors[1], lw = 3, marker = "", mfc = colors[1], ms = 7, mec = colors[1], mew= 0)
line = axes.plot(Btheta_array, ADMRObject.rzz_array[2,:], label = r"$\phi$ = 30")
plt.setp(line, ls ="-", c = colors[2], lw = 3, marker = "", mfc = colors[2], ms = 7, mec = colors[2], mew= 0)
line = axes.plot(Btheta_array, ADMRObject.rzz_array[3,:], label = r"$\phi$ = 45")
plt.setp(line, ls ="-", c = colors[3], lw = 3, marker = "", mfc = colors[3], ms = 7, mec = colors[3], mew= 0)

######################################################
plt.legend(loc = 3, fontsize = 16, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################


##///Set ticks space and minor ticks space ///#
xtics = 30 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks

majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.show()
fig.savefig("data_NdLSCO_0p21/fit_NdLSCO_0p21.pdf", bbox_inches = "tight")
plt.close()
