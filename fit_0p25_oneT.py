import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from lmfit import minimize, Parameters, fit_report
from matplotlib.backends.backend_pdf import PdfPages

from bandstructure import BandStructure
from conductivity import Conductivity
from admr import ADMR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

sample_name = r"Nd-LSCO $p$ = 0.25"

# Which temperature to fit?
T = 6  # in Kelvin

## Initial parameters
gamma_0_ini  = 0 # in THZ
gamma_0_vary = False

gamma_dos_max_ini = 215 # in THz
gamma_dos_max_vary = True

gamma_k_ini  = 90 # in THz
gamma_k_vary = True

power_ini    = 13
power_vary   = True

mu_ini       = -0.826
mu_vary      = False

t_ini       = 190
t_vary      = False

Btheta_array = np.arange(0, 95, 5)
Bamp = 45  # in Telsa

## Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

data_dict = {} # keys (T, phi), content [filename, theta, rzz, theta_cut]
data_dict[25, 0] = ["data_NdLSCO_0p25/0p25_0degr_45T_25K.dat", 0, 1, 90]
data_dict[25, 15] = ["data_NdLSCO_0p25/0p25_15degr_45T_25K.dat", 0, 1, 90]
data_dict[25, 30] = ["data_NdLSCO_0p25/0p25_30degr_45T_25K.dat", 0, 1, 90]
data_dict[25, 45] = ["data_NdLSCO_0p25/0p25_45degr_45T_25K.dat", 0, 1, 90]

data_dict[20, 0] = ["data_NdLSCO_0p25/0p25_0degr_45T_20K.dat", 0, 1, 90]
data_dict[20, 15] = ["data_NdLSCO_0p25/0p25_15degr_45T_20K.dat", 0, 1, 90]
data_dict[20, 30] = ["data_NdLSCO_0p25/0p25_30degr_45T_20K.dat", 0, 1, 90]
data_dict[20, 45] = ["data_NdLSCO_0p25/0p25_45degr_45T_20K.dat", 0, 1, 90]

data_dict[12, 0] = ["data_NdLSCO_0p25/0p25_0degr_45T_12K.dat", 0, 1, 83.5]
data_dict[12, 15] = ["data_NdLSCO_0p25/0p25_15degr_45T_12K.dat", 0, 1, 83.5]
data_dict[12, 45] = ["data_NdLSCO_0p25/0p25_45degr_45T_12K.dat", 0, 1, 83.5]

data_dict[6, 0] = ["data_NdLSCO_0p25/0p25_0degr_45T_6K.dat", 0, 1, 73.5]
data_dict[6, 15] = ["data_NdLSCO_0p25/0p25_15degr_45T_6K.dat", 0, 1, 73.5]
data_dict[6, 45] = ["data_NdLSCO_0p25/0p25_45degr_45T_6K.dat", 0, 1, 73.5]

## Create array of phi at a temperature
Bphi_array = []
for t, phi in data_dict.keys():
    if (T == t):
        Bphi_array.append(phi)
Bphi_array.sort()

# Cut Btheta_array to theta_cut
Btheta_cut_array = np.zeros(len(Bphi_array))
for i, phi in enumerate(Bphi_array):
    Btheta_cut_array[i] = data_dict[T, phi][3]
Btheta_cut_min = np.min(Btheta_cut_array) # minimum cut for Btheta_array

# New Btheta_array with cut off if necessary
Btheta_array = Btheta_array[Btheta_array <= Btheta_cut_min]

## Interpolate data over theta of simulation
rzz_matrix = np.zeros((len(Bphi_array), len(Btheta_array)))
for i, phi in enumerate(Bphi_array):
    filename  = data_dict[T, phi][0]
    col_theta = data_dict[T, phi][1]
    col_rzz   = data_dict[T, phi][2]

    data  = np.loadtxt(filename, dtype="float", comments="#")
    theta = data[:, col_theta]
    rzz   = data[:, col_rzz]
    rzz_i = np.interp(Btheta_array, theta, rzz)

    rzz_matrix[i,:] = rzz_i


## Function residual ########
def residualFunc(pars, rzz_matrix):

    gamma_0 = pars["gamma_0"].value
    gamma_dos_max = pars["gamma_dos_max"].value
    gamma_k = pars["gamma_k"].value
    power = pars["power"].value
    t = pars["t"].value
    mu = pars["mu"].value

    print("gamma_0 = ", gamma_0)
    print("gamma_dos_max = ", gamma_dos_max)
    print("gamma_k = ", gamma_k)
    print("power = ", power)
    print("t = ", t)
    print("mu = ", mu)

    # ADMR
    start_total_time = time.time()

    bandObject = BandStructure(bandname="hPocket",
                           a=3.74767, b=3.74767, c=13.2,
                           t=t, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
                           mu=mu,
                           numberOfKz=7, mesh_ds=np.pi/20)

    bandObject.discretize_FS()
    bandObject.densityOfState()
    bandObject.doping()
    condObject = Conductivity(bandObject, Bamp=45, gamma_0=gamma_0, gamma_k=gamma_k, power=power, gamma_dos_max=gamma_dos_max)
    ADMRObject = ADMR([condObject])
    ADMRObject.Bphi_array = np.array(Bphi_array)
    ADMRObject.Btheta_array = np.array(Btheta_array)
    ADMRObject.runADMR()
    print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

    diff_rzz_matrix = np.zeros_like(rzz_matrix)
    for i in range(len(Bphi_array)):
        diff_rzz_matrix[i, :] = rzz_matrix[i, :] - ADMRObject.rzz_array[i, :]

    return diff_rzz_matrix

## Initialize
pars = Parameters()
pars.add("gamma_0", value = gamma_0_ini, vary = gamma_0_vary, min = 0)
pars.add("gamma_dos_max",      value = gamma_dos_max_ini, vary = gamma_dos_max_vary, min = 0)
pars.add("gamma_k", value = gamma_k_ini, vary = gamma_k_vary, min = 0)
pars.add("power",   value = power_ini, vary = power_vary, min = 0)
pars.add("t",      value = t_ini, vary = t_vary)
pars.add("mu",      value = mu_ini, vary = mu_vary)

## Run fit algorithm
out = minimize(residualFunc, pars, args=(rzz_matrix,))

## Display fit report
print(fit_report(out.params))

## Export final parameters from the fit
gamma_0 = out.params["gamma_0"].value
gamma_dos_max      = out.params["gamma_dos_max"].value
gamma_k = out.params["gamma_k"].value
power   = out.params["power"].value
t       = out.params["t"].value
mu      = out.params["mu"].value

## Compute ADMR with final parameters from the fit
bandObject = BandStructure(bandname="hPocket",
                        a=3.74767, b=3.74767, c=13.2,
                        t=t, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
                        mu=mu,
                        numberOfKz=7, mesh_ds=np.pi/20)

bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()
condObject = Conductivity(bandObject, Bamp=45, gamma_0=gamma_0, gamma_k=gamma_k, power=power, gamma_dos_max=gamma_dos_max)
ADMRObject = ADMR([condObject])
ADMRObject.Bphi_array = np.array(Bphi_array)
ADMRObject.Btheta_array = np.array(Btheta_array)
ADMRObject.runADMR()
ADMRObject.fileADMR(folder="data_NdLSCO_0p25")

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
## Figures ////////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

fig_list = []

## Parameters ///////////////////////////////////////////////////////////////////#
fig_list.append(condObject.figParameters(fig_show=False))

####################################################
## Plot Parameters #################################

fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

#############################################
fig.text(0.99,0.02, sample_name, fontsize=20, color="r", ha="right")

fig.text(0.84,0.89, r"$T$ = " + str(T) + " K", ha = "left")
fig.text(0.84,0.82, r"$H$ = " + str(Bamp) + " T", ha = "left")
#############################################

#############################################
axes.set_xlim(0, 90)
# axes.set_ylim(1+1.2*(min_y-1),1.2*(max_y-1)+1)
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)
#############################################


colors = ['k', '#3B528B', 'r', '#C7E500']

for i in range(len(Bphi_array)):
    line = axes.plot(Btheta_array, rzz_matrix[i,:])
    plt.setp(line, ls ="", c = colors[i], lw = 3, marker = "o", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

for i, phi in enumerate(Bphi_array):
    line = axes.plot(Btheta_array, ADMRObject.rzz_array[i,:], label = r"$\phi$ = " + str(phi))
    plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

######################################################
plt.legend(loc = 1, fontsize = 14, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################

##///Set ticks space and minor ticks space ///#
xtics = 30 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks

majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

fig_list.append(fig)
#///////////////////////////////////////////////////////////////////////////////

plt.show()

## Save figures list --------------
file_figures = PdfPages("data_NdLSCO_0p25/fit_NdLSCO_0p25_" + str(T) + "K.pdf")
for fig in fig_list[::-1]:
    file_figures.savefig(fig)
file_figures.close()
