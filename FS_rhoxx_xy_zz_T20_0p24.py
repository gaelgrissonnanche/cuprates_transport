import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

from bandstructure import BandStructure
from conductivity import Conductivity
from admr import ADMR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

e = 1.6e-19 # C

## Field parameters ------------------
Bmin = 0.1
Bmax = 40
Bstep = 10
B_array = np.arange(Bmin, Bmax, Bstep)

## BandObject ------------------------
bandObject = BandStructure(bandname="LargePocket",
                           a=3.74767, b=3.74767, c=13.2,
                           t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
                           mu=-0.826,
                           numberOfKz=7, mesh_ds=1/40)

bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()

# # bandObject.figMultipleFS2D()
# # bandObject.figDiscretizeFS2D()

## Conductivity Object ---------------
condObject = Conductivity(bandObject, Bamp=Bmin, Bphi=0, Btheta=0, T=25,
                          gamma_0=14.0, gamma_k=78.1, power=12, gamma_dos_max=0) # T = 20K, p = 0.24 from fit ADMR
condObject.Ntime = 1000 # better for high magnetic field values
condObject.epsilon_N = 10

## Transport coeffcients -------------

## Empty arrays
rhoxx_array = np.empty_like(B_array, dtype=np.float64)
rhoxy_array = np.empty_like(B_array, dtype=np.float64)
rhozz_array = np.empty_like(B_array, dtype=np.float64)
RH_array = np.empty_like(B_array, dtype=np.float64)

for i, B in enumerate(B_array):

    condObject.Bamp = B
    condObject.solveMovementFunc()
    condObject.chambersFunc(0, 0)
    condObject.chambersFunc(0, 1)
    condObject.chambersFunc(2, 2)
    sigma_xx = condObject.sigma[0, 0]
    sigma_xy = condObject.sigma[0, 1]
    sigma_zz = condObject.sigma[2, 2]

    rhoxx = sigma_xx / ( sigma_xx**2 + sigma_xy**2) # Ohm.m
    rhoxy = sigma_xy / ( sigma_xx**2 + sigma_xy**2) # Ohm.m
    rhozz = 1 / sigma_zz # Ohm.m
    RH = rhoxy / B # m^3/C

    rhoxx_array[i] = rhoxx
    rhoxy_array[i] = rhoxy
    rhozz_array[i] = rhozz
    RH_array[i] = RH


## Info results ----------------------
nH = 1 / (RH_array[0] * e)
d = bandObject.c*1e-10 / 2
V = (bandObject.a*1e-10)**2 * d
n = V * nH
p = n - 1
print("p = " + "{0:.3f}".format(bandObject.p))
print("n - 1 = ", np.round(p, 3))


## Fig / File name -------------------
dummy = ADMR([condObject], Bphi_array=[0])
file_name = "results_sim/FS_Rxx_xy_zz" + dummy.fileNameFunc()[3:]

## Save Data -------------------------
Data = np.vstack((B_array, rhoxx_array*1e8, rhoxy_array*1e8, rhozz_array*1e8, RH_array*1e9))
Data = Data.transpose()

np.savetxt(file_name + ".dat", Data, fmt='%.7e',
           header="B[T]\trhoxx[microOhm.cm]\trhoxy[microOhm.cm]\trhozz[microOhm.cm]\tRH[mm^3/C]", comments="#")




## Figures ----------------------------------------------------------------------#

fig_list = []

## Parameters ///////////////////////////////////////////////////////////////////#
fig_list.append(condObject.figParameters(fig_show=False))


## rho_zz ///////////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

#############################################
# fig.text(0.79,0.22, r"$1 - n$ = " + "{0:.3f}".format(p) , ha = "right")
#############################################


# Data @ 35T from Daou 2009 @ 20K
line = axes.plot([35], [6.59], label=r"$p$ = 0.24, T = 20K (data)")
plt.setp(line, ls ="", c = '#c0c0c0', lw = 3, marker = "s", mfc = '#c0c0c0', ms = 9, mec = '#c0c0c0', mew= 0)

line = axes.plot(B_array, rhozz_array*1e5, label=r"$p$ = " + "{0:.3f}".format(bandObject.p) + " (sim)")
plt.setp(line, ls ="-", c = '#ff6a6a', lw = 3, marker = "", mfc = '#ff6a6a', ms = 7, mec = '#ff6a6a', mew= 0)

#############################################
axes.set_xlim(0, Bmax)
axes.set_ylim(0, 1.25*np.max(rhozz_array*1e5))
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$H$ ( T )", labelpad=8)
axes.set_ylabel(r"$\rho_{\rm zz}$ ( m$\Omega$ cm )", labelpad=8)
#############################################

plt.legend(loc=4, fontsize=14, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 10 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

fig_list.append(fig)
#///////////////////////////////////////////////////////////////////////////////


## rho_xx ///////////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

#############################################
# fig.text(0.79,0.22, r"$1 - n$ = " + "{0:.3f}".format(p) , ha = "right")
#############################################

## Load resistivity data ####################
data = np.loadtxt("data_NdLSCO_0p25/NdLSCOp24_rho_RH_T20_H0_37p5T_FS.txt",
                  dtype="float",
                  comments="#")
B_data = data[:, 0]
rhoxx_data = data[:, 1]

line = axes.plot(B_data, rhoxx_data, label=r"$p$ = 0.24, T = 20K (data)")
plt.setp(line, ls ="-", c = '#c0c0c0', lw = 3, marker = "", mfc = '#c0c0c0', ms = 7, mec = '#c0c0c0', mew= 0)

line = axes.plot(B_array, rhoxx_array*1e8, label=r"$p$ = " + "{0:.3f}".format(bandObject.p) + " (sim)")
plt.setp(line, ls ="-", c = '#0080ff', lw = 3, marker = "", mfc = '#0080ff', ms = 7, mec = '#0080ff', mew= 0)

#############################################
axes.set_xlim(0, Bmax)
# axes.set_ylim(0, 1.25*np.max(rhoxx_array*1e8))
axes.set_ylim(0, 40)
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$H$ ( T )", labelpad=8)
axes.set_ylabel(r"$\rho_{\rm xx}$ ( $\mu\Omega$ cm )", labelpad=8)
#############################################

plt.legend(loc=4, fontsize=14, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 10 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

fig_list.append(fig)
#///////////////////////////////////////////////////////////////////////////////


## RH ///////////////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

axes.axhline(y=0, ls="--", c="k", linewidth=0.6)

#############################################
# fig.text(0.79, 0.22, r"$1 - n$ = " + "{0:.3f}".format(p), ha="right")
#############################################

## Load resistivity data ####################
data = np.loadtxt("data_NdLSCO_0p25/NdLSCOp24_rho_RH_T20_H0_37p5T_FS.txt",
                  dtype="float",
                  comments="#")
B_data = data[:, 0]
RH_data = data[:, 2]

line = axes.plot(B_data, RH_data, label=r"$p$ = 0.24, T = 20K (data)")
plt.setp(line, ls ="-", c = '#c0c0c0', lw = 3, marker = "", mfc = '#c0c0c0', ms = 7, mec = '#c0c0c0', mew= 0)

line = axes.plot(B_array, RH_array * 1e9, label=r"$p$ = " + "{0:.3f}".format(bandObject.p)+ " (sim)")
plt.setp(line, ls="-", c='#00ff80', lw=3, marker="",
         mfc='#00ff80', ms=7, mec='#00ff80', mew=0)

#############################################
axes.set_xlim(0, Bmax)
# axes.set_ylim(0, 1.25*np.max(RH_array*1e9))
axes.set_ylim(0, 1.0)
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$H$ ( T )", labelpad=8)
axes.set_ylabel(r"$R_{\rm H}$ ( mm$^3$ / C )", labelpad=8)
#############################################

plt.legend(loc=1, fontsize=14, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 10  # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
# ytics = 1
# mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
# put the format of the number of ticks
majorFormatter = FormatStrFormatter('%g')

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

# axes.yaxis.set_major_locator(MultipleLocator(ytics))
# axes.yaxis.set_major_formatter(majorFormatter)
# axes.yaxis.set_minor_locator(MultipleLocator(mytics))

fig_list.append(fig)
#///////////////////////////////////////////////////////////////////////////////


plt.show()


## Save figures list --------------
file_figures = PdfPages(file_name + ".pdf")
for fig in fig_list[::-1]:
    file_figures.savefig(fig)
file_figures.close()
