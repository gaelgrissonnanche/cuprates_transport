import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import os
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


## Field parameters ------------------
Bmin = 0.1
Bmax = 100
Bstep = 10
B_array = np.arange(Bmin, Bmax, Bstep)

## ADMR Published Nature 2021 ////////////////////////////////////////////////////
params = {
    "band_name": "Nd-LSCO",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 160,
    "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
    "resolution": [21, 21, 7],
    "k_max": [pi, pi, 2*pi],
    "N_time": 1000,
    "T" : 0,
    "Bamp": 45,
    "Btheta_step": 5,
    "scattering_models":["isotropic", "cos2phi"],
    "scattering_params":{"gamma_0":12.595, "gamma_k": 0, "power": 12},
}


## Scattering parameters
scattering_dict = {} # [gamma_0, gamma_k, power]
scattering_dict[25] = [12.595, 63.823, 12]
scattering_dict[20] = [11.937, 63.565, 12]
scattering_dict[12] = [10.663, 63.599, 12]
scattering_dict[6] = [9.628, 63.929, 12]


## BandObject ------------------------
bandObject = BandStructure(**params)
bandObject.runBandStructure()

## Conductivity Object ---------------
condObject = Conductivity(bandObject, **params)

## Transport coeffcients -------------

rhoxx_dict = {}
rhoxy_dict = {}

for T in sorted(scattering_dict.keys()):
    ## Empty arrays
    rhoxx_array = np.empty_like(B_array, dtype=np.float64)
    rhoxy_array = np.empty_like(B_array, dtype=np.float64)

    # Scattering parameters
    condObject.gamma_0 = scattering_dict[T][0]
    condObject.gamma_k = scattering_dict[T][1]
    condObject.power   = scattering_dict[T][2]

    for i, B in enumerate(B_array):
        condObject.Bamp = B
        condObject.runTransport()
        sigma_xx = condObject.sigma[0, 0]
        sigma_xy = condObject.sigma[0, 1]

        rhoxx = sigma_xx / ( sigma_xx**2 + sigma_xy**2) # Ohm.m
        rhoxy = sigma_xy / ( sigma_xx**2 + sigma_xy**2) # Ohm.m

        rhoxx_array[i] = rhoxx
        rhoxy_array[i] = rhoxy

    rhoxx_dict[T] = rhoxx_array
    rhoxy_dict[T] = rhoxy_array




# ## Fig / File name -------------------
# dummy = ADMR([condObject], Bphi_array=[0])
# file_name = "sim/NdLSCO_0p25/" + dummy.fileNameFunc()[3:]

# ## Save Data -------------------------
# Data = np.vstack((B_array, rhoxx_array*1e8, rhoxy_array*1e8, rhozz_array*1e8, RHa_array*1e9))
# Data = Data.transpose()

# np.savetxt(file_name + ".dat", Data, fmt='%.7e',
#            header="B[T]\trhoxx[microOhm.cm]\trhoxy[microOhm.cm]\trhozz[microOhm.cm]\tRH[mm^3/C]", comments="#")




## Figures ----------------------------------------------------------------------#

colors = ["#475CAA", "#48B149", "#ED9851", "#DC5457"][::-1]

fig_list = []


## rho_xx ///////////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize=(9.2, 5.6))
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

#############################################
fig.text(0.21,0.22, r"Nd-LSCO $p > p^{*}$")
#############################################

for i, T in enumerate(sorted(scattering_dict.keys())):
    line = axes.plot(B_array, rhoxx_dict[T]*1e8, label=r"$T$ = " + "{0:.0f}".format(T) + " K")
    plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

#############################################
axes.set_xlim(0, Bmax)
# axes.set_ylim(0, 1.25*np.max(rhoxx_array*1e8))
axes.set_ylim(0, 30)
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$H$ ( T )", labelpad=8)
axes.set_ylabel(r"$\rho_{\rm xx}$ ( $\mu\Omega$ cm )", labelpad=8)
#############################################

plt.legend(loc=4, fontsize=16, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 15 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
ytics = 10
mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
# put the format of the number of ticks
majorFormatter = FormatStrFormatter('%g')

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

axes.yaxis.set_major_locator(MultipleLocator(ytics))
axes.yaxis.set_major_formatter(majorFormatter)
axes.yaxis.set_minor_locator(MultipleLocator(mytics))

fig_list.append(fig)
#///////////////////////////////////////////////////////////////////////////////


## RH /////////////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize=(9.2, 5.6))
fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

axes.axhline(y=0, ls="--", c="k", linewidth=0.6)

#############################################
fig.text(0.79,0.87, r"Nd-LSCO $p > p^{*}$", ha="right")
#############################################

for i, T in enumerate(sorted(scattering_dict.keys())):
    line = axes.plot(B_array, rhoxy_dict[T]/B_array*1e9, label=r"$T$ = " + "{0:.0f}".format(T) + " K")
    plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

#############################################
axes.set_xlim(0, Bmax)
# axes.set_ylim(0, 1.25*np.max(RHa_array*1e9))
axes.set_ylim(0, 0.6)
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$H$ ( T )", labelpad=8)
axes.set_ylabel(r"$R_{\rm H}$ ( mm$^3$ / C )", labelpad=8)
#############################################

plt.legend(loc=3, fontsize=16, frameon=False, numpoints=1, markerscale=1, handletextpad=0.5)

## Set ticks space and minor ticks space ############
xtics = 15  # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
ytics = 0.2
mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
# put the format of the number of ticks
majorFormatter = FormatStrFormatter('%g')

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

axes.yaxis.set_major_locator(MultipleLocator(ytics))
axes.yaxis.set_major_formatter(majorFormatter)
axes.yaxis.set_minor_locator(MultipleLocator(mytics))

fig_list.append(fig)
#///////////////////////////////////////////////////////////////////////////////


plt.show()


# ## Save figures list --------------
# script_name = os.path.basename(__file__)
# figurename = script_name[0:-3] + ".pdf"

# file_figures = PdfPages(figurename)
# for fig in fig_list[::-1]:
#     file_figures.savefig(fig, bbox_inches = "tight")
# file_figures.close()
