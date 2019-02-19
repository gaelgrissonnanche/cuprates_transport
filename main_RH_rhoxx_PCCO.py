import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter

from bandstructure import BandStructure
from conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

e = 1.6e-19 # C
a = 3.95e-10 #m
c = 12.07e-10 # m

Bmin = 0.1
Bmax = 10
Bstep = 0.2
B_array = np.arange(Bmin, Bmax, Bstep)

# Parameters Simon
bandObject = BandStructure(mu=-0.2, t=250, tp=-0.17, tpp=0.08, tz=0, tz2=0, a=a*1e10, b=a*1e10, c=c*1e10, mesh_ds = np.pi/15, numberOfKz=1)
# Parameters Toni Helm
# bandObject = BandStructure(mu=1, t=380, tp=0.32, tpp=0.16, tz=0, tz2=0, a=a*1e10, b=a*1e10, c=c*1e10, mesh_ds = np.pi/15, numberOfKz=1)
bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()

# Data = np.vstack((bandObject.kf[0,:], bandObject.kf[1,:], bandObject.kf[2,:]))
# Data = Data.transpose()
# np.savetxt("Fermi_Surface_grid.dat", Data, fmt='%.7e',
# header = "kx\tky\tkz", comments = "#")

# condObject = Conductivity(bandObject, Bamp=5, Bphi=0, Btheta=0, gamma_0=100, gamma_k=0, power=2)
# condObject.solveMovementFunc()
# condObject.figOnekft()

bandObject.figMultipleFS2D()
# # bandObject.figDiscretizeFS2D()

rhoxx_array = np.empty_like(B_array, dtype = np.float64)
rhoxy_array = np.empty_like(B_array, dtype = np.float64)
RH_array = np.empty_like(B_array, dtype = np.float64)
for i, B in enumerate(B_array):

    condObject = Conductivity(bandObject, Bamp=B, Bphi=0, Btheta=0, gamma_0=20, gamma_k=0, power=2)
    condObject.solveMovementFunc()
    condObject.chambersFunc(0, 0)
    condObject.chambersFunc(0, 1)
    sigma_xx = condObject.sigma[0,0]
    sigma_xy = condObject.sigma[0,1]

    rhoxx = sigma_xx / ( sigma_xx**2 + sigma_xy**2) # Ohm.m
    rhoxy = sigma_xy / ( sigma_xx**2 + sigma_xy**2) # Ohm.m
    RH = rhoxy / B # m^3/C

    rhoxx_array[i] = rhoxx
    rhoxy_array[i] = rhoxy
    RH_array[i] = RH

## Save Data
Data = np.vstack((B_array, rhoxx_array*1e8, rhoxx_array*1e8, RH_array*1e9))
Data = Data.transpose()

np.savetxt("RH_Gael.dat", Data, fmt='%.7e',
header = "B[T]\trhoxx[microOhm.cm]\trhoxy[microOhm.cm]\tRH[mm^3/C]", comments = "#")

nH = 1 / ( RH_array[-1] * - e )
d = c / 2
V = a**2 * d
n = V * nH
print("n = ", np.round(n, 3) )




## RH plot ////////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95)

axes.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

#############################################
fig.text(0.79,0.86, r"$p$ = " + "{0:.3f}".format(bandObject.p), ha = "right")
fig.text(0.79,0.80, r"$n$ = " + "{0:.3f}".format(n) , ha = "right")
#############################################

#############################################
axes.set_xlim(Bmin,Bmax)   # limit for xaxis
# axes.set_ylim(-3.5,1.2) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$H$ ( T )", labelpad = 8)
axes.set_ylabel(r"$R_{\rm H}$ ( mm$^3$ / C )", labelpad = 8)
#############################################

## Allow to shift the label ticks up or down with set_pad
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

line = axes.plot(B_array, RH_array * 1e9)
plt.setp(line, ls ="-", c = '#46FFA1', lw = 3, marker = "o", mfc = '#46FFA1', ms = 7, mec = '#46FFA1', mew= 0)
######################################################

# ## Set ticks space and minor ticks space ############
# #####################################################.

# xtics = 30 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks
# # ytics = 1
# # mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

# axes.yaxis.set_major_locator(MultipleLocator(ytics))
# axes.yaxis.set_major_formatter(majorFormatter)
# axes.yaxis.set_minor_locator(MultipleLocator(mytics))

######################################################

# script_name = os.path.basename(__file__)
# figurename = script_name[0:-3] + ".pdf"

# fig.savefig(figurename, bbox_inches = "tight")
# plt.close()
#///////////////////////////////////////////////////////////////////////////////



## rho_xx plot ////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95)

#############################################
fig.text(0.79,0.28, r"$p$ = " + "{0:.3f}".format(bandObject.p), ha = "right")
fig.text(0.79,0.22, r"$n$ = " + "{0:.3f}".format(n) , ha = "right")
#############################################

## Allow to shift the label ticks up or down with set_pad
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

line = axes.plot(B_array, rhoxx_array * 1e8)
plt.setp(line, ls ="-", c = '#5B22FF', lw = 3, marker = "o", mfc = '#5B22FF', ms = 7, mec = '#5B22FF', mew= 0)

#############################################
axes.set_xlim(Bmin,Bmax)   # limit for xaxis
axes.set_ylim(bottom = 0)
axes.set_xlabel(r"$H$ ( T )", labelpad = 8)
axes.set_ylabel(r"$\rho_{\rm xx}$ ( $\mu\Omega$ cm )", labelpad = 8)
#############################################

# ## Set ticks space and minor ticks space ############
# #####################################################.

# xtics = 30 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks
# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
# ######################################################

# script_name = os.path.basename(__file__)
# figurename = script_name[0:-3] + ".pdf"

plt.show()
# fig.savefig(figurename, bbox_inches = "tight")
# plt.close()
#///////////////////////////////////////////////////////////////////////////////




# ## Cumulatit exp plot /////////////////////////////////////////////////////////#

# condObject = Conductivity(bandObject, Bamp=45, Bphi=0, Btheta=0, gamma_0 = 25, gamma_k = 0, power = 0)
# condObject.solveMovementFunc()
# condObject.figOnekft()
# t_test = condObject.t
# exp_test = np.exp(- condObject.tOverTauFunc(condObject.kft[:, 0, :], condObject.vft[:, 0, :]))

# fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
# fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95)

# #############################################
# fig.text(0.79,0.22, r"$\tau_{\rm N}$ = " + "{0:.3f}".format(condObject.tau_0) + " ps", ha = "right")
# #############################################

# ## Allow to shift the label ticks up or down with set_pad
# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# line = axes.plot(t_test, np.cumsum(exp_test))
# plt.setp(line, ls ="-", c = '#5B22FF', lw = 3, marker = "", mfc = '#5B22FF', ms = 7, mec = '#5B22FF', mew= 0)

# #############################################
# axes.set_xlim(0, None)
# axes.set_ylim(0, None)
# axes.set_xlabel(r"t ( ps )", labelpad = 8)
# axes.set_ylabel(r"$\sum_{\rm t}$ exp( $\int_{0}^{t}$ ${\rm \dfrac{-dt'}{\tau ( t' )}} )$", labelpad = 8)
# #############################################

# ## Set ticks space and minor ticks space ############
# #####################################################.

# # xtics = 30 # space between two ticks
# # mxtics = xtics / 2.  # space between two minor ticks
# # majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# # axes.xaxis.set_major_locator(MultipleLocator(xtics))
# # axes.xaxis.set_major_formatter(majorFormatter)
# # axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
# ######################################################

# # script_name = os.path.basename(__file__)
# # figurename = script_name[0:-3] + ".pdf"

# plt.show()
# # fig.savefig(figurename, bbox_inches = "tight")
# # plt.close()


# ## Integral -t / tau plot /////////////////////////////////////////////////////#

# condObject = Conductivity(bandObject, Bamp=45, Bphi=0, Btheta=0, gamma_0 = 25, gamma_k = 70, power = 12)
# condObject.solveMovementFunc()
# condObject.figOnekft()
# t_test = condObject.t
# integral_test = condObject.tOverTauFunc(condObject.kft[:, 0, :], condObject.vft[:, 0, :])

# fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
# fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95)

# #############################################
# fig.text(0.79,0.22, r"$\tau_{\rm N}$ = " + "{0:.3f}".format(condObject.tau_0) + " ps", ha = "right")
# #############################################

# ## Allow to shift the label ticks up or down with set_pad
# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# line = axes.plot(t_test, integral_test)
# plt.setp(line, ls ="-", c = '#5B22FF', lw = 3, marker = "", mfc = '#5B22FF', ms = 7, mec = '#5B22FF', mew= 0)

# #############################################
# axes.set_xlim(0, None)
# axes.set_ylim(0, None)
# axes.set_xlabel(r"t ( ps )", labelpad = 8)
# axes.set_ylabel(r"$\int_{0}^{t}$ ${\rm \dfrac{dt'}{\tau ( t' )}}$", labelpad = 8)
# #############################################

# ## Set ticks space and minor ticks space ############
# #####################################################.

# # xtics = 30 # space between two ticks
# # mxtics = xtics / 2.  # space between two minor ticks
# # majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# # axes.xaxis.set_major_locator(MultipleLocator(xtics))
# # axes.xaxis.set_major_formatter(majorFormatter)
# # axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
# ######################################################

# # script_name = os.path.basename(__file__)
# # figurename = script_name[0:-3] + ".pdf"

# plt.show()
# # fig.savefig(figurename, bbox_inches = "tight")
# # plt.close()
