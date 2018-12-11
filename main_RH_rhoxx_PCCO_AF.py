import time
from numpy import pi
from copy import deepcopy

from band import *
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

a = 3.95e-10 # m
c = 12.07e-10 # m

Bmin = 0.1
Bmax = 10
Bstep = 0.2
B_array = np.arange(Bmin, Bmax, Bstep)



## TWO BAND  ////////////////////////////////////////////////////////////////#
hPocket = Pocket(mu=-0.2, t=250, tp=-0.17, tpp=0.08, tz=0, tz2=0, a=a*1e10, b=a*1e10, M = 0.1, mesh_ds = pi / 40, numberOfKz = 1)
# # hPocket.tp = -0.24
ePocket = deepcopy(hPocket)
ePocket.electronPocket=True
# # setMuToDoping([hPocket,ePocket],0.15,muStart=-0.9)
# hPocket.mu = ePocket.mu = -0.2
# hPocket.figMultipleFS2D()
# ePocket.figMultipleFS2D()

# doping([hPocket, ePocket])

# ## Discretize >>>>>>>>>>>>>>>>>>>>>>>#
# ePocket.mesh_ds=pi/30
hPocket.discretize_FS()
hPocket.densityOfState()
hPocket.doping()
# ePocket.mesh_ds=pi/30
ePocket.discretize_FS()
ePocket.densityOfState()
ePocket.doping()


rhoxx_array = np.empty_like(B_array, dtype = np.float64)
rhoxy_array = np.empty_like(B_array, dtype = np.float64)
RH_array = np.empty_like(B_array, dtype = np.float64)
for i, B in enumerate(B_array):

    hPocketCondObject = Conductivity(hPocket, Bamp=B, gamma_0=15, gamma_k=0, power=12, gamma_dos=0)
    hPocketCondObject.solveMovementFunc()
    hPocketCondObject.chambersFunc(0, 0)
    hPocketCondObject.chambersFunc(0, 1)
    ePocketCondObject = Conductivity(ePocket, Bamp=B, gamma_0=15, gamma_k=0, power=12, gamma_dos=0)
    ePocketCondObject.solveMovementFunc()
    ePocketCondObject.chambersFunc(0, 0)
    ePocketCondObject.chambersFunc(0, 1)
    sigma_xx = hPocketCondObject.sigma[0,0] + ePocketCondObject.sigma[0,0]
    sigma_xy = hPocketCondObject.sigma[0,1] + ePocketCondObject.sigma[0,1]

    rhoxx = sigma_xx / ( sigma_xx**2 + sigma_xy**2) # Ohm.m
    rhoxy = sigma_xy / ( sigma_xx**2 + sigma_xy**2) # Ohm.m
    RH = rhoxy / B # m^3/C

    rhoxx_array[i] = rhoxx
    rhoxy_array[i] = rhoxy
    RH_array[i] = RH





## RH plot ////////////////////////////////////////////////////////////////////#
fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95)

axes.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

#############################################
# fig.text(0.79,0.86, r"$p$ = " + "{0:.3f}".format(bandObject.p), ha = "right")
# fig.text(0.79,0.80, r"$n$ = " + "{0:.3f}".format(n) , ha = "right")
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
# fig.text(0.79,0.28, r"$p$ = " + "{0:.3f}".format(bandObject.p), ha = "right")
# fig.text(0.79,0.22, r"$n$ = " + "{0:.3f}".format(n) , ha = "right")
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