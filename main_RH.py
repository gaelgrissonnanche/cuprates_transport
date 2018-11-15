import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter

from band import BandStructure
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

e = 1.6e-19 # C
a = 3.74e-10 #m
c = 13.3e-10 # m

B_array = np.arange(1, 150, 10)
RH_array = np.empty_like(B_array, dtype = np.float64)

bandObject = BandStructure(mu = -1.25, numberOfKz=71, mesh_ds=np.pi/20)
# bandObject.figMultipleFS2D()

for i, B in enumerate(B_array):

    condObject = Conductivity(bandObject, Bamp=B, Bphi=0, Btheta=0, gamma_0=1, gamma_k=0, power=0)
    condObject.solveMovementFunc()
    condObject.chambersFunc(0, 0)
    condObject.chambersFunc(0, 1)
    sigma_xx = condObject.sigma[0,0]
    sigma_xy = condObject.sigma[0,1]

    rho_xy = sigma_xy / ( sigma_xx**2 + sigma_xy**2)
    RH = rho_xy / B * 1e9 # mm^3/C

    RH_array[i] = RH

nH = 1 / ( RH_array[-1]*1e-9 * e )
d = c / 2
V = a**2 * d
n = V * np.abs(nH)
p = 1 - n
print("1 - n = ", np.round(p, 3) )

## Plot Parameters #################################

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure

fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

#############################################
fig.text(0.79,0.86, r"$p$ = " + "{0:.3f}".format(bandObject.p), ha = "right")
fig.text(0.79,0.80, r"$1 - n$ = " + "{0:.3f}".format(p) , ha = "right")
#############################################

#############################################
# axes.set_xlim(0,110)   # limit for xaxis
# axes.set_ylim(-3.5,1.2) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$H$ ( T )", labelpad = 8)
axes.set_ylabel(r"$R_{\rm H}$ ( mm$^3$ / C )", labelpad = 8)
#############################################

## Allow to shift the label ticks up or down with set_pad
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)


line = axes.plot(B_array, RH_array)
plt.setp(line, ls ="-", c = '#46FFA1', lw = 3, marker = "o", mfc = '#46FFA1', ms = 7, mec = '#46FFA1', mew= 0)
######################################################

######################################################
# plt.legend(loc = 0, fontsize = 12, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.1)
######################################################


## Set ticks space and minor ticks space ############
#####################################################.

xtics = 30 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
# ytics = 1
# mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

# axes.yaxis.set_major_locator(MultipleLocator(ytics))
# axes.yaxis.set_major_formatter(majorFormatter)
# axes.yaxis.set_minor_locator(MultipleLocator(mytics))

######################################################

# script_name = os.path.basename(__file__)
# figurename = script_name[0:-3] + ".pdf"

plt.show()
# fig.savefig(figurename, bbox_inches = "tight")
# plt.close()











