import numpy as np
import scipy as sp
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

## Plotting >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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

# cmap = mpl.cm.get_cmap("viridis", len(files))
# colors = cmap(np.arange(len(files)))
# colors[-1] = (1, 0, 0, 1)

####################################################
## Plot Parameters #################################

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# axes.axhline(y=0, ls ="--", c ="k", linewidth=0.6)

#############################################
# fig.text(0.79,0.23, r"label", ha = "right")
#############################################


##############################################################################
### Load and Plot Data #######################################################
##############################################################################

######################################################
data = np.loadtxt("quick_figure.dat", dtype="float", comments="#", delimiter=',')

x = data[:,0]
y = data[:,1]

# x_interp = np.arange(x[0],x[-1],0.01)
# y_interp = interpolate.UnivariateSpline(x, y, k = 2, s = 0.1)(x_interp)

color = '#FF0000'
# color = '#00EFD0'
line = axes.plot(x, y, label = r"curve 1")
plt.setp(line, ls ="-", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
######################################################


######################################################
axes.legend(loc = 3, fontsize = 12, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
# axes.legend(bbox_to_anchor=(0.5, 0.5), loc = 1, fontsize = 12, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################


#############################################
# axes.set_xlim(0,90)   # limit for xaxis
# axes.set_ylim(-3.5,1.2) # leave the ymax auto, but fix ymin
# axes.set_xscale('log')
# axes.set_yscale('log')
axes.set_xlabel(r"x", labelpad = 8)
axes.set_ylabel(r"y", labelpad = 8)
axes.tick_params(axis='x', which='major', pad=7, length=8)
axes.tick_params(axis='y', which='major', pad=8, length=8)
axes.tick_params(axis='x', which='minor', pad=7, length=4)
axes.tick_params(axis='y', which='minor', pad=8, length=4)
#############################################

## Set ticks space and minor ticks space ############
#####################################################.

# xtics = 30 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks
# ytics = 1
# mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

# axes.yaxis.set_major_locator(MultipleLocator(ytics))
# axes.yaxis.set_major_formatter(majorFormatter)
# axes.yaxis.set_minor_locator(MultipleLocator(mytics))

######################################################

script_name = os.path.basename(__file__)
figurename = script_name[0:-3] + ".pdf"

plt.show()
fig.savefig(figurename, bbox_inches = "tight")
plt.close()

