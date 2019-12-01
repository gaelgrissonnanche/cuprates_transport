import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#



## ONE BAND Horio et al. ///////////////////////////////////////////////////////
params = {
    "bandname": "LargePocket",
    "a": 3.74767,
    "b": 3.74767,
    "c": 13.2,
    "t": 190,
    "tp": -0.14,
    "tpp": 0.07,
    "tz": 0.07,
    "tz2": 0.00,
    "mu": -0.826,
    "fixdoping": 0.24,
    "numberOfKz": 61,
    "mesh_ds": 1/100,
    "T" : 0,
}

## Bandstructure
bandObject = BandStructure(**params)

tz_array = np.array([0, 0.03, 0.07])
mu_array = np.linspace(-0.95, -0.75, 500)
p_matrix = np.empty((len(tz_array), len(mu_array)))
dos_epsilon_matrix = np.empty((len(tz_array), len(mu_array)))

k = 1
for i, tz in enumerate(tz_array):
    bandObject.tz = tz

    for j, mu in enumerate(mu_array):
        # bandObject.setMuToDoping(p)
        bandObject.mu = mu
        bandObject.doping()
        print(str(k) + " / " + str(len(tz_array) * len(mu_array)))
        bandObject.discretize_FS()#(mesh_xy_rough=1001)
        bandObject.dos_k_func()
        bandObject.dos_epsilon_func()

        p_matrix[i,j] = bandObject.p
        dos_epsilon_matrix[i,j] = bandObject.dos_epsilon

        k+=1

    index = np.argsort(p_matrix[i,:])
    p_matrix[i, :] = p_matrix[i, index]
    dos_epsilon_matrix[i, :] = dos_epsilon_matrix[i, index]




##############################################################################
### Plotting #################################################################
##############################################################################

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

figures_path = "sim/NdLSCO_0p25/" + os.path.basename(__file__)[0:-3] + ".pdf"

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure

fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size


#############################################
# axes.set_xlim(0,100)
# axes.set_ylim(-4.5,0.5)
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$p$", labelpad = 8)
axes.set_ylabel(r"DOS ( eV$^{-1}$ )")
#############################################

## Color map
cmap = mpl.cm.get_cmap("viridis", len(tz_array))
colors = cmap(np.arange(len(tz_array)))
colors[-1] = (1, 0, 0, 1)



### Load and Plot Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

for i, tz in enumerate(tz_array):
    line = axes.plot(p_matrix[i,:], dos_epsilon_matrix[i,:] * 1e3, label=r"$t_{\rm z}$ = " + "{:.2f}".format(tz))
    plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)


plt.legend(loc = 0, fontsize = 16, frameon = False, numpoints=1, markerscale=1.0, handletextpad=0.5)

# ## Set ticks space and minor ticks space ############
# #####################################################.

# xtics = 25 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks
# ytics = 2
# mytics = ytics / 2. # or "AutoMinorLocator(2)" if ytics is not fixed, just put 1 minor tick per interval
# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

# axes.yaxis.set_major_locator(MultipleLocator(ytics))
# axes.yaxis.set_major_formatter(majorFormatter)
# axes.yaxis.set_minor_locator(MultipleLocator(mytics))

######################################################


plt.show()
fig.savefig(figures_path, bbox_inches = "tight")
plt.close()
