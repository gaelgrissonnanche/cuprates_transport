import numpy as np
from numpy import pi, cos, sin
import scipy as sp
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import plotly.graph_objects as go
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


a = 3.83 # angstrom
c = 23.20 # angstrom
d = c / 2

k00 = 7.45 / 10 # in angstrom^-1
k40 = -0.19 / 10 # in angstrom^-1
k21 = 0.031 / 10 # in angstrom^-1
k61 = 0.021 / 10 # in angstrom^-1
k101 = -0.0085 / 10 # in angstrom^-1
exagerate = 4

phi_array = np.linspace(0, 2*pi, 100)
kz_array = np.linspace(0, 2*pi/c, 3)

# phii, kzz = np.meshgrid(phi, kz, indexing = 'ij')
# kF = k00 + k40*cos(4*phii) + cos(kzz*d) * (k21*sin(2*phii) + k61*sin(6*phii) + k101*sin(10*phi))


## Plot ////////////////////////////////////////////////////////////////////////
mpl.rcdefaults()
mpl.rcParams['font.size'] = 24.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 24.
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6
mpl.rcParams['axes.linewidth'] = 0.6
mpl.rcParams['pdf.fonttype'] = 3

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

cmap = mpl.cm.get_cmap("viridis", len(kz_array))
colors = cmap(np.arange(len(kz_array)))
colors[-1] = (1, 0, 0, 1)

axes.axhline(y = pi, ls ="--", c ="k", linewidth = 0.6)
axes.axvline(x = pi, ls ="--", c ="k", linewidth = 0.6)

legend_list = [r"$k_{\rm z}$ = 0", r"$k_{\rm z}$ = $\pi$/c", r"$k_{\rm z}$ = 2$\pi$/c"]

for i, kz in enumerate(kz_array):
    kF = k00 + k40*cos(4*phi_array) + exagerate * cos(kz*d) * (k21*sin(2*phi_array) + k61*sin(6*phi_array) + k101*sin(10*phi_array))
    kF_x = kF * cos(phi_array)
    kF_y = kF * sin(phi_array)

    line = axes.plot(pi + kF_x*a, pi + kF_y*a, label = legend_list[i])
    plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)


plt.legend(bbox_to_anchor = (0.15,0.78), loc = 2, fontsize = 12, frameon = False, numpoints = 1, markerscale = 1, handletextpad = 0.5)

axes.set_aspect(aspect=1)
axes.set_xlim(0, 2 * pi)
axes.set_ylim(0, 2 * pi)
axes.set_xticks([0, pi, 2*pi])
axes.set_xticklabels([r"0", r"$\pi$", r"$2\pi$"])
axes.set_yticks([0, pi, 2*pi])
axes.set_yticklabels([r"0", r"$\pi$", r"$2\pi$"])
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$k_{\rm y}$")
#############################################

######################################################
script_name = os.path.basename(__file__) # donne le nom du script avec l’extension .py
figurename = script_name[0:-3] + ".png"

plt.show()
# fig.savefig("sim/Tl2201_Tc_20K/" + figurename, bbox_inches = "tight")
# plt.close()
######################################################

