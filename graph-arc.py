import numpy as np
from numpy import pi
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def isInFBZAF(kx, ky):

    d1 = ky - kx - pi
    d2 = ky - kx + pi
    d3 = ky + kx - pi
    d4 = ky + kx + pi

    return np.logical_and((d1 <= 0)*(d2 >= 0), (d3 <= 0)*(d4 >= 0))

mesh_xy_rough = 100
kx_a = np.linspace(-pi, pi, mesh_xy_rough)
ky_a = np.linspace(-pi, pi, mesh_xy_rough)
kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='xy')

kx_f = kxx.flatten()
ky_f = kyy.flatten()



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



fig, axes = plt.subplots(1, 1, figsize=(5.6, 5.6))
fig.subplots_adjust(left=0.24, right=0.87, bottom=0.29, top=0.91)

in_fbz = isInFBZAF(kx_f, ky_f)

color = "r"
line = axes.plot(kx_f[in_fbz], ky_f[in_fbz])
plt.setp(line, ls="", c=color, lw=3, marker="o",
        mfc=color, ms=3, mec=color, mew=0)
color = "k"
line = axes.plot(kx_f[np.logical_not(in_fbz)], ky_f[np.logical_not(in_fbz)])
plt.setp(line, ls="", c=color, lw=3, marker="o",
         mfc=color, ms=3, mec=color, mew=0)

axes.set_xlim(-pi, pi)
axes.set_ylim(-pi, pi)
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$k_{\rm x}$", labelpad=8)
axes.set_ylabel(r"$k_{\rm y}$", labelpad=8)

axes.set_xticks([-pi, 0., pi])
axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
axes.set_yticks([-pi, 0., pi])
axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

plt.show()
#//////////////////////////////////////////////////////////////////////#
