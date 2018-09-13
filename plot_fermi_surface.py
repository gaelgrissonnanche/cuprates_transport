# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import sqrt, exp, cos, sin, log, pi
from scipy.optimize import brentq, approx_fprime
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
import pickle
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Constant //////
hbar = 1.05e34
e = 1.6e19
m0 = 9.1e31
kB = 1.38e23
jtoev = 6.242e18

## Parameters //////
c = 13.2
d = c / 2
a = 5.3 / sqrt(2)
b = 5.3 / sqrt(2)

mu = 605 # VHs = 600
t = 525
tp = -115
tpp = 35
tz = 11

mesh = 1000

## Band structure /////////////////////////////////////////////////////////////#
def e_2D_func(kx, ky, mu, a, t, tp, tpp):
    e_2D = -mu + 2 * t * ( cos(kx*a) + cos(ky*a) ) + 4 * tp * cos(kx*a) * cos(ky*a) + 2 * tpp * ( cos(2*kx*a) + cos(2*ky*a) )
    return e_2D

def e_z_func(kx, ky, kz, tz, a, d):
    sigma = cos(kx*a/2) * cos(ky*a/2)
    e_z = 2 * tz * sigma * ( cos(kx*a) - cos(ky*a) )**2 * cos(kz*d)
    return e_z

def e_3D_func(kx, ky, kz, mu, a, d, t, tp, tpp, tz):
    e_3D = e_2D_func(kx, ky, mu, a, t, tp, tpp) + \
           e_z_func(kx, ky, kz, tz, a, d)
    return e_3D

def e_3D_func_radial(r, theta, kz, mu, a, d, t, tp, tpp, tz):
    kx = r * cos(theta)
    ky = r * sin(theta)
    return e_3D_func(kx, ky, kz, mu, a, d, t, tp, tpp, tz)

def e_3D_func_for_gradient(k, mu, a, d, t, tp, tpp, tz):
    kx = k[0]
    ky = k[1]
    kz = k[2]
    return e_3D_func(kx, ky, kz, mu, a, d, t, tp, tpp, tz)

## Meshgrid ///////////////////////////////////////////////////////////////////#
kx = np.linspace(-pi/a, pi/a, mesh)
ky = np.linspace(-pi/b, pi/b, mesh)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

## Create partial functions
e_3D_func_p = partial(e_3D_func, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)
e_3D_func_radial_p = partial(e_3D_func_radial, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)
e_3D_func_for_gradient_p = partial(e_3D_func_for_gradient, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)

## Discretize the Fermi surface ///////////////////////////////////////////////#
kzf_a = np.linspace(-pi / c, pi / c, 11)
theta_a = np.linspace(0, 2 * pi, 101)
kxf_a = np.empty((theta_a.shape[0], kzf_a.shape[0]))
kyf_a = np.empty((theta_a.shape[0], kzf_a.shape[0]))
ef_3D = np.empty((theta_a.shape[0], theta_a.shape[0], kzf_a.shape[0]))

for j, kzf in enumerate(kzf_a):
    for i, theta in enumerate(theta_a):

        try:
            rf = brentq(e_3D_func_radial_p, a = 0, b = 0.8, args = (theta, kzf))
            kxf_a[i ,j] = rf * cos(theta)
            kyf_a[i ,j] = rf * sin(theta)

        except ValueError: # in case the Fermi surface is not continuous
            kxf_a[i ,j] = np.NaN
            kyf_a[i ,j] = np.NaN

## Velocity on Fermi surface
for i in range(len(kxf_a)):
    for j in range(len(kyf_a)):
        for l in range(len(kzf_a)):
            k = np.array([kxf_a[i ,l], kyf_a[i ,l], kzf_a[l]])
            ef_3D = approx_fprime(k, e_3D_func_for_gradient_p, epsilon = 1e-4)



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
## Figures ////////////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

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

#///// Create Figure //////#
fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

#///// Allow to shift the label ticks up or down with set_pad /////#
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Labels //////////////////////////////#
# fig.text(0.79,0.86, samplename, ha = "right")
# fig.text(0.83,0.87, r"$T$ /  $H$  /  $\phi$ ", color = 'k', ha = 'left'))

#//// Plots ////////////////////////////////#

line = axes.contour(kxx, kyy, e_3D_func_p(kx = kxx, ky = kyy, kz = - pi / c), 0, colors = '#FF0000', linewidths = 3)
line = axes.plot(kxf_a[:,0], kyf_a[:,0])
plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)  # set properties


axes.set_xlim(-pi/a, pi/a)   # limit for xaxis
axes.set_ylim(-pi/b, pi/b) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)


# ##///Set ticks space and minor ticks space ///#
# xtics = 60 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks

# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
# axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))


axes.locator_params(axis = 'y', nbins = 6)

#//////////////////////////////////////////////////////////////////////////////#
plt.show()

# script_name = os.path.basename(__file__) # donne le nom du script avec l’extension .py
# figurename = script_name[0:-3] + ".pdf"
# fig.savefig("../figures/" + figurename, bbox_inches = "tight")
plt.close()
#//////////////////////////////////////////////////////////////////////////////#