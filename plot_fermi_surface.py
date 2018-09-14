# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import sqrt, exp, cos, sin, log, pi
from scipy.optimize import brentq, approx_fprime
from scipy.integrate import odeint
from functools import partial
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
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

mesh_xy = 101
mesh_z = 11

B_amp = 1
B_theta = 0
B_phi = 0

B = B_amp * np.array([sin(B_theta)*cos(B_phi), sin(B_theta)*cos(B_phi), cos(B_theta)])

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

## Create partial functions
e_3D_func_p = partial(e_3D_func, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)
e_3D_func_radial_p = partial(e_3D_func_radial, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)
e_3D_func_for_gradient_p = partial(e_3D_func_for_gradient, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)

## Discretize the Fermi surface ///////////////////////////////////////////////#
kzf_a = np.linspace(-pi / c, pi / c, mesh_z)
theta_a = np.linspace(0, 2 * pi, mesh_xy)

kf = np.empty( (mesh_xy * mesh_z, 3))


for j, kzf in enumerate(kzf_a):
    for i, theta in enumerate(theta_a):

        try:
            rf = brentq(e_3D_func_radial_p, a = 0, b = 0.8, args = (theta, kzf))
            kf[i + j * mesh_xy, 0] = rf * cos(theta)
            kf[i + j * mesh_xy, 1] = rf * sin(theta)
            kf[i + j * mesh_xy, 2] = kzf

        except ValueError: # in case the Fermi surface is not continuous
            kf[i + j * mesh_xy, 0] = np.NaN
            kf[i + j * mesh_xy, 1] = np.NaN
            kf[i + j * mesh_xy, 2] = np.NaN

## Remove k points NaN
index_nan_kx = ~np.isnan(kf[:,0])
index_nan_ky = ~np.isnan(kf[:,1])
index_nan_kz = ~np.isnan(kf[:,2])

index_row_nan_k = index_nan_kx * index_nan_ky * index_nan_kz
kf = kf[index_row_nan_k, :]

## Compute Velocity ///////////////////////////////////////////////////////////#
vf = np.empty( (kf.shape[0], 3))
for i in range(kf.shape[0]):
    vf[i,:] = approx_fprime(kf[i,:], e_3D_func_for_gradient_p, epsilon = 1e-6)

## Solve movment equation /////////////////////////////////////////////////////#

def diff_func(k, t, B):
    v = approx_fprime(k, e_3D_func_for_gradient_p, epsilon = 1e-6)
    dkdt = np.cross(v, B)
    return dkdt

t = np.linspace(0, 0.002, 100)
k0 = np.array([kf[0, 0], kf[0, 1], kf[0, 2]])
kt = odeint(diff_func, k0, t, args = (B,))


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


##>>>> 2D Fermi Surface >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

# fig.text(0.79,0.86, samplename, ha = "right")
# fig.text(0.83,0.87, r"$T$ /  $H$  /  $\phi$ ", color = 'k', ha = 'left'))

mesh_graph = 100
kx = np.linspace(-pi/a, pi/a, mesh_graph)
ky = np.linspace(-pi/b, pi/b, mesh_graph)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

line = axes.contour(kxx, kyy, e_3D_func_p(kx = kxx, ky = kyy, kz = - pi / c), 0, colors = '#FF0000', linewidths = 3)
line = axes.plot(kf[: mesh_xy*1, 0], kf[: mesh_xy*1, 1])
plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)  # set properties
axes.quiver(kf[: mesh_xy*1, 0], kf[: mesh_xy*1, 1], vf[: mesh_xy*1, 0], vf[: mesh_xy*1, 1])

axes.set_xlim(-pi/a, pi/a)   # limit for xaxis
axes.set_ylim(-pi/b, pi/b) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

axes.locator_params(axis = 'y', nbins = 6)

plt.show()
#//////////////////////////////////////////////////////////////////////////////#


#>>>> k vs t >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

# fig.text(0.79,0.86, samplename, ha = "right")
# fig.text(0.83,0.87, r"$T$ /  $H$  /  $\phi$ ", color = 'k', ha = 'left'))

mesh_graph = 100
kx = np.linspace(-pi/a, pi/a, mesh_graph)
ky = np.linspace(-pi/b, pi/b, mesh_graph)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

line = axes.contour(kxx, kyy, e_3D_func_p(kx = kxx, ky = kyy, kz = - pi / c), 0, colors = '#FF0000', linewidths = 3)
line = axes.plot(kf[0, 0], kf[0, 1])
plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'b', ms = 8, mec = "#7E2320", mew= 0)  # set properties
line = axes.plot(kt[:, 0], kt[:, 1])
plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)  # set properties

axes.set_xlim(-pi/a, pi/a)   # limit for xaxis
axes.set_ylim(-pi/b, pi/b) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

axes.locator_params(axis = 'y', nbins = 6)

plt.show()


#//////////////////////////////////////////////////////////////////////////////#
plt.close()
#//////////////////////////////////////////////////////////////////////////////#




# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.quiver(kf[:,0], kf[:,1], kf[:,2], vf[:,0], vf[:,1], vf[:,2]*10, length=0.00002)

# plt.show()