# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import sqrt, exp, cos, sin, log, pi
from scipy.optimize import brentq, approx_fprime
from scipy.integrate import odeint
from scipy import interpolate
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from band_structure import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Constant //////
# hbar = 1.05e34
# e = 1.6e19
# m0 = 9.1e31
# kB = 1.38e23
# jtoev = 6.242e18
e = 1
hbar = 1
m = 1

## Parameters //////
c = 13.2
d = c / 2
a = 5.3 / sqrt(2)
b = 5.3 / sqrt(2)

mu = 805 # VHs = 600
t = 525
tp = -115
tpp = 35
tz = 11

mesh_xy = 51
mesh_z = 15

B_amp = 0.02
B_phi = 0

tau = 1e-3

mesh_B_theta = 2
B_theta_a = np.array([0, pi/4])
# B_theta_a = np.linspace(0, 90 * pi / 180, mesh_B_theta)

dt = 5e-5
tmin = 0
tmax = 10 * tau



## Fermi Surface t = 0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Create partial functions
e_3D_func_p = partial(e_3D_func, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)
e_3D_func_radial_p = partial(e_3D_func_radial, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)
e_3D_func_for_gradient_p = partial(e_3D_func_for_gradient, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)
e_3D_func_for_gradient_p = partial(e_3D_func_for_gradient, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)
v_3D_func_p = partial(v_3D_func, mu = mu, a = a, d = d, t = t, tp = tp, tpp = tpp, tz= tz)

## 1st unregular discretization of the Fermi surface //////////////////////////#
mesh_xy_rough = mesh_xy * 4 + 1 # the rough discretization needs
                                # to be better than the regular one
kzf_a = np.linspace(-pi / c, pi / c, mesh_z)
theta_a = np.linspace(0, 2 * pi, mesh_xy_rough)
kft0_rough = np.empty( (mesh_xy_rough * mesh_z, 3))

## Compute kf at t = 0
for j, kzf in enumerate(kzf_a):
    for i, theta in enumerate(theta_a):

        try:
            rf = brentq(e_3D_func_radial_p, a = 0, b = 0.8, args = (theta, kzf))
            kft0_rough[i + j * mesh_xy_rough, 0] = rf * cos(theta)
            kft0_rough[i + j * mesh_xy_rough, 1] = rf * sin(theta)
            kft0_rough[i + j * mesh_xy_rough, 2] = kzf


        except ValueError: # in case the Fermi surface is not continuous
            kft0_rough[i + j * mesh_xy_rough, 0] = np.NaN
            kft0_rough[i + j * mesh_xy_rough, 1] = np.NaN
            kft0_rough[i + j * mesh_xy_rough, 2] = np.NaN

## Remove k points NaN
index_nan_kx = ~np.isnan(kft0_rough[:,0])
index_nan_ky = ~np.isnan(kft0_rough[:,1])
index_nan_kz = ~np.isnan(kft0_rough[:,2])
index_row_nan_k = index_nan_kx * index_nan_ky * index_nan_kz
kft0_rough = kft0_rough[index_row_nan_k, :]

## 2nd regular discretization of the Fermi surface ////////////////////////////#
kft0 = np.empty( (mesh_xy * mesh_z, 3) )

for j, kzf in enumerate(kzf_a):
    x = kft0_rough[mesh_xy_rough*j: mesh_xy_rough*(j+1), 0]
    y = kft0_rough[mesh_xy_rough*j: mesh_xy_rough*(j+1), 1]

    dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
    r = np.zeros_like(x)
    r[1:] = np.cumsum(dr) # integrate path
    r_int = np.linspace(0, r.max(), mesh_xy) # regular spaced path
    x_int = np.interp(r_int, r, x) # interpolate
    y_int = np.interp(r_int, r, y)

    kft0[mesh_xy*j: mesh_xy*(j+1), 0] = x_int
    kft0[mesh_xy*j: mesh_xy*(j+1), 1] = y_int
    kft0[mesh_xy*j: mesh_xy*(j+1), 2] = kzf

## Compute Velocity at t = 0
vft0 = v_3D_func_p(kft0[:,0], kft0[:,1], kft0[:,2])

## Quasiparticule orbits >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

def B_func(B_amp, B_theta, B_phi):
    B = B_amp * np.array([sin(B_theta)*cos(B_phi), sin(B_theta)*cos(B_phi), cos(B_theta)])
    return B

## Movement equation //#
def diff_func(k, t, B):
    v =  v_3D_func_p(k[0], k[1], k[2])
    dkdt = ( - e / hbar ) * np.cross(v, - B) # (-) represent -t in vz(-t, kt0) in the Chambers formula
                            # integrated from 0 to +infinity
    return dkdt

def resolve_movement_func(B_amp, B_theta, B_phi, kft0):
    # dt = 5e-5
    # tmin = 0
    # tmax = 10 * tau
    t = np.arange(tmin, tmax, dt)
    kft = np.empty( (kft0.shape[0], t.shape[0], 3))
    vft = np.empty( (kft0.shape[0], t.shape[0], 3))
    # kft/vft[index of starting k @ t0 on FS, index of for k @ t from k @ t0, index of (kx,ky,kz) components]

    ## Compute B ////#
    B = B_func(B_amp, B_theta, B_phi)

    ## Compute kf, vf function of t ///#
    for i0 in range(kft0.shape[0]):
        kft[i0, :, :] = odeint(diff_func, kft0[i0, :], t, args = (B,)) # solve differential equation
        vft[i0, :, :] = v_3D_func_p(kft[i0, :, 0], kft[i0, :, 1], kft[i0, :, 2])

    return kft, vft, t


## Conductivity sigma_zz >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

def sigma_zz(vzft0, vzft, kft0, t, tau):

    prefactor = e**2 / ( 4 * pi**3 )

    dt = t[1] - t[0]
    dk_vector = np.diff(kft0, axis = 0) # gives dk vector from one point on the FS to the other
    # but it 0does not work for dkz as kz constant over a 2D slice of FS
    # needs to implement manually the dkz
    dk_vector[:, 2] = np.ones(dk_vector.shape[0]) * ( 2 * pi / c ) / (mesh_z - 1)
    dk = np.prod(dk_vector, axis = 1) # gives the dk volume product of dkx*dky*dkz

    v_product = np.empty(vzft0.shape[0]-1)

    for i0 in range(vzft0.shape[0]-1):
        vz_sum_over_t = np.sum( vzft[i0, :] * exp(- t / tau) * dt ) # integral over t
        v_product[i0] = vzft0[i0] * vz_sum_over_t

    s_zz = - prefactor * np.sum(dk * v_product) # integral over k

    return s_zz

# Function of B_theta
sigma_zz_a = np.empty(B_theta_a.shape[0])

for j, B_theta in enumerate(B_theta_a):

    kft, vft, t = resolve_movement_func(B_amp, B_theta, B_phi, kft0)
    vzft0 = vft0[:,2]
    vzft = vft[:,:,2]
    s_zz = sigma_zz(vzft0, vzft, kft0, t, tau)
    sigma_zz_a[j] = s_zz
    print("theta = " + str(B_theta * 180 / pi) + ", sigma_zz = " + r"{0:.5e}".format(s_zz))

rho_zz_a = 1 / sigma_zz_a





#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
## Figures ////////////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## For figures, compute t-dependence
kft, vft, t = resolve_movement_func(B_amp = B_amp, B_theta = 0, B_phi = 0, kft0 = kft0)


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

mesh_graph = 1000
kx = np.linspace(-pi/a, pi/a, mesh_graph)
ky = np.linspace(-pi/b, pi/b, mesh_graph)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

line = axes.contour(kxx, kyy, e_3D_func_p(kx = kxx, ky = kyy, kz = - pi / c), 0, colors = '#FF0000', linewidths = 3)
line = axes.plot(kft0[: mesh_xy*1, 0], kft0[: mesh_xy*1, 1]) # mesh_xy means all points for kz = - pi / c
plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
axes.quiver(kft0[: mesh_xy*1, 0], kft0[: mesh_xy*1, 1], vft0[: mesh_xy*1, 0], vft0[: mesh_xy*1, 1]) # mesh_xy means all points for kz = - pi / c

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
line = axes.plot(kft0[0, 0], kft0[0, 1])


plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'b', ms = 8, mec = "#7E2320", mew= 0)  # set properties
line = axes.plot(kft[0,:, 0], kft[0,:, 1])
plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)  # set properties

axes.set_xlim(-pi/a, pi/a)   # limit for xaxis
axes.set_ylim(-pi/b, pi/b) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

axes.locator_params(axis = 'y', nbins = 6)

plt.show()

#>>>> vf vs t >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y = 0, ls ="--", c ="k", linewidth = 0.6)

#///// Allow to shift the label ticks up or down with set_pad /////#
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Labels //////#
# fig.text(0.79,0.86, samplename, ha = "right")

# line = axes.plot(t, vft[0,:, 0])
# plt.setp(line, ls ="-", c = 'r', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)
# line = axes.plot(t, vft[0,:, 1])
# plt.setp(line, ls ="-", c = 'b', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)
line = axes.plot(t, vft[0,:, 2])
plt.setp(line, ls ="-", c = 'g', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)

# axes.set_xlim(0, 90)   # limit for xaxis
# axes.set_ylim(ymin, ymax) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$t$", labelpad = 8)
axes.set_ylabel(r"$v_{\rm z}$", labelpad = 8)

axes.locator_params(axis = 'y', nbins = 6)

plt.show()
#//////////////////////////////////////////////////////////////////////////////#

#>>>> Cumulated velocity vs t >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

#///// Allow to shift the label ticks up or down with set_pad /////#
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Labels //////#
# fig.text(0.79,0.86, samplename, ha = "right")

line = axes.plot(t, np.cumsum( vft[0,:, 2] * exp ( -t / tau ) ))
plt.setp(line, ls ="-", c = 'k', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
line = axes.plot(t, np.cumsum( vft[1,:, 2] * exp ( -t / tau ) ))
plt.setp(line, ls ="-", c = 'r', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
line = axes.plot(t, np.cumsum( vft[2,:, 2] * exp ( -t / tau ) ))
plt.setp(line, ls ="-", c = 'b', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties


# axes.set_xlim(0, 90)   # limit for xaxis
# axes.set_ylim(ymin, ymax) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$t$", labelpad = 8)
axes.set_ylabel(r"cum sum velocity", labelpad = 8)

axes.locator_params(axis = 'y', nbins = 6)

plt.show()
#//////////////////////////////////////////////////////////////////////////////#

#>>>> Rzz vs theta >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

#///// Allow to shift the label ticks up or down with set_pad /////#
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Labels //////#
# fig.text(0.79,0.86, samplename, ha = "right")

line = axes.plot(B_theta_a * 180 / pi, rho_zz_a / rho_zz_a[0])
plt.setp(line, ls ="-", c = 'k', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties

axes.set_xlim(0, 110)   # limit for xaxis
# axes.set_ylim(ymin, ymax) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)


##///Set ticks space and minor ticks space ///#
xtics = 30 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks

majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

axes.locator_params(axis = 'y', nbins = 6)

plt.show()
#//////////////////////////////////////////////////////////////////////////////#


#//////////////////////////////////////////////////////////////////////////////#
plt.close()
#//////////////////////////////////////////////////////////////////////////////#




# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.quiver(kf[:,0], kf[:,1], kf[:,2], vf[:,0], vf[:,1], vf[:,2]*10, length=0.00002)

# plt.show()