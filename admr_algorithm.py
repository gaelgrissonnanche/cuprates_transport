# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import sqrt, exp, cos, sin, log, pi, ones
from scipy.optimize import brentq
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from numba import jit, prange
import time
from skimage import measure

from band_structure import *
from diff_equation import *
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
# c = 13.2
# d = c / 2
# a = 5.3 / sqrt(2)
# b = 5.3 / sqrt(2)

# mu = 805 # VHs = 600
# t = 525
# tp = -115
# tpp = 35
# tz = 11

# tau = 1e-3

c = 1
d = c / 2
a = 1
b = 1

# t   =  1.
# tp  = -0.21 * t
# tpp =  0.066 * t
# tz  =  0.020 * t
# mu  = 1.53
# tau =  25

t   =  1.
tp  = -0.21 * t
tpp =  0.066 * t
tz  =  0.020 * t
mu  = 1.53 * t
tau =  25


divided_FS_by = 1 # number of time the Fermi surface has been divided by
mesh_xy = 32 # must be a multiple of 4
mesh_z = 11 # 11 ideal to be fast and accurate

B_amp = 0.02
B_phi = 15 * pi / 180

mesh_B_theta = 31
B_theta_a = np.linspace(0, 180 * pi / 180, mesh_B_theta)


tmin = 0
tmax = 10 * tau
dt = tmax / 200


## Fermi Surface t = 0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Make mesh_xy a multiple of 4 to preserve symmetry
if mesh_xy % 4 != 0:
    mesh_xy = mesh_xy - (mesh_xy % 4)

band_parameters = np.array([mu, a, d, t, tp, tpp, tz])

start_time_FS = time.time()

# mesh_xy_rough = mesh_xy*10 + 1 # higher it is the rough, better is the interpolated one

# kft0 = np.empty( (mesh_xy * mesh_z, 3) )

# kx = np.linspace(-pi/a, pi/a, mesh_xy_rough)
# ky = np.linspace(-pi/b, pi/b, mesh_xy_rough)
# kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

# kzf_a = np.linspace(-pi / c, pi / c, mesh_z)

# for j, kzf in enumerate(kzf_a):
#     bands = e_3D_func(kxx, kyy, kzf, band_parameters)
#     contour = measure.find_contours(bands, 0)[0]
#     # for contour in contours:

#     # Contour in units proportionnal to size of meshgrid
#     x_raw = contour[:, 0]
#     y_raw = contour[:, 1]

#     # Scale the contour to units of kx and ky
#     x = (x_raw/(mesh_xy_rough-1)-0.5)*2*pi/a
#     y = (y_raw/(mesh_xy_rough-1)-0.5)*2*pi/b

#     # Make the contour start at a high point of symmetry, for example for ky = 0
#     index_xmax = np.argmax(x) # find the index of the first maximum of x
#     x = np.roll(x, x.shape - index_xmax) # roll the elements to get maximum of x first
#     y = np.roll(y, x.shape - index_xmax) # roll the elements to get maximum of x first

#     # Closed contour?
#     if x[1] == x[-1]: # meaning a closed contour
#         x = np.append(x, x[0]) # add the first element to get a closed contour
#         y = np.append(y, y[0]) # to calculate the total length
#         mesh_xy = mesh_xy - (mesh_xy % 4) # respects the 4-order symmetry

#     ds = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
#     s = np.zeros_like(x) # arrays of zeros
#     s[1:] = np.cumsum(ds) # integrate path, s[0] = 0

#     s_int = np.linspace(0, s.max(), mesh_xy + 1) # regular spaced path
#     x_int = np.interp(s_int, s, x)[:-1] # interpolate
#     y_int = np.interp(s_int, s, y)[:-1]

#     kft0[mesh_xy*j: mesh_xy*(j+1), 0] = x_int
#     kft0[mesh_xy*j: mesh_xy*(j+1), 1] = y_int
#     kft0[mesh_xy*j: mesh_xy*(j+1), 2] = kzf






## 1st unregular discretization of the Fermi surface //////////////////////////#
mesh_xy_rough = mesh_xy * 100 + 1 # higher it is the rough, better is the interpolated one

kzf_a = np.linspace(-pi / c, pi / c, mesh_z)
phi_a = np.linspace(0, 2 / divided_FS_by * pi, mesh_xy_rough)
kft0_rough = np.empty( (mesh_xy_rough * mesh_z, 3) )

## Compute kf at t = 0
for j, kzf in enumerate(kzf_a):
    for i, phi in enumerate(phi_a):

        try:
            rf = brentq(e_3D_func_radial, a = 0, b = 3.14, args = (phi, kzf, band_parameters))
            kft0_rough[i + j * mesh_xy_rough, 0] = rf * cos(phi)
            kft0_rough[i + j * mesh_xy_rough, 1] = rf * sin(phi)
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
    ds = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
    s = np.zeros_like(x) # arrays of zeros
    s[1:] = np.cumsum(ds) # integrate path, s[0] = 0

    s_int = np.linspace(0, s.max(), mesh_xy + 1) # regular spaced path
    x_int = np.interp(s_int, s, x)[:-1] # interpolate
    y_int = np.interp(s_int, s, y)[:-1]

    kft0[mesh_xy*j: mesh_xy*(j+1), 0] = x_int
    kft0[mesh_xy*j: mesh_xy*(j+1), 1] = y_int
    kft0[mesh_xy*j: mesh_xy*(j+1), 2] = kzf


## Integration Delta
dkft0 = 1 / (mesh_xy * mesh_z) * ( 2 * pi )**3 / ( a * b * c )

## Compute Velocity at t = 0
vx, vy, vz = v_3D_func(kft0[:,0], kft0[:,1], kft0[:,2], band_parameters)
vft0 = np.array([vx, vy, vz]).transpose()

print("Discretize FS time : %.6s seconds" % (time.time() - start_time_FS))

## Solve differential equation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

@jit(nopython=True)
def solve_movement_func(B_amp, B_theta, B_phi, kft0, band_parameters):
    t = np.arange(tmin, tmax, dt)
    kft = np.empty( (kft0.shape[0], t.shape[0], 3))
    vft = np.empty( (kft0.shape[0], t.shape[0], 3))
    # kft/vft[index of starting k @ t0 on FS, index of for k @ t from k @ t0, index of (kx,ky,kz) components]

    ## Compute B ////#
    B = B_func(B_amp, B_theta, B_phi)

    # start_time_odeint = time.time()
    # ## Compute kf, vf function of t ///#
    # for i0 in range(kft0.shape[0]):
    #     kft[i0, :, :] = odeint(diff_func, kft0[i0, :], t, args = (B, band_parameters)) # solve differential equation
    #     vx, vy, vz = v_3D_func(kft[i0, :, 0], kft[i0, :, 1], kft[i0, :, 2], band_parameters)
    #     vft[i0, :, :] = np.array([vx, vy, vz]).transpose()
    # print("Odeint time : %.6s seconds" % (time.time() - start_time_odeint))

    # start_time_rgk4 = time.time()
    kftp = rgk4_algorithm(kft0, t, B, band_parameters)
    vftp = np.empty(kftp.shape)
    vftp[:,:,0], vftp[:,:,1], vftp[:,:,2] = v_3D_func(kftp[:, :, 0], kftp[:, :, 1], kftp[:, :, 2], band_parameters)
    # print("Rgk4 time : %.6s seconds" % (time.time() - start_time_rgk4))

    kft = kftp
    vft = vftp

    return kft, vft, t, kftp, vftp


## Conductivity sigma_zz >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

@jit(nopython=True)
def sigma_zz(vft0, vzft, kft0, dkft0, t, tau):

    prefactor = e**2 / ( 4 * pi**3 )

    # Time increment
    dt = t[1] - t[0]
    # Density of State
    dos = hbar * sqrt( vft0[:,0]**2 + vft0[:,1]**2 + vft0[:,2]**2 )

    vzft0 = vft0[:,2]

    v_product = np.empty(vzft0.shape[0])
    for i0 in prange(vzft0.shape[0]):
        vz_sum_over_t = np.sum( ( 1 / dos[i0] ) * vzft[i0, :] * exp(- t / tau) * dt ) # integral over t
        v_product[i0] = vzft0[i0] * vz_sum_over_t # integral over z

    s_zz = divided_FS_by * prefactor * np.sum(dkft0 * v_product) # integral over k

    return s_zz

# Function of B_theta
sigma_zz_a = np.empty(B_theta_a.shape[0])

for j, B_theta in enumerate(B_theta_a):

    start_time = time.time()

    kft, vft, t, kftp, vftp = solve_movement_func(B_amp, B_theta, B_phi, kft0, band_parameters)

    s_zz = sigma_zz(vft0, vft[:,:,2], kft0, dkft0, t, tau)
    sigma_zz_a[j] = s_zz

    print("theta = " + str(B_theta * 180 / pi) + ", sigma_zz = " + r"{0:.5e}".format(s_zz))
    print("Calculation time : %.6s seconds" % (time.time() - start_time))


rho_zz_a = 1 / sigma_zz_a
rho_zz_0 = rho_zz_a[0]

## Save Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
len = rho_zz_a.shape[0]
Data = np.vstack((B_theta_a, rho_zz_a / rho_zz_0, B_amp*ones(len), tau*ones(len), mu*ones(len), 1*ones(len), tp*ones(len), tpp*ones(len), tz*ones(len), mesh_xy*ones(len), mesh_z*ones(len)))
Data = Data.transpose()
folder = "../data_sim/"
file_name =  "Rzz" + "_" + str(B_amp) + "_" + str(tau) + "_" + str(mu) + "_" + str(tp) + "_" + str(tpp) + "_" + str(tz) + ".dat"
np.savetxt(folder + file_name, Data, fmt='%.7e', header = "theta[deg]\trhozz(theta)/rhozz(0)\tB\ttau\tmu\tt\ttp\ttpp\ttz\tmesh_xy\tmesh_z", comments = "#")

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
## Figures ////////////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## For figures, compute t-dependence
kft, vft, t, kftp, vftp = solve_movement_func(B_amp, 0, 0, kft0, band_parameters)

mesh_graph = 1001
kx = np.linspace(-pi/a, pi/a, mesh_graph)
ky = np.linspace(-pi/b, pi/b, mesh_graph)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

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

line = axes.contour(kxx, kyy, e_3D_func(kxx, kyy, - pi / c, band_parameters), 0, colors = '#FF0000', linewidths = 3)
line = axes.plot(kft0[: mesh_xy*1, 0], kft0[: mesh_xy*1, 1]) # mesh_xy means all points for kz = - pi / c
plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
# axes.quiver(kft0[: mesh_xy*1, 0], kft0[: mesh_xy*1, 1], vft0[: mesh_xy*1, 0], vft0[: mesh_xy*1, 1], color = 'k') # mesh_xy means all points for kz = - pi / c

# axes.set_xlim(-pi/a, pi/a)   # limit for xaxis
# axes.set_ylim(-pi/b, pi/b) # leave the ymax auto, but fix ymin
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

line = axes.contour(kxx, kyy, e_3D_func(kxx, kyy, - pi / c, band_parameters), 0, colors = '#FF0000', linewidths = 3)
line = axes.plot(kft0[0, 0], kft0[0, 1])
plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'b', ms = 8, mec = "#7E2320", mew= 0)  # set properties
line = axes.plot(kft[0,:, 0], kft[0,:, 1])
plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)  # set properties
line = axes.plot(kftp[0,:, 0], kftp[0,:, 1])
plt.setp(line, ls ="", c = 'b', lw = 3, marker = "s", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0)  # set properties

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
fig, axes = plt.subplots(1, 1, figsize = (10, 5.8)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.17, right = 0.78, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

#///// Allow to shift the label ticks up or down with set_pad /////#
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Labels //////#
fig.text(0.81,0.86, r"$B$ = " + str(B_amp))
fig.text(0.81,0.79, r"$\tau$ = " + str(tau))
fig.text(0.81,0.72, r"$\mu$ = " + str(mu))
fig.text(0.81,0.65, r"$t^\prime$ = " + str(tp))
fig.text(0.81,0.58, r"$t^{\prime\prime}$ = " + str(tpp))
fig.text(0.81,0.51, r"$t_{\rm z}$ = " + str(tz))

line = axes.plot(B_theta_a * 180 / pi, rho_zz_a / rho_zz_a[0])
plt.setp(line, ls ="-", c = '#FF0000', lw = 3, marker = "o", mfc = '#FF0000', ms = 7, mec = "#FF0000", mew= 0)  # set properties

axes.set_xlim(0, 180)   # limit for xaxis
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

folder = "../figures_sim/"
figure_name = "Rzz" + "_" + str(B_amp) + "_" + str(tau) + "_" + str(mu) + "_" + str(tp) + "_" + str(tpp) + "_" + str(tz) + ".pdf"
fig.savefig(folder + figure_name)
#//////////////////////////////////////////////////////////////////////////////#

