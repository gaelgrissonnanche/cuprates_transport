# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import sqrt, exp, log, pi, ones
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from numba import jit, prange
import time

from band_structure import *
from diff_equation import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

start_total_time = time.time()

## Constant //////
# hbar = 1.05e-34
# e = 1.6e-19
# m0 = 9.1e-31
# kB = 1.38e-23

e = 1
hbar = 1
m = 1

## Parameters //////
# c = 13.2
# a = 5.3 / sqrt(2)
# b = 5.3 / sqrt(2)

# mu = 805 # VHs = 600
# t = 525
# tp = -115
# tpp = 35
# tz = 11

# tau = 1e-3

a = 1
b = 1
c = 1


t   =  1.
tp  = -0.14 * t
tpp =  0.07 * t
tz  =  0.07 * t
mu  = 0.9 * t

tau =  25
B_amp = 0.02


symmetry_FS_xy = 1 # number of time the Fermi surface has been divided by along xy
symmetry_FS_z = 1 # number of time the Fermi surface has been divided by along z
mesh_xy = 56 # 28 must be a multiple of 4
mesh_z = 11 # 11 ideal to be fast and accurate
mesh_B_theta = 31
B_theta_max = 180


## Magnetic field tensor //////////////////////////////////////////////////////#
B_phi_a = np.array([0, 15, 30, 45]) * pi / 180
B_theta_a = np.linspace(0, B_theta_max * pi / 180, mesh_B_theta)

## Calculate a meshgrid of the angles for magnetic field (i,j) = (theta, phi)
B_theta_aa, B_phi_aa = np.meshgrid(B_theta_a, B_phi_a, indexing = 'ij')

## Calculate the magnetic field tensor (n, i, j) = (xyz, theta, phi)
B_tensor = B_func(B_amp, B_theta_aa, B_phi_aa)

# Bx = B_tensor[0,:,:].flatten() # put all rows one after the other in a one-dimension
# By = B_tensor[1,:,:].flatten() # array of size size_theta * size_phi, to go back to
# Bz = B_tensor[2,:,:].flatten() # the original, use B[n,:,:] = Bn.reshape(B_theta_aa.shape)

# kf = np.array([1,2,3])
# print(kf.shape)
# print(Bx.shape)
# print(np.ones(Bx.shape))
# kf_a = np.outer(kf, np.ones(Bx.shape))
# print(kf_a)

# print(Bx.reshape(B_theta_aa.shape))


## Fermi Surface t = 0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz])

## Make mesh_xy a multiple of 4 to respect the 4-order symmetry
mesh_xy = mesh_xy - (mesh_xy % 4)

start_time_FS = time.time()

## Discretize FS
kf, vf, dkf, number_contours = discretize_FS(band_parameters, mesh_xy, mesh_z, symmetry_FS_xy, symmetry_FS_z)

print("Discretize FS time : %.6s seconds" % (time.time() - start_time_FS))

## Solve differential equation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# @jit(nopython = True, cache = True)
def solve_movement_func(B_tensor, kf, band_parameters, tmax):

    dt = tmax / 300
    t = np.arange(0, tmax, dt)

    ## Run solver ///#
    kft = rgk4_algorithm(kf, t, B_tensor, band_parameters)
    vft = np.empty_like(kft)
    vft[0,:,:,:], vft[1,:,:,:], vft[2,:,:,:] = v_3D_func(kft[0,:,:,:], kft[1,:,:,:], kft[2,:,:,:], band_parameters)

    return kft, vft, t


# ## Conductivity sigma_zz >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

tmax = 10 * tau
kft, vft, t = solve_movement_func(B_tensor, kf, band_parameters, tmax)

print("Total time : %.6s seconds" % (time.time() - start_total_time))

# print("theta = " + str(B_theta * 180 / pi) + ", sigma_zz = " + r"{0:.5e}".format(s_zz))
# print("Calculation time : %.6s seconds" % (time.time() - start_time))

# @jit(nopython = True, cache = True)
# def sigma_zz(vf, vzft, kf, dkf, t, tau):

#     prefactor = e**2 / ( 4 * pi**3 )

#     # Time increment
#     dt = t[1] - t[0]
#     # Density of State
#     dos = hbar * sqrt( vf[0,:]**2 + vf[1,:]**2 + vf[2,:]**2 )

#     vzf = vf[2,:]

#     v_product = np.empty(vzf.shape[0])
#     for i0 in range(vzf.shape[0]):
#         vz_sum_over_t = np.sum( ( 1 / dos[i0] ) * vzft[i0,:] * exp(- t / tau) * dt ) # integral over t
#         v_product[i0] = vzf[i0] * vz_sum_over_t # integral over z

#     s_zz = prefactor * np.sum(dkf * v_product) # integral over k

#     return s_zz

# # Function of B_theta

# sigma_zz_a = np.empty((B_phi_a.shape[0], B_theta_a.shape[0]))

# for i, B_phi in enumerate(B_phi_a):
#     for j, B_theta in enumerate(B_theta_a):

#         start_time = time.time()

#         tmax = 10 * tau
#         kft, vft, t = solve_movement_func(B_amp, B_theta, B_phi, kf, band_parameters, tmax)
#         s_zz = sigma_zz(vf, vft[2,:,:], kf, dkf, t, tau)
#         sigma_zz_a[i, j] = s_zz

#         print("theta = " + str(B_theta * 180 / pi) + ", sigma_zz = " + r"{0:.5e}".format(s_zz))
#         print("Calculation time : %.6s seconds" % (time.time() - start_time))


# rho_zz_a = 1 / sigma_zz_a # dim (phi, theta)
# rho_zz_0 = rho_zz_a[:,0]

# print("Total time : %.6s seconds" % (time.time() - start_total_time))

# ## Save Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# len = rho_zz_a.shape[1]
# Data = np.vstack((B_theta_a, rho_zz_a[0,:] / rho_zz_0[0], B_amp*ones(len), tau*ones(len), mu*ones(len), 1*ones(len), tp*ones(len), tpp*ones(len), tz*ones(len), mesh_xy*ones(len), mesh_z*ones(len)))
# Data = Data.transpose()
# folder = "../data_sim/"
# file_name =  "Rzz" + "_mu_" + str(mu) + "_tp_" + str(tp) + "_tpp_" + str(tpp) + "_tz_" + str(tz) + "_B_" + str(B_amp) + "_tau_" + str(tau) + ".dat"
# np.savetxt(folder + file_name, Data, fmt='%.7e', header = "theta[deg]\trhozz(theta)/rhozz(0)\tB\ttau\tmu\tt\ttp\ttpp\ttz\tmesh_xy\tmesh_z", comments = "#")

# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
# ## Figures ////////////////////////////////////////////////////////////////////#
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# ## For figures, compute t-dependence
# kft, vft, t = solve_movement_func(B_amp, 0, 0, kf, band_parameters, tmax = 10 * tau)

# mesh_graph = 1001
# kx = np.linspace(-pi/a, pi/a, mesh_graph)
# ky = np.linspace(-pi/b, pi/b, mesh_graph)
# kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

# #///// RC Parameters //////#
# mpl.rcdefaults()
# mpl.rcParams['font.size'] = 24. # change the size of the font in every figure
# mpl.rcParams['font.family'] = 'Arial' # font Arial in every figure
# mpl.rcParams['axes.labelsize'] = 24.
# mpl.rcParams['xtick.labelsize'] = 24
# mpl.rcParams['ytick.labelsize'] = 24
# mpl.rcParams['xtick.direction'] = "in"
# mpl.rcParams['ytick.direction'] = "in"
# mpl.rcParams['xtick.top'] = True
# mpl.rcParams['ytick.right'] = True
# mpl.rcParams['xtick.major.width'] = 0.6
# mpl.rcParams['ytick.major.width'] = 0.6
# mpl.rcParams['axes.linewidth'] = 0.6 # thickness of the axes lines
# mpl.rcParams['pdf.fonttype'] = 3  # Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
#                                     # editing the text in illustrator


# ##>>>> 2D Fermi Surface >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# # axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# # fig.text(0.79,0.86, samplename, ha = "right")
# # fig.text(0.83,0.87, r"$T$ /  $H$  /  $\phi$ ", color = 'k', ha = 'left'))

# line = axes.contour(kxx, kyy, e_3D_func(kxx, kyy, - 2 * pi / c, band_parameters), 0, colors = '#FF0000', linewidths = 3)
# line = axes.plot(kf[0, : mesh_xy*1*number_contours], kf[1, : mesh_xy*1*number_contours]) # mesh_xy means all points for kz = - pi / c
# plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
# axes.quiver(kf[0, : mesh_xy*1*number_contours], kf[1, : mesh_xy*1*number_contours], vf[0, : mesh_xy*1*number_contours], vf[1, : mesh_xy*1*number_contours], color = 'k') # mesh_xy means all points for kz = - pi / c

# # axes.set_xlim(-pi/a, pi/a)   # limit for xaxis
# # axes.set_ylim(-pi/b, pi/b) # leave the ymax auto, but fix ymin
# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
# axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

# axes.locator_params(axis = 'y', nbins = 6)

# plt.show()
# #//////////////////////////////////////////////////////////////////////////////#


# #>>>> k vs t >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# # axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# # fig.text(0.79,0.86, samplename, ha = "right")
# # fig.text(0.83,0.87, r"$T$ /  $H$  /  $\phi$ ", color = 'k', ha = 'left'))

# line = axes.contour(kxx, kyy, e_3D_func(kxx, kyy, - 2 * pi / c, band_parameters), 0, colors = '#FF0000', linewidths = 3)
# line = axes.plot(kft[0, 0,:], kft[1, 0,:])
# plt.setp(line, ls ="-", c = 'b', lw = 1, marker = "", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0) # trajectory
# line = axes.plot(kf[0, 0], kf[1, 0])
# plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 4.5, mec = "b", mew= 1.5)  # starting point
# line = axes.plot(kft[0, 0, -1], kft[1, 0, -1])
# plt.setp(line, ls ="", c = 'b', lw = 1, marker = "o", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0)  # end point

# axes.set_xlim(-pi/a, pi/a)   # limit for xaxis
# axes.set_ylim(-pi/b, pi/b) # leave the ymax auto, but fix ymin
# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
# axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

# axes.locator_params(axis = 'y', nbins = 6)

# plt.show()

# #>>>> vf vs t >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# axes.axhline(y = 0, ls ="--", c ="k", linewidth = 0.6)

# #///// Allow to shift the label ticks up or down with set_pad /////#
# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# #///// Labels //////#
# # fig.text(0.79,0.86, samplename, ha = "right")

# line = axes.plot(t, vft[2, 0,:])
# plt.setp(line, ls ="-", c = 'g', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)

# # axes.set_xlim(0, 90)   # limit for xaxis
# # axes.set_ylim(ymin, ymax) # leave the ymax auto, but fix ymin
# axes.set_xlabel(r"$t$", labelpad = 8)
# axes.set_ylabel(r"$v_{\rm z}$", labelpad = 8)

# axes.locator_params(axis = 'y', nbins = 6)

# plt.show()
# #//////////////////////////////////////////////////////////////////////////////#

# #>>>> Cumulated velocity vs t >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# #///// Allow to shift the label ticks up or down with set_pad /////#
# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# #///// Labels //////#
# # fig.text(0.79,0.86, samplename, ha = "right")

# line = axes.plot(t, np.cumsum( vft[2, 0, :] * exp ( -t / tau ) ))
# plt.setp(line, ls ="-", c = 'k', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
# line = axes.plot(t, np.cumsum( vft[2, 1, :] * exp ( -t / tau ) ))
# plt.setp(line, ls ="-", c = 'r', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
# line = axes.plot(t, np.cumsum( vft[2, 2, :] * exp ( -t / tau ) ))
# plt.setp(line, ls ="-", c = 'b', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties


# # axes.set_xlim(0, 90)   # limit for xaxis
# # axes.set_ylim(ymin, ymax) # leave the ymax auto, but fix ymin
# axes.set_xlabel(r"$t$", labelpad = 8)
# axes.set_ylabel(r"$\sum_{\rm t}$ $v_{\rm z}(t)$$e^{\rm \dfrac{-t}{\tau}}$", labelpad = 8)

# axes.locator_params(axis = 'y', nbins = 6)

# plt.show()
# #//////////////////////////////////////////////////////////////////////////////#

# #>>>> Rzz vs theta >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (10, 5.8)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.17, right = 0.78, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

# #///// Allow to shift the label ticks up or down with set_pad /////#
# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# #///// Labels //////#
# fig.text(0.81,0.86, r"$B$ = " + str(B_amp))
# fig.text(0.81,0.79, r"$\tau$ = " + str(tau))
# fig.text(0.81,0.72, r"$\mu$ = " + str(mu))
# fig.text(0.81,0.65, r"$t^\prime$ = " + str(tp))
# fig.text(0.81,0.58, r"$t^{\prime\prime}$ = " + str(tpp))
# fig.text(0.81,0.51, r"$t_{\rm z}$ = " + str(tz))

# colors = ['k', '#3B528B', 'r', '#C7E500']

# for i, B_phi in enumerate(B_phi_a):
#     line = axes.plot(B_theta_a * 180 / pi, rho_zz_a[i,:] / rho_zz_0[i], label = r"$\phi$ = " + r"{0:.0f}".format(B_phi * 180 / pi))
#     plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "o", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)  # set properties

# axes.set_xlim(0, B_theta_max)   # limit for xaxis
# # axes.set_ylim(ymin, ymax) # leave the ymax auto, but fix ymin
# axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
# axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)

# ######################################################
# plt.legend(loc = 3, fontsize = 14, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
# ######################################################


# ##///Set ticks space and minor ticks space ///#
# xtics = B_theta_max / 6. # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks

# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
# axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# axes.locator_params(axis = 'y', nbins = 6)

# ## Inset
# axes_inset = plt.axes([0.79, 0.21, .2, .2])
# axes_inset.set_aspect(aspect=1)
# axes_inset.contour(kxx, kyy, e_3D_func(kxx, kyy, 0, band_parameters), 0, colors = '#FF0000', linewidths = 1)
# axes_inset.set_xlim(-pi/a,pi/a)
# axes_inset.set_ylim(-pi/b,pi/b)
# axes_inset.set_xticks([])
# axes_inset.set_yticks([])
# axes_inset.axis(**{'linewidth' : 0.2})

# plt.show()

# folder = "../figures_sim/"
# figure_name = "Rzz" + "_mu_" + str(mu) + "_tp_" + str(tp) + "_tpp_" + str(tpp) + "_tz_" + str(tz) + "_B_" + str(B_amp) + "_tau_" + str(tau) + ".pdf"
# fig.savefig(folder + figure_name)
# #//////////////////////////////////////////////////////////////////////////////#

