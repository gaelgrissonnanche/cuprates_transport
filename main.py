## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import exp, pi, ones
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from admr_routine import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Parameters //////
c = 13.3 # in Angstrom
a = 3.74 # in Angstrom
b = 3.74 # in Angstrom

t   =  190 # meV
tp  = -0.14 * t
tpp =  0.07 * t
tz  =  0.07 * t
tz2 = - 0 * t
mu  = 0.83 * t # van Hove 0.84

## Life time
gamma_0 = 110 # in THz
gamma_k = 0 # in THz
power   = 8

## Magnetic field
B_amp = 45 # in Tesla

## Discretization
half_FS_z = True # if False, put a minimum of 11 points
mesh_xy   = 40 # number of (kx,ky) points per contour per kz
mesh_z    = 7 # number of kz



## Magnetic field /////////////////////////////////////////////////////////////#
mesh_B_theta = 23 # number of theta angles for B
B_theta_max  = 110 # theta max for B, in degrees
B_phi_a = np.array([0, 15, 30, 45]) * pi / 180
B_theta_a = np.linspace(0, B_theta_max * pi / 180, mesh_B_theta)

## Array of parameters ////////////////////////////////////////////////////////#
mesh_parameters = np.array([mesh_xy, mesh_z, half_FS_z], dtype = np.float64)
band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz, tz2], dtype = np.float64)
tau_parameters = np.array([gamma_0, gamma_k, power], dtype = np.float64)

admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = True)


# mu_a = np.arange(0.81, 0.835, 0.001)
# for mu in mu_a:
#     mu = mu * t
#     print("mu = " + r"{0:.4f}".format(mu) + " * t" )
#     band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz, tz2], dtype = np.float64)
#     admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = False)

# gamma_k_a = np.arange(0, 120, 20)
# gamma_0_a = np.arange(50, 170, 20)
# for gamma_0 in gamma_0_a:
#     for gamma_k in gamma_k_a:
#         print("gamma_0 = " + r"{0:.1f}".format(gamma_0) + " THz")
#         print("gamma_k = " + r"{0:.1f}".format(gamma_k) + " THz")
#         tau_parameters = np.array([gamma_0, gamma_k, power], dtype = np.float64)
#         admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = False)






#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
## Figures ////////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

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

# ## Make mesh_xy a multiple of 4 to respect the 4-order symmetry
# mesh_xy = mesh_xy - (mesh_xy % 4)
# ## Discretize FS
# kf, vf, dkf, number_contours = discretize_FS(band_parameters, mesh_parameters)
# tau_0 = 1 / gamma_0
# ## For figures, compute t-dependence
# kft, vft, t = solveMovementFunc(B_amp, 0, 0, kf, band_parameters, tmax = 10 * tau_0)

# mesh_graph = 1001
# kx = np.linspace(-pi/a, pi/a, mesh_graph)
# ky = np.linspace(-pi/b, pi/b, mesh_graph)
# kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

# ##>>>> Discretize 2D Fermi Surface >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

# line = axes.contour(kxx*a, kyy*b, e_3D_func(kxx, kyy, 0, band_parameters), 0, colors = '#FF0000', linewidths = 3)
# # line = axes.contour(kxx*a+2*pi, kyy*b, e_3D_func(kxx, kyy, 2 * pi / c, band_parameters), 0, colors = '#8E62FF', linewidths = 3)
# line = axes.plot(kf[0, : mesh_xy*1*number_contours]*a, kf[1, : mesh_xy*1*number_contours]*b) # mesh_xy means all points for kz = - pi / c
# plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
# axes.quiver(kf[0, : mesh_xy*1*number_contours]*a, kf[1, : mesh_xy*1*number_contours]*b, vf[0, : mesh_xy*1*number_contours], vf[1, : mesh_xy*1*number_contours], color = 'k') # mesh_xy means all points for kz = - pi / c

# axes.set_xlim(-pi, pi)
# axes.set_ylim(-pi, pi)
# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
# axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

# axes.set_xticks([-pi, 0., pi])
# axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
# axes.set_yticks([-pi, 0., pi])
# axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

# plt.show()
# #//////////////////////////////////////////////////////////////////////////////#

# ##>>>> 2D Fermi Surface for different kz >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# fig.text(0.27,0.86, r"$k_{\rm z}$ =", fontsize = 14)
# fig.text(0.34,0.86, r"0", fontsize = 14, color = "#FF0000")
# fig.text(0.34,0.83, r"$\pi/c$", fontsize = 14, color = "#00DC39")
# fig.text(0.34,0.80, r"2$\pi/c$", fontsize = 14, color = "#6577FF")

# line = axes.contour(kxx*a, kyy*b, e_3D_func(kxx, kyy, 0, band_parameters), 0, colors = '#FF0000', linewidths = 3)
# line = axes.contour(kxx*a, kyy*b, e_3D_func(kxx, kyy, pi / c, band_parameters), 0, colors = '#00DC39', linewidths = 3)
# line = axes.contour(kxx*a, kyy*b, e_3D_func(kxx, kyy, 2 * pi / c, band_parameters), 0, colors = '#6577FF', linewidths = 3)

# axes.set_xlim(-pi, pi)
# axes.set_ylim(-pi, pi)
# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
# axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

# axes.set_xticks([-pi, 0., pi])
# axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
# axes.set_yticks([-pi, 0., pi])
# axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

# plt.show()
# #//////////////////////////////////////////////////////////////////////////////#

# ##>>>> Life Time >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.15, right = 0.85, bottom = 0.15, top = 0.85) # adjust the box of axes regarding the figure size

# phi = np.linspace(0, 2*pi, 1000)

# ## tau_0
# tau_0_x = tau_0 * cos(phi)
# tau_0_y = tau_0 * sin(phi)
# line = axes.plot(tau_0_x / tau_0, tau_0_y / tau_0, clip_on = False)
# plt.setp(line, ls ="--", c = 'k', lw = 2, marker = "", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
# axes.annotate(r"$\tau_{\rm 0}$", xy = (0.75, 0.75), color = 'k')

# ## tau_k
# tau_k_x = 1 / (gamma_0 + gamma_k * (sin(phi)**2 - cos(phi)**2)**power) * cos(phi)
# tau_k_y = 1 / (gamma_0 + gamma_k * (sin(phi)**2 - cos(phi)**2)**power) * sin(phi)
# line = axes.plot(tau_k_x / tau_0, tau_k_y / tau_0, clip_on = False)
# plt.setp(line, ls ="-", c = '#FF9C54', lw = 3, marker = "", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
# axes.annotate(r"$\tau_{\rm k}$", xy = (0.5, 0.5), color = '#FF9C54')

# ## tau_k_min
# phi_min = 3 * pi / 2
# tau_k_x_min = 1 / (gamma_0 + gamma_k * (sin(phi_min)**2 - cos(phi_min)**2)**power) * cos(phi_min)
# tau_k_y_min = 1 / (gamma_0 + gamma_k * (sin(phi_min)**2 - cos(phi_min)**2)**power) * sin(phi_min)
# line = axes.plot(tau_k_x_min / tau_0, tau_k_y_min / tau_0, clip_on = False)
# plt.setp(line, ls ="", c = '#FF9C54', lw = 3, marker = "o", mfc = '#FF9C54', ms = 9, mec = "#7E2320", mew= 0)
# fraction = np.abs(np.round(tau_k_y_min / tau_0, 2))
# axes.annotate(r"{0:.2f}".format(fraction) + r"$\tau_{\rm 0}$", xy = (-0.3, tau_k_y_min / tau_0 * 0.85), color = '#FF9C54')

# axes.set_xlim(-1, 1)
# axes.set_ylim(-1, 1)
# axes.set_xticklabels([])
# axes.set_yticklabels([])

# plt.show()
# #//////////////////////////////////////////////////////////////////////////////#

# #>>>> k vs t >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)

# fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

# line = axes.contour(kxx*a, kyy*b, e_3D_func(kxx, kyy, 0, band_parameters), 0, colors = '#FF0000', linewidths = 3)
# line = axes.plot(kft[0, 0,:]*a, kft[1, 0,:]*b)
# plt.setp(line, ls ="-", c = 'b', lw = 1, marker = "", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0) # trajectory
# line = axes.plot(kf[0, 0]*a, kf[1, 0]*b)
# plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 4.5, mec = "b", mew= 1.5)  # starting point
# line = axes.plot(kft[0, 0, -1]*a, kft[1, 0, -1]*b)
# plt.setp(line, ls ="", c = 'b', lw = 1, marker = "o", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0)  # end point

# axes.set_xlim(-pi, pi)
# axes.set_ylim(-pi, pi)
# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
# axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

# axes.set_xticks([-pi, 0., pi])
# axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
# axes.set_yticks([-pi, 0., pi])
# axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

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


# line = axes.plot(t, vft[2, -200,:])
# plt.setp(line, ls ="-", c = '#6AFF98', lw = 3, marker = "", mfc = '#6AFF98', ms = 5, mec = "#7E2320", mew= 0)

# axes.set_xlabel(r"$t$", labelpad = 8)
# axes.set_ylabel(r"$v_{\rm z}$ ( $k_{\rm z}$ = $\pi$/$c$ )", labelpad = 8)
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


# line = axes.plot(t, np.cumsum( vft[2, -1, :] * exp ( -t / tau_0 ) ))
# plt.setp(line, ls ="-", c = 'k', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
# line = axes.plot(t, np.cumsum( vft[2, -2, :] * exp ( -t / tau_0 ) ))
# plt.setp(line, ls ="-", c = 'r', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
# line = axes.plot(t, np.cumsum( vft[2, -3, :] * exp ( -t / tau_0 ) ))
# plt.setp(line, ls ="-", c = 'b', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties


# axes.set_xlabel(r"$t$", labelpad = 8)
# axes.set_ylabel(r"$\sum_{\rm t}$ $v_{\rm z}(t)$$e^{\rm \dfrac{-t}{\tau}}$", labelpad = 8)

# axes.locator_params(axis = 'y', nbins = 6)

# plt.show()
# #//////////////////////////////////////////////////////////////////////////////#

