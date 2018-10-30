## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import sqrt, exp, pi, ones
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import time

from band_structure import *
from movement_equation import *
from chambers_formula import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

start_total_time = time.time()

## Constant //////
hbar = 1.05e-34 # m2 kg / s
e = 1.6e-19 # C
# m0 = 9.1e-31 # kg


## Parameters //////
c = 13.3e-10 # in meter
a = 3.74e-10 # in meter
b = 3.74e-10 # in meter



t   =  190e-3 # eV
t = t * 1.602e-19 # in J
tp  = -0.14 * t
tpp =  0.07 * t
tz  =  0.07 * t
tz2 = - 0 * t
mu  = 0.9 * t # van Hove 0.84

## Life time
tau_0 =  1e-14# in seconds
gamma = 0
power = 8

## Magnetic field
B_amp = 45 # in Tesla
mesh_B_theta = 23
B_theta_max = 110

## Discretization
half_FS_z = True # if False, put a minimum of 11 points
mesh_xy = 56
mesh_z = 7


## Magnetic field /////////////////////////////////////////////////////////////#
B_phi_a = np.array([0, 15, 30, 45]) * pi / 180
B_theta_a = np.linspace(0, B_theta_max * pi / 180, mesh_B_theta)


band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz, tz2], dtype = np.float64)
tau_parameters = np.array([tau_0, gamma, power], dtype = np.float64)

## Discretize Fermi Surface >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## Make mesh_xy a multiple of 4 to respect the 4-order symmetry
mesh_xy = mesh_xy - (mesh_xy % 4)
## Discretize FS
kf, vf, dkf, number_contours = discretize_FS(band_parameters, mesh_xy, mesh_z, half_FS_z)


## ADMR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
rho_zz_a = admrFunc(B_amp, B_theta_a, B_phi_a, kf, vf, dkf, band_parameters, tau_parameters)
print(rho_zz_a[:,0])
rho_zz_0 = rho_zz_a[:,0]

print("Total time : %.6s seconds" % (time.time() - start_total_time))


## Save Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
len = rho_zz_a.shape[1]
t = band_parameters[4]
Data = np.vstack((B_theta_a, rho_zz_a[0,:] / rho_zz_0[0], rho_zz_a[1,:] / rho_zz_0[1], rho_zz_a[2,:] / rho_zz_0[2], rho_zz_a[3,:] / rho_zz_0[3],
                  B_amp*ones(len), tau_0*ones(len), gamma*ones(len), power*ones(len), mu*ones(len), t*ones(len), tp*ones(len), tpp*ones(len), tz*ones(len), tz2*ones(len), mesh_xy*ones(len), mesh_z*ones(len)))
Data = Data.transpose()
folder = "results_sim/"
file_name =  "Rzz" + "_mu_" + str(mu) + "_t_" + str(t) + "_tp_" + str(tp) + "_tpp_" + str(tpp) + "_tz_" + str(tz) + "_tz2_" + str(tz2) + \
             "_B_" + str(B_amp) + "_tau0_" + str(tau_0) + "_g_" + str(gamma) + "_p_" + str(power) + ".dat"
np.savetxt(folder + file_name, Data, fmt='%.7e',
header = "theta[deg]\trzz(phi=0)/rzz_0\trzz(phi=15)/rzz_0\trzz(phi=30)/rzz_0\trzz(phi=45)/rzz_0\tB\ttau0\tgamma\tpower\tmu\tt\ttp\ttpp\ttz\ttz2\tmesh_xy\tmesh_z", comments = "#")




#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
## Figures ////////////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

## For figures, compute t-dependence
kft, vft, t = solveMovementFunc(B_amp, 0, 0, kf, band_parameters, tmax = 10 * tau_0)
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

fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

line = axes.contour(kxx*a, kyy*b, e_3D_func(kxx, kyy, 0, band_parameters), 0, colors = '#FF0000', linewidths = 3)
line = axes.plot(kf[0, : mesh_xy*1*number_contours]*a, kf[1, : mesh_xy*1*number_contours]*b) # mesh_xy means all points for kz = - pi / c
plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
axes.quiver(kf[0, : mesh_xy*1*number_contours]*a, kf[1, : mesh_xy*1*number_contours]*b, vf[0, : mesh_xy*1*number_contours], vf[1, : mesh_xy*1*number_contours], color = 'k') # mesh_xy means all points for kz = - pi / c

axes.set_xlim(-pi, pi)
axes.set_ylim(-pi, pi)
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

axes.set_xticks([-pi, 0., pi])
axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
axes.set_yticks([-pi, 0., pi])
axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

plt.show()
#//////////////////////////////////////////////////////////////////////////////#

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
# tau_k_x = 1 / (1 / tau_0 + gamma * (sin(phi)**2 - cos(phi)**2)**power) * cos(phi)
# tau_k_y = 1 / (1 / tau_0 + gamma * (sin(phi)**2 - cos(phi)**2)**power) * sin(phi)
# line = axes.plot(tau_k_x / tau_0, tau_k_y / tau_0, clip_on = False)
# plt.setp(line, ls ="-", c = '#FF9C54', lw = 3, marker = "", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
# axes.annotate(r"$\tau_{\rm k}$", xy = (0.5, 0.5), color = '#FF9C54')

# ## tau_k_min
# phi_min = 3 * pi / 2
# tau_k_x_min = 1 / (1 / tau_0 + gamma * (sin(phi_min)**2 - cos(phi_min)**2)**power) * cos(phi_min)
# tau_k_y_min = 1 / (1 / tau_0 + gamma * (sin(phi_min)**2 - cos(phi_min)**2)**power) * sin(phi_min)
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

#>>>> k vs t >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

# axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

line = axes.contour(kxx*a, kyy*b, e_3D_func(kxx, kyy, 0, band_parameters), 0, colors = '#FF0000', linewidths = 3)
line = axes.plot(kft[0, 5,:]*a, kft[1, 5,:]*b)
plt.setp(line, ls ="-", c = 'b', lw = 1, marker = "", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0) # trajectory
line = axes.plot(kf[0, 5]*a, kf[1, 5]*b)
plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 4.5, mec = "b", mew= 1.5)  # starting point
line = axes.plot(kft[0, 5, -1]*a, kft[1, 5, -1]*b)
plt.setp(line, ls ="", c = 'b', lw = 1, marker = "o", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0)  # end point

axes.set_xlim(-pi, pi)
axes.set_ylim(-pi, pi)
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

axes.set_xticks([-pi, 0., pi])
axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
axes.set_yticks([-pi, 0., pi])
axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

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


line = axes.plot(t, vft[2, -200,:])
plt.setp(line, ls ="-", c = '#6AFF98', lw = 3, marker = "", mfc = '#6AFF98', ms = 5, mec = "#7E2320", mew= 0)

# axes.set_xlim(0, 90)
# axes.set_ylim(ymin, ymax)
axes.set_xlabel(r"$t$", labelpad = 8)
axes.set_ylabel(r"$v_{\rm z}$ ( $k_{\rm z}$ = $\pi$/$c$ )", labelpad = 8)
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

line = axes.plot(t, np.cumsum( vft[2, -1, :] * exp ( -t / tau_0 ) ))
plt.setp(line, ls ="-", c = 'k', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
line = axes.plot(t, np.cumsum( vft[2, -2, :] * exp ( -t / tau_0 ) ))
plt.setp(line, ls ="-", c = 'r', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
line = axes.plot(t, np.cumsum( vft[2, -3, :] * exp ( -t / tau_0 ) ))
plt.setp(line, ls ="-", c = 'b', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties


# axes.set_xlim(0, 90)
# axes.set_ylim(ymin, ymax)
axes.set_xlabel(r"$t$", labelpad = 8)
axes.set_ylabel(r"$\sum_{\rm t}$ $v_{\rm z}(t)$$e^{\rm \dfrac{-t}{\tau}}$", labelpad = 8)

axes.locator_params(axis = 'y', nbins = 6)

plt.show()
#//////////////////////////////////////////////////////////////////////////////#

#>>>> Rzz vs theta >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
fig, axes = plt.subplots(1, 1, figsize = (10.5, 5.8)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.15, right = 0.75, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

#///// Allow to shift the label ticks up or down with set_pad /////#
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

#///// Labels //////#
t = band_parameters[4]
label_parameters = [r"$B$ = " + str(B_amp),
                    "",
                    r"$\tau_{\rm 0}$ = " + str(tau_0),
                    r"$\gamma$ = " + str(gamma),
                    r"power = " + str(power),
                    "",
                    r"$t$ = " + "{0:.3e}".format(t),
                    r"$\mu$ = " + "{0:.3f}".format(mu/t) + r" $t$",
                    r"$t^\prime$ = " + "{0:.3f}".format(tp/t) + r" $t$",
                    r"$t^{\prime\prime}$ = " + "{0:.3f}".format(tpp/t) + r" $t$",
                    r"$t_{\rm z}$ = " + "{0:.3f}".format(tz/t) + r" $t$",
                    r"$t_{\rm z}^{\prime}$ = " + "{0:.3f}".format(tz2/t) + r" $t$"]

h_label = 0.92
for label in label_parameters:
    fig.text(0.78, h_label, label, fontsize = 16)
    h_label -= 0.045

colors = ['k', '#3B528B', 'r', '#C7E500']

for i, B_phi in enumerate(B_phi_a):
    line = axes.plot(B_theta_a * 180 / pi, rho_zz_a[i,:] / rho_zz_0[i], label = r"$\phi$ = " + r"{0:.0f}".format(B_phi * 180 / pi))
    plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "o", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)  # set properties

axes.set_xlim(0, B_theta_max)
# axes.set_ylim(ymin, ymax)
axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)

######################################################
plt.legend(loc = 0, fontsize = 14, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################


##///Set ticks space and minor ticks space ///#
xtics = 30. # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks

majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

axes.locator_params(axis = 'y', nbins = 6)

## Inset FS //////////////////////////////////#
mesh_graph = 201
kx = np.linspace(-pi/a, pi/a, mesh_graph)
ky = np.linspace(-pi/b, pi/b, mesh_graph)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

axes_inset_FS = plt.axes([0.74, 0.18, .18, .18])
axes_inset_FS.set_aspect(aspect=1)
CS_0 = axes_inset_FS.contour(kxx, kyy, e_3D_func(kxx, kyy, 0, band_parameters), 0, colors = '#FF0000', linewidths = 1)
axes_inset_FS.annotate(r"0", xy = (- pi/a * 0.9, - pi/b * 0.7), color = 'r', fontsize = 8)
CS_pi_c =axes_inset_FS.contour(kxx, kyy, e_3D_func(kxx, kyy, pi / c, band_parameters), 0, colors = '#6577FF', linewidths = 1)
axes_inset_FS.annotate(r"$\pi$/c", xy = (- pi/a * 0.9, - pi/b * 0.9), color = '#6577FF', fontsize = 8)
axes_inset_FS.set_xlim(-pi/a,pi/a)
axes_inset_FS.set_ylim(-pi/b,pi/b)
axes_inset_FS.set_xticks([])
axes_inset_FS.set_yticks([])
axes_inset_FS.axis(**{'linewidth' : 0.2})


## Inset tau /////////////////////////////////#
axes_inset_tau = plt.axes([0.85, 0.18, .18, .18])
axes_inset_tau.set_aspect(aspect=1)

phi = np.linspace(0, 2*pi, 1000)
## tau_0
tau_0_x = tau_0 * cos(phi)
tau_0_y = tau_0 * sin(phi)
line = axes_inset_tau.plot(tau_0_x / tau_0, tau_0_y / tau_0, clip_on = False)
plt.setp(line, ls ="-", c = 'k', lw = 1, marker = "", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
axes_inset_tau.annotate(r"$\tau_{\rm 0}$", xy = (0.65, 0.75), color = 'k', fontsize = 10)
## tau_k
tau_k_x = 1 / (1 / tau_0 + gamma * (sin(phi)**2 - cos(phi)**2)**power) * cos(phi)
tau_k_y = 1 / (1 / tau_0 + gamma * (sin(phi)**2 - cos(phi)**2)**power) * sin(phi)
line = axes_inset_tau.plot(tau_k_x / tau_0, tau_k_y / tau_0, clip_on = False)
plt.setp(line, ls ="-", c = '#FF9C54', lw = 1, marker = "", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
axes_inset_tau.annotate(r"$\tau_{\rm k}$", xy = (0.4, 0.45), color = '#FF9C54', fontsize = 10)
## tau_k_min
phi_min = 3 * pi / 2
tau_k_x_min = 1 / (1 / tau_0 + gamma * (sin(phi_min)**2 - cos(phi_min)**2)**power) * cos(phi_min)
tau_k_y_min = 1 / (1 / tau_0 + gamma * (sin(phi_min)**2 - cos(phi_min)**2)**power) * sin(phi_min)
line = axes_inset_tau.plot(tau_k_x_min / tau_0, tau_k_y_min / tau_0, clip_on = False)
plt.setp(line, ls ="", c = '#FF9C54', lw = 3, marker = "o", mfc = '#FF9C54', ms = 4, mec = "#7E2320", mew= 0)
fraction = np.abs(np.round(tau_k_y_min / tau_0, 2))
axes_inset_tau.annotate(r"{0:.2f}".format(fraction) + r"$\tau_{\rm 0}$", xy = (-0.35, tau_k_y_min / tau_0 * 0.8), color = '#FF9C54', fontsize = 8)

axes_inset_tau.set_xlim(-1,1)
axes_inset_tau.set_ylim(-1,1)
axes_inset_tau.set_xticks([])
axes_inset_tau.set_yticks([])
axes_inset_tau.axis(**{'linewidth' : 0.2})

plt.show()

t = band_parameters[4]
folder = "results_sim/"
figure_name =  "Rzz" + "_mu_" + str(mu) + "_t_" + str(t) + "_tp_" + str(tp) + "_tpp_" + str(tpp) + "_tz_" + str(tz) + "_tz2_" + str(tz2) + \
               "_B_" + str(B_amp) + "_tau0_" + str(tau_0) + "_g_" + str(gamma) + "_p_" + str(power) + ".pdf"
fig.savefig(folder + figure_name)
#//////////////////////////////////////////////////////////////////////////////#

