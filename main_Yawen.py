import time

from band import BandStructure
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

t = 533.616
bandObject = BandStructure(t=t, mu=-569.868/t, tp=-113.561/t,
                           tpp=23.2192/t, tz=8.7296719/t, tz2=-0.89335299/t)

# condObject = Conductivity(bandObject, Bamp=45, Bphi=0, Btheta=0, gamma_0 = 25*2*3.14, gamma_k = 0, power = 0)
# condObject.solveMovementFunc()
# condObject.figOnekft()

start_total_time = time.time()
ADMRObject = ADMR(bandObject, Bamp=45, gamma_0=25, gamma_k=0, power=0)
ADMRObject.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

ADMRObject.fileADMR()
ADMRObject.figADMR()



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








# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
# ## Figures ////////////////////////////////////////////////////////////////#
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

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


# ## Discretize FS
# kf, vf, dkf, numberPointsPerKz_list = discretize_FS(band_parameters, mesh_parameters)

# tau_0 = 1 / gamma_0
# ## For figures, compute t-dependence
# kft, vft, t = solveMovementFunc(B_amp, 0, 0, kf, band_parameters, tmax = 10 * tau_0)

# mesh_graph = 1001
# kx = np.linspace(-pi/a, pi/a, mesh_graph)
# ky = np.linspace(-pi/b, pi/b, mesh_graph)
# kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

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







