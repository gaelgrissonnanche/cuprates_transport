import numpy as np
from numpy import pi, cos, sin
from skimage import measure
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
from lmfit import minimize, Parameters, fit_report
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

a = 3.87 # angstrom
c = 23.20 # angstrom
d = c / 2


## Parameters
mesh_phi = 100
exagerate = 1

k00 = 7.45 / 10 # in angstrom^-1
k40 = -0.19 / 10 # in angstrom^-1
k21 = 0.031 / 10 # in angstrom^-1
k61 = 0.021 / 10 # in angstrom^-1
k101 = -0.0085 / 10 # in angstrom^-1

k21 = 0
k61 = 0
k101 = 0

phi_array = np.linspace(0, 2*pi, mesh_phi)
kz_array = np.array([0, pi/c, 2*pi/c])


## Compute Fermi surface from Harmonic expansion ///////////////////////////////
kF_x_Harm = np.empty((3, len(phi_array)))
kF_y_Harm = np.empty((3, len(phi_array)))

for i, kz in enumerate(kz_array):
    kF = k00 + k40*cos(4*phi_array) + exagerate * cos(kz*d) * (k21*sin(2*phi_array) + k61*sin(6*phi_array) + k101*sin(10*phi_array))
    kF_x_Harm[i, :] = kF * cos(phi_array) * a + pi
    kF_y_Harm[i, :] = kF * sin(phi_array) * a + pi


## Plot \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
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


# ## Figure PZB //////////////////////////////////////////////////////////////////
# fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
# fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

# colors = ["#FF0000", "#00DC39", "#6577FF"]

# axes.axhline(y = pi, ls ="--", c ="k", linewidth = 0.6)
# axes.axvline(x = pi, ls ="--", c ="k", linewidth = 0.6)

# legend_list = [r"$k_{\rm z}$ = 0", r"$k_{\rm z}$ = $\pi$/c", r"$k_{\rm z}$ = 2$\pi$/c"]

# for i, kz in enumerate(kz_array):
#     line = axes.plot(kF_x_Harm[i,:], kF_y_Harm[i,:], label = legend_list[i])
#     plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)


# plt.legend(bbox_to_anchor = (0.15,0.78), loc = 2, fontsize = 12, frameon = False, numpoints = 1, markerscale = 1, handletextpad = 0.5)

# axes.set_aspect(aspect=1)
# axes.set_xlim(0, 2 * pi)
# axes.set_ylim(0, 2 * pi)
# axes.set_xticks([0, pi, 2*pi])
# axes.set_xticklabels([r"0", r"$\pi$", r"$2\pi$"])
# axes.set_yticks([0, pi, 2*pi])
# axes.set_yticklabels([r"0", r"$\pi$", r"$2\pi$"])
# axes.tick_params(axis='x', which='major', pad=7)
# axes.tick_params(axis='y', which='major', pad=8)
# axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
# axes.set_ylabel(r"$k_{\rm y}$")
# ################################################################################



# ## Figure vs phi ///////////////////////////////////////////////////////////////
# fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
# fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95)

# colors = ["#FF0000", "#00DC39", "#6577FF"]

# # axes.axhline(y = pi, ls ="--", c ="k", linewidth = 0.6)
# # axes.axvline(x = pi, ls ="--", c ="k", linewidth = 0.6)

# legend_list = [r"$k_{\rm z}$ = 0", r"$k_{\rm z}$ = $\pi$/c", r"$k_{\rm z}$ = 2$\pi$/c"]

# for i, kz in enumerate(kz_array):
#     line = axes.plot(phi_array, pi + kF_x_Harm[i,:]*a, label = r"$k_{\rm x}$, " + legend_list[i])
#     plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)
#     line = axes.plot(phi_array, pi + kF_y_Harm[i,:]*a, label = r"$k_{\rm y}$, " + legend_list[i])
#     plt.setp(line, ls ="--", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)


# plt.legend(bbox_to_anchor = (0,0), loc = 3, fontsize = 12, frameon = False, numpoints = 1, markerscale = 1, handletextpad = 0.5)

# axes.set_xlim(0, 2 * pi)
# # axes.set_ylim(0, 2 * pi)
# axes.set_xticks([0, pi, 2*pi])
# axes.set_xticklabels([r"0", r"$\pi$", r"$2\pi$"])
# # axes.set_yticks([0, pi, 2*pi])
# # axes.set_yticklabels([r"0", r"$\pi$", r"$2\pi$"])
# axes.tick_params(axis='x', which='major', pad=7)
# axes.tick_params(axis='y', which='major', pad=8)
# axes.set_xlabel(r"$\phi$", labelpad = 8)
# axes.set_ylabel(r"$k_{\rm n}$")
# ################################################################################


##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## Tight Binding (TB) ////////////////////////////////////////////////

params = {
    "bandname": "HolePocket",
    "a": 3.87,
    "b": 3.87,
    "c": 23.20,
    "t": 181,
    "tp": -0.40,
    # "tpp": 0.001,
    "tppp" : 0,
    "tpppp" : 0,
    "tz": -0.015,
    # "tz": 0.0285*8,
    "tz2": 0.01,
    "tz3": 0,
    # "tz4": 0.005,
    "mu": -1.55,
    "fixdoping": 0.25,
    "numberOfKz": 7,
    "mesh_ds": 1 / 20,
    "Ntime": 500,
    "T": 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 20, 28, 36, 44],
    "gamma_0": 4.2,
    "gamma_k": 0,
    "power": 2,
    "gamma_dos_max": 0,
    "factor_arcs": 1,
}

## ONE BAND Horio et al. /////////////////////////////////////////////////////////
bandObject = BandStructure(**params)
bandObject.doping(printDoping=True)

def interp_FS_TB(bandObject, phi_array, mesh_xy_rough=501):

    kx_a = np.linspace(0, 2*pi / bandObject.a, mesh_xy_rough)
    ky_a = np.linspace(0, 2*pi / bandObject.b, mesh_xy_rough)
    kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

    kF_x_TB = np.empty((3, len(phi_array)))
    kF_y_TB = np.empty((3, len(phi_array)))

    for i, kz in enumerate(kz_array):
        bands = bandObject.e_3D_func(kxx, kyy, kz)
        contour = measure.find_contours(bands, 0)
        kx = contour[0][:, 0] / (mesh_xy_rough - 1) * 2*pi
        ky = contour[0][:, 1] / (mesh_xy_rough - 1) * 2*pi / (bandObject.b / bandObject.a)  # anisotropy
        phi = np.arctan2(ky-pi, kx-pi) + pi
        index_order = np.argsort(phi)
        kF_x_TB[i,:] = np.interp(phi_array, phi[index_order], kx[index_order] - pi) + pi
        kF_y_TB[i,:] = np.interp(phi_array, phi[index_order], ky[index_order] - pi) + pi

    return kF_x_TB, kF_y_TB




# ## Function residual ########
# def residualFunc(pars, bandObject, phi_array, kF_x_Harm, kF_y_Harm):
#     bandObject.mu  = pars["mu"].value
#     bandObject.tp  = pars["tp"].value
#     bandObject.tpp = pars["tpp"].value
#     bandObject.tz  = pars["tz"].value
#     bandObject.tz2 = pars["tz2"].value
#     bandObject.tz3 = pars["tz3"].value

#     print("mu = " + str(bandObject.mu))
#     print("tp = " + str(bandObject.tp))
#     print("tpp = " + str(bandObject.tpp))
#     print("tz = " + str(bandObject.tz))
#     print("tz2 = " + str(bandObject.tz2))
#     print("tz3 = " + str(bandObject.tz3)+"\n")


#     kF_x_TB, kF_y_TB = interp_FS_TB(bandObject, phi_array)

#     diff_x = kF_x_Harm - kF_x_TB
#     diff_y = kF_y_Harm - kF_y_TB

#     return np.concatenate((diff_x, diff_y))


# ## Initialize
# pars = Parameters()
# pars.add("mu", value=params["mu"], vary=True, min=-1.3, max=-1.6)
# pars.add("tp", value=params["tp"], vary=True, min=-0.7, max=-0.2)
# pars.add("tpp", value=params["tpp"], vary=False) #, min=0.06, max=0.22)
# pars.add("tz", value=params["tz"], vary=True, min=-0.05, max=0.05)
# pars.add("tz2", value=params["tz2"], vary=True, min=-0.05, max=0.05)
# pars.add("tz3", value=params["tz3"], vary=True, min=-0.05, max=0.05)

# ## Run fit algorithm
# out = minimize(residualFunc, pars, args=(bandObject, phi_array, kF_x_Harm, kF_y_Harm))

# ## Display fit report
# print(fit_report(out.params))

# ## Export final parameters from the fit
# bandObject.mu = out.params["mu"].value
# bandObject.tp = out.params["tp"].value
# bandObject.tpp = out.params["tpp"].value
# bandObject.tz = out.params["tz"].value
# bandObject.tz2 = out.params["tz2"].value
# bandObject.tz3 = out.params["tz3"].value



kF_x_TB, kF_y_TB = interp_FS_TB(bandObject, phi_array)


## Figure PZB //////////////////////////////////////////////////////////////////
fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

colors = ["#FF0000", "#00DC39", "#6577FF"]

axes.axhline(y = pi, ls ="--", c ="k", linewidth = 0.6)
axes.axvline(x = pi, ls ="--", c ="k", linewidth = 0.6)

legend_list = [r"$k_{\rm z}$ = 0", r"$k_{\rm z}$ = $\pi$/c", r"$k_{\rm z}$ = 2$\pi$/c"]


for i, kz in enumerate(kz_array):
    line = axes.plot(kF_x_Harm[i,:], kF_y_Harm[i,:], label = legend_list[i])
    plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)
for i, kz in enumerate(kz_array):
    line = axes.plot(kF_x_TB[i,:], kF_y_TB[i,:], label = legend_list[i])
    plt.setp(line, ls ="--", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 2, mec = colors[i], mew= 0)


plt.legend(bbox_to_anchor = (0.15,0.78), loc = 2, fontsize = 12, frameon = False, numpoints = 1, markerscale = 1, handletextpad = 0.5)

axes.set_aspect(aspect=1)
axes.set_xlim(0, 2 * pi/a)
axes.set_ylim(0, 2 * pi/a)
axes.set_xticks([0, pi, 2*pi])
axes.set_xticklabels([r"0", r"$\pi$", r"$2\pi$"])
axes.set_yticks([0, pi, 2*pi])
axes.set_yticklabels([r"0", r"$\pi$", r"$2\pi$"])
axes.tick_params(axis='x', which='major', pad=7)
axes.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
axes.set_ylabel(r"$k_{\rm y}$")
################################################################################



plt.show()