import numpy as np
from numpy import pi, exp
from scipy.constants import Boltzmann, hbar, elementary_charge, physical_constants
from skimage import measure
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

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

def figScatteringColor(self, kz=0, gamma_min=None, gamma_max=None, mesh_xy=501):
    bObj = self.bandObject
    fig, axes = plt.subplots(1, 1, figsize=(6.5, 5.6))
    fig.subplots_adjust(left=0.10, right=0.85, bottom=0.20, top=0.9)

    kx_a = np.linspace(-pi / bObj.a, pi / bObj.a, mesh_xy)
    ky_a = np.linspace(-pi / bObj.b, pi / bObj.b, mesh_xy)

    kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

    bands = bObj.e_3D_func(kxx, kyy, kz)
    contours = measure.find_contours(bands, 0)

    gamma_max_list = []
    gamma_min_list = []
    for contour in contours:

        # Contour come in units proportionnal to size of meshgrid
        # one want to scale to units of kx and ky
        kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / bObj.a
        ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / bObj.b
        vx, vy, vz = bObj.v_3D_func(kx, ky, kz)

        gamma_kz = 1 / self.tau_total_func(kx, ky, kz, vx, vy, vz)
        gamma_max_list.append(np.max(gamma_kz))
        gamma_min_list.append(np.min(gamma_kz))

        points = np.array([kx*bObj.a, ky*bObj.b]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('gnuplot'))
        lc.set_array(gamma_kz)
        lc.set_linewidth(4)

        line = axes.add_collection(lc)


    if gamma_min == None:
        gamma_min = min(gamma_min_list)
    if gamma_max == None:
        gamma_min = min(gamma_min_list)
    line.set_clim(gamma_min, gamma_max)
    cbar = fig.colorbar(line, ax=axes)
    cbar.minorticks_on()
    cbar.ax.set_ylabel(r'$\Gamma_{\rm tot}$ ( THz )', rotation=270, labelpad=40)

    kz = np.round(kz/(np.pi/bObj.c), 1)
    fig.text(0.97, 0.05, r"$k_{\rm z}$ = " + "{0:g}".format(kz) + r"$\pi$/c",
             ha="right", color="r")
    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$k_{\rm x}$", labelpad=8)
    axes.set_ylabel(r"$k_{\rm y}$", labelpad=8)

    axes.set_xticks([-pi, 0., pi])
    axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    axes.set_yticks([-pi, 0., pi])
    axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

    plt.show()

def figScatteringPhi(self, kz=0, mesh_xy=501):
    bObj = self.bandObject
    fig, axes = plt.subplots(1, 1, figsize=(6.5, 4.6))
    fig.subplots_adjust(left=0.20, right=0.8, bottom=0.20, top=0.9)
    axes2 = axes.twinx()
    axes2.set_axisbelow(True)

    ###
    kx_a = np.linspace(-pi / bObj.a, pi / bObj.a, mesh_xy)
    ky_a = np.linspace(-pi / bObj.b, pi / bObj.b, mesh_xy)

    kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

    bands = bObj.e_3D_func(kxx, kyy, kz)
    contours = measure.find_contours(bands, 0)

    for contour in contours:

        # Contour come in units proportionnal to size of meshgrid
        # one want to scale to units of kx and ky
        kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / bObj.a
        ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / bObj.b
        vx, vy, vz = bObj.v_3D_func(kx, ky, kz)

        gamma_kz = 1 / self.tau_total_func(kx, ky, kz, vx, vy, vz)

        phi = np.rad2deg(np.arctan2(ky,kx))

        line = axes2.plot(phi, vz)
        plt.setp(line, ls ="", c = '#80ff80', lw = 3, marker = "o",
                 mfc = '#80ff80', ms = 3, mec = "#7E2320", mew= 0)  # set properties

        line = axes.plot(phi, gamma_kz)
        plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k',
                 ms = 3, mec = "#7E2320", mew= 0)  # set properties

    axes.set_xlim(0, 180)
    axes.set_xticks([0, 45, 90, 135, 180])
    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$\phi$", labelpad = 8)
    axes.set_ylabel(r"$\Gamma_{\rm tot}$ ( THz )", labelpad=8)
    axes2.set_ylabel(r"$v_{\rm z}$", rotation = 270, labelpad =25, color="#80ff80")

    kz = np.round(kz/(np.pi/bObj.c), 1)
    fig.text(0.97, 0.05, r"$k_{\rm z}$ = " + "{0:g}".format(kz) + r"$\pi$/c",
             ha="right", color="r", fontsize = 20)
    # axes.tick_params(axis='x', which='major', pad=7)
    # axes.tick_params(axis='y', which='major', pad=8)
    # axes.set_xlabel(r"$k_{\rm x}$", labelpad=8)
    # axes.set_ylabel(r"$k_{\rm y}$", labelpad=8)

    # axes.set_xticks([-pi, 0., pi])
    # axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    # axes.set_yticks([-pi, 0., pi])
    # axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

    plt.show()


def figOnekft(self, index_kf = 0, meshXY = 1001):
    mesh_graph = meshXY
    bObj = self.bandObject
    kx = np.linspace(-pi / bObj.a, pi / bObj.a, mesh_graph)
    ky = np.linspace(-pi / bObj.b, pi / bObj.b, mesh_graph)
    kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

    fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6))
    fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91)

    fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

    line = axes.contour(kxx*bObj.a, kyy*bObj.b,
                        bObj.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
    line = axes.plot(self.kft[0, index_kf,:]*bObj.a, self.kft[1, index_kf,:]*bObj.b)
    plt.setp(line, ls ="-", c = 'b', lw = 1, marker = "", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0) # trajectory
    line = axes.plot(bObj.kf[0, index_kf]*bObj.a, bObj.kf[1, index_kf]*bObj.b)
    plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 4.5, mec = "b", mew= 1.5)  # starting point
    line = axes.plot(self.kft[0, index_kf, -1]*bObj.a, self.kft[1, index_kf, -1]*bObj.b)
    plt.setp(line, ls ="", c = 'b', lw = 1, marker = "o", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0)  # end point

    axes.set_xlim(-pi, pi)
    axes.set_ylim(-pi, pi)
    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
    axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

    axes.set_xticks([-pi, 0., pi])
    axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    axes.set_yticks([-pi, 0., pi])
    axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

    plt.show()


def figOnevft(self, index_kf = 0):
    fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
    fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95)

    axes.axhline(y = 0, ls ="--", c ="k", linewidth = 0.6)

    line = axes.plot(self.time_array, self.vft[2, index_kf,:])
    plt.setp(line, ls ="-", c = '#6AFF98', lw = 3, marker = "", mfc = '#6AFF98', ms = 5, mec = "#7E2320", mew= 0)

    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$t$", labelpad = 8)
    axes.set_ylabel(r"$v_{\rm z}$", labelpad = 8)
    axes.locator_params(axis = 'y', nbins = 6)

    plt.show()

def figCumulativevft(self, index_kf = -1):
    fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
    fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95)

    line = axes.plot(self.time_array, np.cumsum(self.vft[2, index_kf, :] * exp(-self.t_o_tau[index_kf, :])))
    plt.setp(line, ls ="-", c = 'k', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties

    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$t$", labelpad = 8)
    axes.set_ylabel(r"$\sum_{\rm t}$ $v_{\rm z}(t)$$e^{\rm \dfrac{-t}{\tau}}$", labelpad = 8)

    axes.locator_params(axis = 'y', nbins = 6)

    plt.show()

def figdfdE(self):
    fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
    fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95)

    axes.axhline(y=0, ls="--", c="k", linewidth=0.6)

    dfdE_cut    = self._dfdE_cut_percent * np.abs(self.dfdE(0))
    epsilon_cut = self.energyCutOff(dfdE_cut)
    d_epsilon   = self.epsilon_array[1]-self.epsilon_array[0]
    epsilon = np.arange(-epsilon_cut-10*d_epsilon, epsilon_cut+11*d_epsilon, d_epsilon)
    dfdE = self.dfdE(epsilon)
    line = axes.plot(epsilon, -dfdE)
    plt.setp(line, ls ="-", c = 'r', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties

    line = axes.plot(epsilon_cut, dfdE_cut)
    plt.setp(line, ls ="", c = 'k', lw = 3, marker = "s", mfc = 'k', ms = 6, mec = "#7E2320", mew= 0)  # set properties
    line = axes.plot(-epsilon_cut, dfdE_cut)
    plt.setp(line, ls ="", c = 'k', lw = 3, marker = "s", mfc = 'k', ms = 6, mec = "#7E2320", mew= 0)  # set properties


    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$\epsilon$", labelpad = 8)
    axes.set_ylabel(r"-$df/d\epsilon$ ( units of t )", labelpad = 8)

    axes.locator_params(axis = 'y', nbins = 6)

    plt.show()

#---------------------------------------------------------------------------

def figParameters(self, fig_show=True):
    bObj = self.bandObject
    # (1,1) means one plot, and figsize is w x h in inch of figure
    fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
    # adjust the box of axes regarding the figure size
    fig.subplots_adjust(left=0.15, right=0.25, bottom=0.18, top=0.95)
    axes.remove()

    # Band name
    fig.text(0.72, 0.92, bObj.band_name, fontsize=20, color='#00d900')
    try:
        bObj._band_params["M"]
        fig.text(0.41, 0.92, "AF", fontsize=20,
                    color="#FF0000")
    except:
        None

    # Band Formulas
    fig.text(0.45, 0.445, "Band formula", fontsize=16,
                color='#008080')
    fig.text(0.45, 0.4, r"$a$ = " + "{0:.2f}".format(bObj.a) + r" $\AA$,  " +
                        r"$b$ = " + "{0:.2f}".format(bObj.b) + r" $\AA$,  " +
                        r"$c$ = " + "{0:.2f}".format(bObj.c) + r" $\AA$", fontsize=12)

    # r"$c$ " + "{0:.2f}".format(bObj.c)
    bandFormulaE2D = r"$\epsilon_{\rm k}^{\rm 2D}$ = - $\mu$" +\
        r" - 2$t$ (cos($k_{\rm x}a$) + cos($k_{\rm y}b$))" +\
        r" - 4$t^{'}$ (cos($k_{\rm x}a$) cos($k_{\rm y}b$))" + "\n" +\
        r"          - 2$t^{''}$ (cos(2$k_{\rm x}a$) + cos(2$k_{\rm y}b$))" + "\n"
    fig.text(0.45, 0.27, bandFormulaE2D, fontsize=10)

    bandFormulaEz = r"$\epsilon_{\rm k}^{\rm z}$   =" +\
        r" - 2$t_{\rm z}$ cos($k_{\rm z}c/2$) cos($k_{\rm x}a/2$) cos($k_{\rm y}b/2$) (cos($k_{\rm x}a$) - cos($k_{\rm y}b$))$^2$" + "\n" +\
        r"          - 2$t_{\rm z}^{'}$ cos($k_{\rm z}c/2$)"
    fig.text(0.45, 0.21, bandFormulaEz, fontsize=10)


    # AF Band Formula
    try:
        bObj._band_params["M"]
        if bObj.electronPocket == True:
            sign_symbol = "+"
        else:
            sign_symbol = "-"
        AFBandFormula = r"$\epsilon_{\rm k}^{\rm 3D " + sign_symbol + r"}$ = 1/2 ($\epsilon_{\rm k}^{\rm 2D}$ + $\epsilon_{\rm k+Q}^{\rm 2D}$) " +\
            sign_symbol + \
            r" $\sqrt{1/4(\epsilon_{\rm k}^{\rm 2D} - \epsilon_{\rm k+Q}^{\rm 2D})^2 + \Delta_{\rm AF}^2}$ + $\epsilon_{\rm k}^{\rm z}$"
        fig.text(0.45, 0.15, AFBandFormula,
                    fontsize=10, color="#FF0000")
    except:
        bandFormulaE3D = r"$\epsilon_{\rm k}^{\rm 3D}$   = $\epsilon_{\rm k}^{\rm 2D}$ + $\epsilon_{\rm k}^{\rm z}$"
        fig.text(0.45, 0.15, bandFormulaE3D, fontsize=10)


    # Scattering Formula
    fig.text(0.45, 0.08, "Scattering formula",
                fontsize=16, color='#008080')
    scatteringFormula = r"$\Gamma_{\rm tot}$ = [ $\Gamma_{\rm 0}$ + " + \
        r"$\Gamma_{\rm k}$ |cos$^{\rm n}$(2$\phi$)| + $\Gamma_{\rm DOS}^{\rm max}$ (DOS / DOS$^{\rm max}$) ] $A_{\rm arcs}$"
    fig.text(0.45, 0.03, scatteringFormula, fontsize=10)

    # Parameters Bandstructure
    fig.text(0.45, 0.92, "Band Parameters", fontsize=16,
                color='#008080')
    label_parameters = [r"t = " + "{0:.1f}".format(bObj.energy_scale) + " meV"] +\
                       [key + " = " + "{0:+.3f}".format(value) + r" $t$" for (key, value) in sorted(bObj._band_params.items()) if key!="t"]

    try:  # if it is a AF band
        bObj._band_params["M"]
        label_parameters.append(
            r"$\Delta_{\rm AF}$ =  " + "{0:+.3f}".format(bObj._band_params["M"]) + r"   $t$")
    except:
        None

    h_label = 0.88
    for label in label_parameters:
        fig.text(0.45, h_label, label, fontsize=12)
        h_label -= 0.035

    # Scattering parameters
    fig.text(0.72, 0.86, "Scattering Parameters",
                fontsize=16, color='#008080')
    label_parameters = [
        r"$\Gamma_{\rm 0}$       = " + "{0:.1f}".format(self.gamma_0) +
        "   THz",
        r"$\Gamma_{\rm DOS}^{\rm max}$   = " +
        "{0:.1f}".format(self.gamma_dos_max) + "   THz",
        r"$\Gamma_{\rm k}$       = " + "{0:.1f}".format(self.gamma_k) +
        "   THz",
        r"$n$         = " + "{0:.1f}".format(self.power),
        r"$A_{\rm arcs}$   = " + "{0:.1f}".format(self.factor_arcs),
        r"$\Gamma_{\rm tot}^{\rm max}$    = " +
        "{0:.1f}".format(self.gamma_tot_max) + "   THz",
        r"$\Gamma_{\rm tot}^{\rm min}$     = " +
        "{0:.1f}".format(self.gamma_tot_min) + "   THz",
    ]
    h_label = 0.82
    for label in label_parameters:
        fig.text(0.72, h_label, label, fontsize=12)
        h_label -= 0.035

    ## Inset FS ///////////////////////////////////////////////////////////#
    a = bObj.a
    b = bObj.b
    c = bObj.c

    mesh_graph = 201
    kx = np.linspace(-pi/a, pi/a, mesh_graph)
    ky = np.linspace(-pi/b, pi/b, mesh_graph)
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij')

    axes_FS = plt.axes([-0.02, 0.56, .4, .4])
    axes_FS.set_aspect(aspect=1)
    axes_FS.contour(kxx, kyy, bObj.e_3D_func(
        kxx, kyy, 0), 0, colors='#FF0000', linewidths=1)
    axes_FS.contour(kxx, kyy, bObj.e_3D_func(
        kxx, kyy, pi / c), 0, colors='#00DC39', linewidths=1)
    axes_FS.contour(kxx, kyy, bObj.e_3D_func(
        kxx, kyy, 2 * pi / c), 0, colors='#6577FF', linewidths=1)
    fig.text(0.30, 0.67, r"$k_{\rm z}$", fontsize=14)
    fig.text(0.30, 0.63, r"0", color='#FF0000', fontsize=14)
    fig.text(0.30, 0.60, r"$\pi$/c", color='#00DC39', fontsize=14)
    fig.text(0.30, 0.57, r"$2\pi$/c", color='#6577FF', fontsize=14)

    axes_FS.set_xlabel(r"$k_{\rm x}$", labelpad=0, fontsize=14)
    axes_FS.set_ylabel(r"$k_{\rm y}$", labelpad=-5, fontsize=14)

    axes_FS.set_xticks([-pi/a, 0., pi/a])
    axes_FS.set_xticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=14)
    axes_FS.set_yticks([-pi/b, 0., pi/b])
    axes_FS.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=14)
    # axes_FS.tick_params(axis='x', which='major', pad=7)
    # axes_FS.tick_params(axis='y', which='major', pad=8)

    ## Inset Scattering rate ////////////////////////////////////////////////#
    axes_srate = plt.axes([-0.02, 0.04, .4, .4])
    axes_srate.set_aspect(aspect=1)

    mesh_xy = 501
    kx_a = np.linspace(-pi / bObj.a, pi / bObj.a,
                       mesh_xy)
    ky_a = np.linspace(-pi / bObj.b, pi / bObj.b,
                       mesh_xy)

    kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

    bands = bObj.e_3D_func(kxx, kyy, 0)
    contours = measure.find_contours(bands, 0)

    gamma_max_list = []
    gamma_min_list = []
    for contour in contours:

        # Contour come in units proportionnal to size of meshgrid
        # one want to scale to units of kx and ky
        kx = (contour[:, 0] /
              (mesh_xy - 1) - 0.5) * 2 * pi / bObj.a
        ky = (contour[:, 1] /
              (mesh_xy - 1) - 0.5) * 2 * pi / bObj.b
        vx, vy, vz = bObj.v_3D_func(kx, ky, np.zeros_like(kx))

        gamma_kz0 = 1 / self.tau_total_func(kx, ky, 0, vx, vy, vz)
        gamma_max_list.append(np.max(gamma_kz0))
        gamma_min_list.append(np.min(gamma_kz0))

        points = np.array([kx * bObj.a,
                           ky * bObj.b]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('gnuplot'))
        lc.set_array(gamma_kz0)
        lc.set_linewidth(4)

        line = axes_srate.add_collection(lc)

    gamma_max = max(gamma_max_list)
    gamma_min = min(gamma_min_list)
    line.set_clim(gamma_min, gamma_max)
    cbar = fig.colorbar(line, ax=axes_srate)
    cbar.minorticks_on()
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel(r'$\Gamma_{\rm tot}$ ( THz )',
                       rotation=270,
                       labelpad=20,
                       fontsize=14)

    fig.text(0.295, 0.405, r"$k_{\rm z}$=0", fontsize=10, ha="right")
    axes_srate.tick_params(axis='x', which='major')
    axes_srate.tick_params(axis='y', which='major')
    axes_srate.set_xlabel(r"$k_{\rm x}$", fontsize=14)
    axes_srate.set_ylabel(r"$k_{\rm y}$", fontsize=14, labelpad=-5)

    axes_srate.set_xticks([-pi, 0., pi])
    axes_srate.set_xticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=14)
    axes_srate.set_yticks([-pi, 0., pi])
    axes_srate.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=14)

    ## Show figure ////////////////////////////////////////////////////////#
    if fig_show == True:
        plt.show()
    else:
        plt.close(fig)
    #//////////////////////////////////////////////////////////////////////////////#

    return fig
