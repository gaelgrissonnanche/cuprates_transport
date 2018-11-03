## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import pi, ones
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import time

from band_structure import *
from movement_equation import *
from chambers_formula import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


def admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = True):
    a = band_parameters[0]
    b = band_parameters[1]
    c = band_parameters[2]
    mu = band_parameters[3]
    t = band_parameters[4]
    tp = band_parameters[5]
    tpp = band_parameters[6]
    tz = band_parameters[7]
    tz2 = band_parameters[8]

    mesh_xy   = mesh_parameters[0]
    mesh_z    = mesh_parameters[1]

    gamma_0 = tau_parameters[0]
    gamma_k = tau_parameters[1]
    power   = tau_parameters[2]
    tau_0 = 1 / gamma_0

    ## Start computing time
    start_total_time = time.time()

    ## Discretize Fermi Surface >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

    ## Make mesh_xy a multiple of 4 to respect the 4-order symmetry
    mesh_xy = mesh_xy - (mesh_xy % 4)
    ## Discretize FS
    kf, vf, dkf, numberPointsPerKz_list = discretize_FS(band_parameters, mesh_parameters)

    ## Compute the Hole doping >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    p = dopingFunc(band_parameters)[0]
    print("p = " + "{0:.3f}".format(p))

    ## ADMR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    rho_zz_a = RzzAngleFunc(B_amp, B_theta_a, B_phi_a, kf, vf, dkf, band_parameters, tau_parameters)
    rho_zz_0 = rho_zz_a[:,0]

    ## End computing time
    print("Total time : %.6s seconds" % (time.time() - start_total_time))


    ## Save Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    len = rho_zz_a.shape[1]
    t = band_parameters[4]
    Data = np.vstack((B_theta_a, rho_zz_a[0,:] / rho_zz_0[0], rho_zz_a[1,:] / rho_zz_0[1], rho_zz_a[2,:] / rho_zz_0[2], rho_zz_a[3,:] / rho_zz_0[3],
                      B_amp*ones(len), gamma_0*ones(len), gamma_k*ones(len), power*ones(len), t*ones(len), tp*ones(len), tpp*ones(len), tz*ones(len), tz2*ones(len), mu*ones(len), mesh_xy*ones(len), mesh_z*ones(len)))
    Data = Data.transpose()
    folder = "results_sim/"
    file_name_parameters  = [r"p_"   + "{0:.3f}".format(p),
                             r"B_"   + "{0:.0f}".format(B_amp),
                             r"g0_"  + "{0:.1f}".format(gamma_0),
                             r"gk_"  + "{0:.1f}".format(gamma_k),
                             r"pwr_" + "{0:.0f}".format(power),
                             r"t_"   + "{0:.1f}".format(t),
                             r"mu_"  + "{0:.4f}".format(mu/t),
                             r"tp_"  + "{0:.3f}".format(tp/t),
                             r"tpp_" + "{0:.3f}".format(tpp/t),
                             r"tz_"  + "{0:.3f}".format(tz/t),
                             r"tz2_" + "{0:.3f}".format(tz2/t)]

    file_name =  "Rzz"
    for string in file_name_parameters:
        file_name += "_" + string

    np.savetxt(folder + file_name + ".dat", Data, fmt='%.7e',
    header = "theta[deg]\trzz(phi=0)/rzz_0\trzz(phi=15)/rzz_0\trzz(phi=30)/rzz_0\trzz(phi=45)/rzz_0\tB[T]\tgamma_0[THz]\tgamma_k[THz]\tpower\tt[meV]\ttp\ttpp\ttz\ttz2\tmu\tmesh_xy\tmesh_z", comments = "#")


    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
    ## Figures ////////////////////////////////////////////////////////////////#
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

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

    #>>>> Rzz vs theta >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
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
    label_parameters = [r"$p$ = " + "{0:.3f}".format(p),
                        r"$B$ = " + "{0:.0f}".format(B_amp) + " T",
                        "",
                        r"$\Gamma_{\rm 0}$ = " + "{0:.1f}".format(gamma_0) + " THz",
                        r"$\Gamma_{\rm k}$ = " + "{0:.1f}".format(gamma_k) + " THz",
                        r"power = " + "{0:.0f}".format(power),
                        "",
                        r"$t$ = " + "{0:.1f}".format(t) + " meV",
                        r"$\mu$ = " + "{0:.4f}".format(mu/t) + r" $t$",
                        r"$t^\prime$ = " + "{0:.3f}".format(tp/t) + r" $t$",
                        r"$t^{\prime\prime}$ = " + "{0:.3f}".format(tpp/t) + r" $t$",
                        r"$t_{\rm z}$ = " + "{0:.3f}".format(tz/t) + r" $t$",
                        r"$t_{\rm z}^{\prime}$ = " + "{0:.3f}".format(tz2/t) + r" $t$"]

    h_label = 0.92
    for label in label_parameters:
        fig.text(0.78, h_label, label, fontsize = 14)
        h_label -= 0.04

    colors = ['k', '#3B528B', 'r', '#C7E500']

    for i, B_phi in enumerate(B_phi_a):
        line = axes.plot(B_theta_a * 180 / pi, rho_zz_a[i,:] / rho_zz_0[i], label = r"$\phi$ = " + r"{0:.0f}".format(B_phi * 180 / pi))
        plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "o", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)  # set properties

    axes.set_xlim(0, 110)
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
    axes_inset_FS.contour(kxx, kyy, e_3D_func(kxx, kyy, 0, band_parameters), 0, colors = '#FF0000', linewidths = 1)
    axes_inset_FS.annotate(r"0", xy = (- pi/a * 0.9, pi/b * 0.75), color = 'r', fontsize = 8)
    axes_inset_FS.contour(kxx, kyy, e_3D_func(kxx, kyy, pi / c, band_parameters), 0, colors = '#00DC39', linewidths = 1)
    axes_inset_FS.annotate(r"$\pi$/c", xy = (- pi/a * 0.9, - pi/b * 0.9), color = '#00DC39', fontsize = 8)
    axes_inset_FS.contour(kxx, kyy, e_3D_func(kxx, kyy, 2 * pi / c, band_parameters), 0, colors = '#6577FF', linewidths = 1)
    axes_inset_FS.annotate(r"$2\pi$/c", xy = (pi/a * 0.5, - pi/b * 0.9), color = '#6577FF', fontsize = 8)
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
    tau_k_x = 1 / (gamma_0 + gamma_k * (sin(phi)**2 - cos(phi)**2)**power) * cos(phi)
    tau_k_y = 1 / (gamma_0 + gamma_k * (sin(phi)**2 - cos(phi)**2)**power) * sin(phi)
    line = axes_inset_tau.plot(tau_k_x / tau_0, tau_k_y / tau_0, clip_on = False)
    plt.setp(line, ls ="-", c = '#FF9C54', lw = 1, marker = "", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
    axes_inset_tau.annotate(r"$\tau_{\rm k}$", xy = (0.4, 0.45), color = '#FF9C54', fontsize = 10)
    ## tau_k_min
    phi_min = 3 * pi / 2
    tau_k_x_min = 1 / (gamma_0 + gamma_k * (sin(phi_min)**2 - cos(phi_min)**2)**power) * cos(phi_min)
    tau_k_y_min = 1 / (gamma_0 + gamma_k * (sin(phi_min)**2 - cos(phi_min)**2)**power) * sin(phi_min)
    line = axes_inset_tau.plot(tau_k_x_min / tau_0, tau_k_y_min / tau_0, clip_on = False)
    plt.setp(line, ls ="", c = '#FF9C54', lw = 3, marker = "o", mfc = '#FF9C54', ms = 4, mec = "#7E2320", mew= 0)
    fraction = np.abs(np.round(tau_k_y_min / tau_0, 2))
    axes_inset_tau.annotate(r"{0:.2f}".format(fraction) + r"$\tau_{\rm 0}$", xy = (-0.35, tau_k_y_min / tau_0 * 0.8), color = '#FF9C54', fontsize = 8)

    axes_inset_tau.set_xlim(-1,1)
    axes_inset_tau.set_ylim(-1,1)
    axes_inset_tau.set_xticks([])
    axes_inset_tau.set_yticks([])
    axes_inset_tau.axis(**{'linewidth' : 0.2})

    if fig_show == True:
        plt.show()

    folder = "results_sim/"
    fig.savefig(folder + file_name + ".pdf")
    #//////////////////////////////////////////////////////////////////////////#