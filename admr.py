import numpy as np
from numpy import cos, sin, pi, sqrt, ones
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

class ADMR:
    def __init__(self, bandObject, Bamp=45, gamma_0=152, gamma_k=649, power=12, a0 = 0):
        # Band object
        self.bandObject = bandObject ## WARNING do not modify within this object

        # Scattering rate
        self.gamma_0 = gamma_0 # in THz
        self.gamma_k = gamma_k # in THz
        self.power   = int(power)
        if self.power % 2 == 1:
            self.power += 1
        self.a0 = a0

        # Magnetic field
        self.Bamp         = Bamp # in Tesla
        self.Btheta_min   = 0 # in degrees
        self.Btheta_max   = 110 # in degrees
        self.Btheta_step  = 5 # in degress
        self.Btheta_array = np.arange(self.Btheta_min, self.Btheta_max + self.Btheta_step, self.Btheta_step)
        self.Bphi_array   = np.array([0, 15, 30, 45])

        # Array r_zz(phi, theta) = rho_zz(phim theta) / rho_zz(phi, theta = 0)
        self.rzz_array = None
        self.condObject_dict = {}

        # Time parameters
        self.tau_0 = 1 / self.gamma_0 # in picoseconds
        self.tmax = 10 * self.tau_0 # in picoseconds
        self.Ntime = 300 # number of steps in time
        self.dt = self.tmax / self.Ntime
        self.t = np.arange(0, self.tmax, self.dt)

        # Time-dependent kf, vf dict[phi, theta]
        self.kft_dict = {}
        self.vft_dict = {}

    def runADMR(self):
        rho_zz_array = np.empty((self.Bphi_array.shape[0], self.Btheta_array.shape[0]), dtype = np.float64)

        for l, phi in enumerate(self.Bphi_array):
            for m, theta in enumerate(self.Btheta_array):

                condObject = Conductivity(self.bandObject, self.Bamp, phi, theta,
                                        self.gamma_0, self.gamma_k, self.power, self.a0)
                condObject.tau_0 = self.tau_0
                condObject.tmax  = self.tmax
                condObject.Ntime = self.Ntime
                condObject.dt    = self.dt
                condObject.t     = self.t

                condObject.solveMovementFunc()
                condObject.chambersFunc(i = 2, j = 2)

                rho_zz_array[l, m] = 1 / condObject.sigma[2,2]
                self.condObject_dict[phi, theta] = condObject
                self.kft_dict[phi, theta] = condObject.kft
                self.vft_dict[phi, theta] = condObject.vft

        rho_zz_0_array = np.outer(rho_zz_array[:, 0], np.ones(self.Btheta_array.shape[0]))
        self.rzz_array = rho_zz_array / rho_zz_0_array

    def fileNameFunc(self):
        file_parameters_list  = [r"p_"   + "{0:.3f}".format(self.bandObject.p),
                                 r"B_"   + "{0:.0f}".format(self.Bamp),
                                 r"g0_"  + "{0:.1f}".format(self.gamma_0),
                                 r"gk_"  + "{0:.1f}".format(self.gamma_k),
                                 r"pwr_"   + "{0:.0f}".format(self.power),
                                 r"t_"   + "{0:.1f}".format(self.bandObject.t),
                                 r"mu_"  + "{0:.3f}".format(self.bandObject.mu),
                                 r"tp_"  + "{0:.3f}".format(self.bandObject.tp),
                                 r"tpp_" + "{0:.3f}".format(self.bandObject.tpp),
                                 r"tz_"  + "{0:.3f}".format(self.bandObject.tz),
                                 r"tz2_" + "{0:.3f}".format(self.bandObject.tz2)]
        file_name =  "Rzz"
        for string in file_parameters_list:
            file_name += "_" + string

        return file_name

    def fileADMR(self, file_folder = "results_sim"):
        array_1 = np.ones(self.rzz_array.shape[1])
        Data = np.vstack((self.Btheta_array, self.rzz_array[0,:], self.rzz_array[1,:], self.rzz_array[2,:], self.rzz_array[3,:],
                          self.Bamp*array_1, self.gamma_0*array_1, self.gamma_k*array_1, self.power*array_1, self.bandObject.t*array_1, self.bandObject.tp*array_1, self.bandObject.tpp*array_1, self.bandObject.tz*array_1, self.bandObject.tz2*array_1, self.bandObject.mu*array_1, self.bandObject.mesh_ds*array_1, self.bandObject.numberOfKz*array_1))
        Data = Data.transpose()

        np.savetxt(file_folder + "/" + self.fileNameFunc() + ".dat", Data, fmt='%.7e',
        header = "theta[deg]\trzz(phi=0)\trzz(phi=15)\trzz(phi=30)\trzz(phi=45)\tB[T]\tgamma_0[THz]\tgamma_k[THz]\tpower\tt[meV]\ttp\ttpp\ttz\ttz2\tmu\tmesh_ds\tmesh_z", comments = "#")


    def figADMR(self, fig_show = True, fig_save = True, fig_folder = "results_sim"):
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

        ## Master figure //////////////////////////////////////////////////////#
        fig, axes = plt.subplots(1, 1, figsize = (10.5, 5.8)) # (1,1) means one plot, and figsize is w x h in inch of figure
        fig.subplots_adjust(left = 0.15, right = 0.75, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

        axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

        for tick in axes.xaxis.get_major_ticks():
            tick.set_pad(7)
        for tick in axes.yaxis.get_major_ticks():
            tick.set_pad(8)

        ## Labels
        label_parameters = [r"$p$ = " + "{0:.3f}".format(self.bandObject.p),
                            r"$B$ = " + "{0:.0f}".format(self.Bamp) + " T",
                            "",
                            r"$\Gamma_{\rm 0}$ = " + "{0:.1f}".format(self.gamma_0) + " THz",
                            r"$\Gamma_{\rm k}$ = " + "{0:.1f}".format(self.gamma_k) + " THz",
                            r"power = " + "{0:.0f}".format(self.power),
                            "",
                            r"$t$ = " + "{0:.1f}".format(self.bandObject.t) + " meV",
                            r"$\mu$ = " + "{0:.3f}".format(self.bandObject.mu) + r" $t$",
                            r"$t^\prime$ = " + "{0:.3f}".format(self.bandObject.tp) + r" $t$",
                            r"$t^{\prime\prime}$ = " + "{0:.3f}".format(self.bandObject.tpp) + r" $t$",
                            r"$t_{\rm z}$ = " + "{0:.3f}".format(self.bandObject.tz) + r" $t$",
                            r"$t_{\rm z}^{\prime}$ = " + "{0:.3f}".format(self.bandObject.tz2) + r" $t$"]

        h_label = 0.92
        for label in label_parameters:
            fig.text(0.78, h_label, label, fontsize = 14)
            h_label -= 0.04

        ## Colors
        if self.Bphi_array.shape[0] > 4:
            cmap = mpl.cm.get_cmap('jet', self.Bphi_array.shape[0])
            colors = cmap(np.arange(self.Bphi_array.shape[0]))
        else:
            colors = ['k', '#3B528B', 'r', '#C7E500']

        ## Plot ADMR
        for i, B_phi in enumerate(self.Bphi_array):
            line = axes.plot(self.Btheta_array, self.rzz_array[i, :], label = r"$\phi$ = " + r"{0:.0f}".format(B_phi))
            plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "o", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)  # set properties

        axes.set_xlim(0, self.Btheta_max)
        axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
        axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)

        plt.legend(loc = 0, fontsize = 14, frameon = False, numpoints = 1, markerscale = 1, handletextpad = 0.5)

        ## Set ticks space and minor ticks
        xtics = 30. # space between two ticks
        mxtics = xtics / 2.  # space between two minor ticks
        majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks
        axes.xaxis.set_major_locator(MultipleLocator(xtics))
        axes.xaxis.set_major_formatter(majorFormatter)
        axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
        axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axes.locator_params(axis = 'y', nbins = 6)

        ## Inset FS ///////////////////////////////////////////////////////////#
        a = self.bandObject.a
        b = self.bandObject.b
        c = self.bandObject.c

        mesh_graph = 201
        kx = np.linspace(-pi/a, pi/a, mesh_graph)
        ky = np.linspace(-pi/b, pi/b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

        axes_inset_FS = plt.axes([0.74, 0.18, .18, .18])
        axes_inset_FS.set_aspect(aspect=1)
        axes_inset_FS.contour(kxx, kyy, self.bandObject.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 1)
        axes_inset_FS.annotate(r"0", xy = (- pi/a * 0.9, pi/b * 0.75), color = 'r', fontsize = 8)
        axes_inset_FS.contour(kxx, kyy, self.bandObject.e_3D_func(kxx, kyy, pi / c), 0, colors = '#00DC39', linewidths = 1)
        axes_inset_FS.annotate(r"$\pi$/c", xy = (- pi/a * 0.9, - pi/b * 0.9), color = '#00DC39', fontsize = 8)
        axes_inset_FS.contour(kxx, kyy, self.bandObject.e_3D_func(kxx, kyy, 2 * pi / c), 0, colors = '#6577FF', linewidths = 1)
        axes_inset_FS.annotate(r"$2\pi$/c", xy = (pi/a * 0.5, - pi/b * 0.9), color = '#6577FF', fontsize = 8)
        axes_inset_FS.set_xlim(-pi/a,pi/a)
        axes_inset_FS.set_ylim(-pi/b,pi/b)
        axes_inset_FS.set_xticks([])
        axes_inset_FS.set_yticks([])
        axes_inset_FS.axis(**{'linewidth' : 0.2})


        ## Inset Life Time ////////////////////////////////////////////////////#
        tau_0 = self.tau_0
        gamma_0 = self.gamma_0
        gamma_k = self.gamma_k
        power = self.power

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

        ## Save figure
        if fig_save == True:
            fig.savefig(fig_folder + "/" + self.fileNameFunc() + ".pdf")
        if fig_show == True:
            plt.show()