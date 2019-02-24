import numpy as np
from numpy import cos, sin, pi, sqrt, ones
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

class ADMR:
    def __init__(self, initialcondObjectList, Btheta_min=0, Btheta_max=110,
                 Btheta_step=5, Bphi_array=[0, 15, 30, 45]):

        # Band dictionary
        self.initialCondObjectDict = {} # will contain the condObject for each band, with key their bandname
        for condObject in initialcondObjectList:
            self.initialCondObjectDict[condObject.bandObject.bandname] = condObject
        self.bandNamesList = list(self.initialCondObjectDict.keys())

        # Magnetic field
        self.Btheta_min   = 0           # in degrees
        self.Btheta_max   = 110         # in degrees
        self.Btheta_step  = 5           # in degrees
        self.Btheta_array = np.arange(self.Btheta_min, self.Btheta_max + self.Btheta_step, self.Btheta_step)
        self.Bphi_array = np.array(Bphi_array)

        # Conductivity dictionaries
        self.condObjectDict = {}        # will implicitely contain all results of runADMR
        self.kftDict = {}
        self.vftDict = {}

        # Resistivity rray rho_zz / rho_zz(0)
        self.rzz_array = None


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def runADMR(self):
        rho_zz_array = np.empty((self.Bphi_array.shape[0], self.Btheta_array.shape[0]), dtype= np.float64)

        for l, phi in enumerate(self.Bphi_array):
            for m, theta in enumerate(self.Btheta_array):

                sigma_zz = 0
                for (bandname, iniCondObject) in list(self.initialCondObjectDict.items()):

                    iniCondObject.Bphi = phi
                    iniCondObject.Btheta = theta

                    iniCondObject.solveMovementFunc()
                    iniCondObject.chambersFunc(i=2, j=2)

                    sigma_zz += iniCondObject.sigma[2, 2]

                    # Store in dictionaries
                    self.condObjectDict[bandname, phi, theta] = iniCondObject
                    self.kftDict[bandname, phi, theta] = iniCondObject.kft
                    self.vftDict[bandname, phi, theta] = iniCondObject.vft
                    # self.vproduct_dict[phi, theta] = condObject.VelocitiesProduct(i=2, j=2)

                rho_zz_array[l, m] = 1 / sigma_zz

        rho_zz_0_array = np.outer(rho_zz_array[:, 0], np.ones(self.Btheta_array.shape[0]))
        self.rzz_array = rho_zz_array / rho_zz_0_array

    def fileNameFunc(self):
        # To point to bandstructure parameters, we use just one band
        # as they should share the same parameters
        CondObject0 = self.initialCondObjectDict[self.bandNamesList[0]]
        bandObject0 = CondObject0.bandObject

        # Detect if bands come from the AF reconstruction
        try:
            bandObject0.M # if M exists in one of the band (meaning AF FSR)
            bandAF = True
        except:
            bandAF = False

        # Create the list of parameters for the filename
        file_parameters_list  = [r"p"   + "{0:.3f}".format(bandObject0.p),
                                 r"B"   + "{0:.0f}".format(CondObject0.Bamp),
                                 r"t"   + "{0:.1f}".format(bandObject0.t),
                                 r"mu"  + "{0:.3f}".format(bandObject0.mu),
                                 r"tp"  + "{0:.3f}".format(bandObject0.tp),
                                 r"tpp" + "{0:.3f}".format(bandObject0.tpp),
                                 r"tz"  + "{0:.3f}".format(bandObject0.tz),
                                 r"tzz" + "{0:.3f}".format(bandObject0.tz2)
                                ]

        if bandAF == True:
            file_parameters_list.append(r"M" + "{0:.3f}".format(bandObject0.M))

        for (bandname, iniCondObject) in self.initialCondObjectDict.items():
            file_parameters_list.extend([bandname,
                                         r"gzero" + "{0:.1f}".format(iniCondObject.gamma_0),
                                         r"gk"  + "{0:.1f}".format(iniCondObject.gamma_k),
                                         r"pwr" + "{0:.0f}".format(iniCondObject.power)
                                        ])

        if bandAF == True:
            file_name = "Rzz_AF"
        else:
            file_name = "Rzz"

        for string in file_parameters_list:
            file_name += "_" + string
        return file_name



    def fileADMR(self, folder=""):
        # To point to bandstructure parameters, we use just one band
        # as they should share the same parameters
        CondObject0 = self.initialCondObjectDict[self.bandNamesList[0]]
        bandObject0 = CondObject0.bandObject

        # Detect if bands come from the AF reconstruction
        try:
            bandObject0.M # if M exists in one of the band (meaning AF FSR)
            bandAF = True
        except:
            bandAF = False

        # Build Data 2D-array
        Ones = np.ones_like(self.Btheta_array) # column of 1 with same size of Btheta_array
        rzzMatrix = self.rzz_array[0,:] # initialize with first phi value
        rzzHeader = "rzz(phi=" + str(self.Bphi_array[0])+ ")\t"
        for l, phi in enumerate(self.Bphi_array):
            if l==0:
                pass # because we already have rzz_array for the inital value of phi
            else:
                rzzMatrix = np.vstack((rzzMatrix, self.rzz_array[l,:]))
                rzzHeader += "rzz(phi=" + str(self.Bphi_array[l])+ ")\t"

        Data = np.vstack((self.Btheta_array, rzzMatrix,
                          CondObject0.Bamp * Ones,
                          bandObject0.t    * Ones, bandObject0.tp * Ones,
                          bandObject0.tpp  * Ones, bandObject0.tz * Ones,
                          bandObject0.tz2  * Ones, bandObject0.mu * Ones,
                          bandObject0.mesh_ds * Ones, bandObject0.numberOfKz * Ones))

        if bandAF == True:
            Data = np.vstack((Data, bandObject0.M*Ones))

        condHeader = ""
        for (bandname, iniCondObject) in self.initialCondObjectDict.items():
            Data = np.vstack((Data,
                              iniCondObject.gamma_0 * Ones,
                              iniCondObject.gamma_k * Ones,
                              iniCondObject.power   * Ones))
            condHeader += bandname + "_g_0[THz]\t" + \
                          bandname + "_g_k[THz]\t" + \
                          bandname + "_power\t"

        Data = Data.transpose()

        # Build header
        DataHeader = "theta[deg]\t" + rzzHeader + \
                     "B[T]\tt[meV]\ttp\ttpp\ttz\ttz2\tmu\tmesh_ds\tmesh_z\t" + \
                     condHeader

        np.savetxt(folder + "/" + self.fileNameFunc() + ".dat", Data, fmt='%.7e',
        header = DataHeader, comments = "#")



    def figADMR(self, fig_show=True, fig_save=True, folder=""):
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

        ## Create the list object of figures
        fig_list = []

        ## Bands figures //////////////////////////////////////////////////////#
        for (bandname, iniCondObject) in self.initialCondObjectDict.items():
            fig, axes = plt.subplots(1, 1, figsize = (10.5, 5.8)) # (1,1) means one plot, and figsize is w x h in inch of figure
            fig.subplots_adjust(left = 0.15, right = 0.75, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

            # Labels
            label_parameters = [r"$p$ = " + "{0:.3f}".format(iniCondObject.bandObject.p),
                                r"$B$ = " + "{0:.0f}".format(iniCondObject.Bamp) + " T",
                                "",
                                r"$\Gamma_{\rm 0}$ = " + "{0:.1f}".format(iniCondObject.gamma_0) + " THz",
                                r"$\Gamma_{\rm k}$ = " + "{0:.1f}".format(iniCondObject.gamma_k) + " THz",
                                r"power = " + "{0:.0f}".format(iniCondObject.power),
                                "",
                                r"$t$ = " + "{0:.1f}".format(iniCondObject.bandObject.t) + " meV",
                                r"$\mu$ = " + "{0:.3f}".format(iniCondObject.bandObject.mu) + r" $t$",
                                r"$t^\prime$ = " + "{0:.3f}".format(iniCondObject.bandObject.tp) + r" $t$",
                                r"$t^{\prime\prime}$ = " + "{0:.3f}".format(iniCondObject.bandObject.tpp) + r" $t$",
                                r"$t_{\rm z}$ = " + "{0:.3f}".format(iniCondObject.bandObject.tz) + r" $t$",
                                r"$t_{\rm z}^{\prime}$ = " + "{0:.3f}".format(iniCondObject.bandObject.tz2) + r" $t$"]

            h_label = 0.92
            for label in label_parameters:
                fig.text(0.78, h_label, label, fontsize = 14)
                h_label -= 0.04

            ## Inset FS ///////////////////////////////////////////////////////////#
            a = iniCondObject.bandObject.a
            b = iniCondObject.bandObject.b
            c = iniCondObject.bandObject.c

            mesh_graph = 201
            kx = np.linspace(-pi/a, pi/a, mesh_graph)
            ky = np.linspace(-pi/b, pi/b, mesh_graph)
            kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

            axes_inset_FS = plt.axes([0.74, 0.18, .18, .18])
            axes_inset_FS.set_aspect(aspect=1)
            axes_inset_FS.contour(kxx, kyy, iniCondObject.bandObject.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 1)
            axes_inset_FS.annotate(r"0", xy = (- pi/a * 0.9, pi/b * 0.75), color = 'r', fontsize = 8)
            axes_inset_FS.contour(kxx, kyy, iniCondObject.bandObject.e_3D_func(kxx, kyy, pi / c), 0, colors = '#00DC39', linewidths = 1)
            axes_inset_FS.annotate(r"$\pi$/c", xy = (- pi/a * 0.9, - pi/b * 0.9), color = '#00DC39', fontsize = 8)
            axes_inset_FS.contour(kxx, kyy, iniCondObject.bandObject.e_3D_func(kxx, kyy, 2 * pi / c), 0, colors = '#6577FF', linewidths = 1)
            axes_inset_FS.annotate(r"$2\pi$/c", xy = (pi/a * 0.5, - pi/b * 0.9), color = '#6577FF', fontsize = 8)
            axes_inset_FS.set_xlim(-pi/a,pi/a)
            axes_inset_FS.set_ylim(-pi/b,pi/b)
            axes_inset_FS.set_xticks([])
            axes_inset_FS.set_yticks([])
            axes_inset_FS.axis(**{'linewidth' : 0.2})

            ## Inset Life Time ////////////////////////////////////////////////////#
            tau_0 = iniCondObject.tau_0
            gamma_0 = iniCondObject.gamma_0
            gamma_k = iniCondObject.gamma_k
            power = iniCondObject.power

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

            fig_list.append(fig)

        ## ADMR figure ////////////////////////////////////////////////////////#
        fig, axes = plt.subplots(1, 1, figsize = (10.5, 5.8)) # (1,1) means one plot, and figsize is w x h in inch of figure
        fig.subplots_adjust(left = 0.15, right = 0.75, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

        axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

        ## Colors
        if self.Bphi_array.shape[0] > 4:
            cmap = mpl.cm.get_cmap('jet', self.Bphi_array.shape[0])
            colors = cmap(np.arange(self.Bphi_array.shape[0]))
        else:
            colors = ['#000000', '#3B528B', '#FF0000', '#C7E500']

        ## Plot ADMR
        for i, B_phi in enumerate(self.Bphi_array):
            line = axes.plot(self.Btheta_array, self.rzz_array[i, :], label = r"$\phi$ = " + r"{0:.0f}".format(B_phi))
            plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "o", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)  # set properties

        axes.set_xlim(0, self.Btheta_max)
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
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

        fig_list.append(fig)


        ## Save figure ////////////////////////////////////////////////////////#
        if fig_save == True:
            file_figures = PdfPages(folder + "/" + self.fileNameFunc() + ".pdf")
            for fig in fig_list[::-1]:
                file_figures.savefig(fig)
            file_figures.close()

        ## Show figure ////////////////////////////////////////////////////////#
        if fig_show == True:
            plt.show()
