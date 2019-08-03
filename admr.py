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
        self.totalFilling = 0 # total bands filling (of electron) over all bands
        for condObject in initialcondObjectList:
            self.totalFilling += condObject.bandObject.n
            self.initialCondObjectDict[condObject.bandObject.bandname] = condObject
        self.bandNamesList = list(self.initialCondObjectDict.keys())
        self.totalHoleDoping = 1 - self.totalFilling # total bands hole doping over all bands


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

        print("------------------------------------------------")
        print("Start ADMR computation")

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

        print("End ADMR computation")
        print("------------------------------------------------")

    #---------------------------------------------------------------------------
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
        file_parameters_list  = [r"p"   + "{0:.3f}".format(self.totalHoleDoping),
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
                                         r"gdos" + "{0:.1f}".format(iniCondObject.gamma_dos),
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

    #---------------------------------------------------------------------------
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
        for l in range(self.Bphi_array.shape[0]):
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
                              iniCondObject.gamma_dos * Ones,
                              iniCondObject.gamma_k * Ones,
                              iniCondObject.power   * Ones))
            condHeader += bandname + "_g_0[THz]\t" + \
                          bandname + "_g_dos[THz]\t" + \
                          bandname + "_g_k[THz]\t" + \
                          bandname + "_power\t"

        Data = Data.transpose()

        # Build header
        DataHeader = "theta[deg]\t" + rzzHeader + \
                     "B[T]\tt[meV]\ttp\ttpp\ttz\ttz2\tmu\tmesh_ds\tmesh_z\t" + \
                     condHeader

        np.savetxt(folder + "/" + self.fileNameFunc() + ".dat", Data, fmt='%.7e',
        header = DataHeader, comments = "#")


    #---------------------------------------------------------------------------
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

        ## Parameters figures ///////////////////////////////////////////////////#
        for iniCondObject in self.initialCondObjectDict.values():
            fig_list.append(iniCondObject.figParameters(fig_show=False))

        ## ADMR figure ////////////////////////////////////////////////////////#
        fig, axes = plt.subplots(1, 1, figsize = (10.5, 5.8)) # (1,1) means one plot, and figsize is w x h in inch of figure
        fig.subplots_adjust(left = 0.15, right = 0.75, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

        axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

        # To point to bandstructure parameters, we use just one band
        # as they should share the same parameters
        CondObject0 = self.initialCondObjectDict[self.bandNamesList[0]]
        bandObject0 = CondObject0.bandObject

        # Labels
        fig.text(0.8, 0.9, r"$B$ = " + "{0:.0f}".format(CondObject0.Bamp) + " T")
        fig.text(0.8, 0.84, r"$p$ = " + "{0:.3f}".format(self.totalHoleDoping))

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
        axes.locator_params(axis='y', nbins=6)

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
