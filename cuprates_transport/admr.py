import numpy as np
from numpy import cos, sin, pi, sqrt, ones
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

class ADMR:
    def __init__(self, condObject_list, Btheta_min=0, Btheta_max=110,
                 Btheta_step=5, Bphi_array=[0, 15, 30, 45], show_progress=True, **trash):
        # Band dictionary
        self.condObject_dict = {} # will contain the condObject for each band, with key their bandname
        self.total_filling = 0 # total bands filling (of electron) over all bands
        for condObject in condObject_list:
            self.total_filling += condObject.bandObject.n
            self.condObject_dict[condObject.bandObject.band_name] = condObject
        self.band_names = list(self.condObject_dict.keys())
        self.total_hole_doping = 1 - self.total_filling # total bands hole doping over all bands

        ## Miscellaneous
        self.show_progress = show_progress # shows progress bar or not

        # Magnetic field
        self.Btheta_min   = Btheta_min    # in degrees
        self.Btheta_max   = Btheta_max    # in degrees
        self.Btheta_step  = Btheta_step  # in degrees
        self.Btheta_array = np.arange(self.Btheta_min, self.Btheta_max + self.Btheta_step, self.Btheta_step)
        self.Bphi_array = np.array(Bphi_array)

        # Resistivity array rho_zz
        self.rhozz_array = None
        # Resistivity array rho_zz / rho_zz(0)
        self.rzz_array = None


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def runADMR(self):
        rhozz_array = np.empty((self.Bphi_array.size, self.Btheta_array.size), dtype= np.float64)
        if self.show_progress is True:
            iterator = enumerate(tqdm(self.Bphi_array, ncols=80, unit="phi", desc="ADMR"))
        else:
            iterator = enumerate(self.Bphi_array)
        for l, phi in iterator:
            for m, theta in enumerate(self.Btheta_array):
                sigma_tensor = 0
                for _, condObject in list(self.condObject_dict.items()):
                    condObject.Bphi = phi
                    condObject.Btheta = theta
                    condObject.runTransport()
                    sigma_tensor += condObject.sigma
                rho = np.linalg.inv(sigma_tensor)
                rhozz_array[l, m] = rho[2,2]
        rhozz_0_array = np.outer(rhozz_array[:, 0], np.ones(self.Btheta_array.shape[0]))
        self.rhozz_array = rhozz_array
        self.rzz_array = rhozz_array / rhozz_0_array

    #---------------------------------------------------------------------------
    def file_name_func(self):
        # To point to bandstructure parameters, we use just one band
        # as they should share the same parameters
        condObject0 = self.condObject_dict[self.band_names[0]]
        bandObject0 = condObject0.bandObject

        # Detect if bands come from the AF reconstruction
        try:
            bandObject0._band_params["M"] # if M exists in one of the band (meaning AF FSR)
            bandAF = True
        except:
            bandAF = False

        # Create the list of parameters for the filename
        file_parameters_list  = [r"p"   + "{0:.3f}".format(self.total_hole_doping),
                                 r"T"   + "{0:.0f}".format(condObject0.T),
                                 r"B"   + "{0:.0f}".format(condObject0.Bamp),
                                 r"t" + "{0:.1f}".format(bandObject0.energy_scale)] +\
        [key + "{0:.3f}".format(value) for (key, value) in sorted(bandObject0._band_params.items()) if key!="t"]

        if bandAF == True:
            file_parameters_list.append(r"M" + "{0:.3f}".format(bandObject0._band_params["M"]))

        for (band_name, condObject) in self.condObject_dict.items():
            file_parameters_list.extend([band_name,
                                         r"gzero" + "{0:.1f}".format(condObject.gamma_0),
                                         r"gdos" + "{0:.1f}".format(condObject.gamma_dos_max),
                                         r"gk"  + "{0:.1f}".format(condObject.gamma_k),
                                         r"pwr" + "{0:.1f}".format(condObject.power),
                                         r"arc" + "{0:.1f}".format(condObject.factor_arcs)
                                        ])

        if bandAF == True:
            file_name = "AF"
        else:
            file_name = ""

        for i, string in enumerate(file_parameters_list):
            if i==0:
                file_name += string
            else:
                file_name += "_" + string
        return file_name

    #---------------------------------------------------------------------------
    def fileADMR(self, folder="", filename=None):
        # To point to bandstructure parameters, we use just one band
        # as they should share the same parameters
        condObject0 = self.condObject_dict[self.band_names[0]]
        bandObject0 = condObject0.bandObject

        # Detect if bands come from the AF reconstruction
        try:
            bandObject0._band_params["M"] # if M exists in one of the band (meaning AF FSR)
            bandAF = True
        except:
            bandAF = False

        ## Build Data 2D-array & Header -----------------------------------------
        Ones = np.ones_like(self.Btheta_array) # column of 1 with same size of Btheta_array
        rhozzMatrix = self.rhozz_array[0,:] # initialize with first phi value
        rhozzHeader = "rhozz(phi=" + str(self.Bphi_array[0])+ ")[Ohm.m]\t"
        for l in range(self.Bphi_array.size):
            if l==0:
                pass # because we already have rhozz_array for the inital value of phi
            else:
                rhozzMatrix = np.vstack((rhozzMatrix, self.rhozz_array[l,:]))
                rhozzHeader += "rho_zz(phi=" + str(self.Bphi_array[l])+ ")[Ohm.m]\t"

        Data = np.vstack((self.Btheta_array, rhozzMatrix, condObject0.Bamp * Ones))
        Data = np.vstack((Data, np.round(bandObject0.energy_scale,0) * Ones))
        DataHeader = "theta[deg]\t" + rhozzHeader + "B[T]\tt[meV]\t"

        for (key, value) in sorted(bandObject0._band_params.items()):
            if key!="t":
                Data = np.vstack((Data, np.round(value,3)*Ones))
                DataHeader += key + "\t"
        if bandAF == True:
            Data = np.vstack((Data, np.round(bandObject0._band_params["M"],3)*Ones))
            DataHeader += "M\t"

        Data = np.vstack((Data, bandObject0.res_xy * Ones, bandObject0.res_z * Ones))
        DataHeader += "res_xy\tres_z\t"

        condHeader = ""
        for (band_name, condObject) in self.condObject_dict.items():
            Data = np.vstack((Data,
                              condObject.gamma_0 * Ones,
                              condObject.gamma_dos_max * Ones,
                              condObject.gamma_k * Ones,
                              condObject.power   * Ones))
            condHeader += band_name + "_g_0[THz]\t" + \
                          band_name + "_g_dos_max[THz]\t" + \
                          band_name + "_g_k[THz]\t" + \
                          band_name + "_power\t"
        DataHeader += condHeader

        Data = Data.transpose()

        ## Build header ---------------------------------------------------------
        if folder != "":
            folder += "/"
        if filename == None:
            filename = folder + "Rzz_" + self.file_name_func() + ".dat"
        else:
            filename = folder + filename
        np.savetxt(filename, Data, fmt='%.7e',
        header = DataHeader, comments = "#")


    def figADMR(self, fig_show=True, fig_save=True, folder="", filename=None):
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

        ## ADMR figure ////////////////////////////////////////////////////////#
        fig, axes = plt.subplots(1, 1, figsize = (10.5, 5.8)) # (1,1) means one plot, and figsize is w x h in inch of figure
        fig.subplots_adjust(left = 0.15, right = 0.70, bottom = 0.18, top = 0.92) # adjust the box of axes regarding the figure size

        axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

        # To point to bandstructure parameters, we use just one band
        # as they should share the same parameters
        condObject0 = self.condObject_dict[self.band_names[0]]

        # Labels
        fig.text(0.99, 0.9, r"$T$ = " + "{0:.0f}".format(condObject0.T) + " K", ha="right")
        fig.text(0.99, 0.84, r"$B$ = " + "{0:.0f}".format(condObject0.Bamp) + " T", ha="right")
        fig.text(0.99, 0.78, r"$p$ = " + "{0:.3f}".format(self.total_hole_doping), ha="right")

        ## Colors
        if self.Bphi_array.size > 4:
            cmap = mpl.cm.get_cmap('jet', self.Bphi_array.size)
            colors = cmap(np.arange(self.Bphi_array.size))
        else:
            # colors = ['#1A52A5', '#00D127', '#CE0000'][::-1]
            colors = ['#000000', '#3B528B', '#FF0000', '#C7E500']

        axes2 = axes.twinx()
        axes2.set_axisbelow(True)

        ## Plot ADMR
        for i, B_phi in enumerate(self.Bphi_array):
            line = axes.plot(self.Btheta_array, self.rzz_array[i, :], label = r"$\phi$ = " + r"{0:.0f}".format(B_phi))
            plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "o", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)  # set properties

        axes.set_xlim(0, self.Btheta_max)
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
        axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)
        ymin, ymax = axes.axis()[2:]
        axes2.set_ylim(ymin * self.rhozz_array[0, 0] * 1e5, ymax * self.rhozz_array[0, 0] * 1e5)
        axes2.set_ylabel(r"$\rho_{\rm zz}$ ( m$\Omega$ cm )", labelpad = 40, rotation = 270)


        axes.legend(loc = 0, fontsize = 14, frameon = False, numpoints = 1, markerscale = 1, handletextpad = 0.5)

        ## Set ticks space and minor ticks
        xtics = 30. # space between two ticks
        mxtics = xtics / 2.  # space between two minor ticks
        majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks
        # axes.xaxis.set_major_locator(MultipleLocator(xtics))
        # axes.xaxis.set_major_formatter(majorFormatter)
        # axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
        axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axes.locator_params(axis='y', nbins=6)

        fig_list.append(fig)

        ## Show figure ////////////////////////////////////////////////////////#
        if fig_show == True:
            plt.show()

        ## Parameters figures ///////////////////////////////////////////////////#
        for condObject in self.condObject_dict.values():
            fig_list.append(condObject.figParameters(fig_show=fig_show))

        ## Save figure ////////////////////////////////////////////////////////#
        if fig_save == True:
            if folder != "":
                folder += "/"
            if filename == None:
                filename = folder + "Rzz_" + self.file_name_func() + ".pdf"
            else:
                filename = folder + filename
            file_figures = PdfPages(filename)
            for fig in fig_list:
                file_figures.savefig(fig)
            file_figures.close()


