import json
import numpy as np
import time
from copy import deepcopy
from lmfit import minimize, Parameters, report_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class FittingADMR:
    def __init__(self, init_member, ranges_dict, data_dict, pipi_FSR=False,
                 folder="",
                 method="differential_evolution",
                 population=100, N_generation=20, mutation_s=0.1, crossing_p=0.9,
                 normalized_data=True,
                 **trash):
        ## Initialize
        self.init_member = deepcopy(init_member)
        self.member      = deepcopy(init_member)
        self.ranges_dict = ranges_dict
        self.data_dict   = data_dict
        self.folder      = folder
        self.normalized_data = normalized_data
        self.pipi_FSR    = pipi_FSR
        self.pars        = Parameters()
        for param_name, param_range in self.ranges_dict.items():
            if param_name in self.init_member.keys():
                self.pars.add(param_name, value = self.init_member[param_name], min = param_range[0], max = param_range[-1])
            elif param_name in self.init_member["band_params"].keys():
                self.pars.add(param_name, value = self.init_member["band_params"][param_name], min = param_range[0], max = param_range[-1])
        self.method      = method # "shgo", "differential_evolution", "leastsq"
        ## Differential evolution
        self.population  = population
        self.N_generation= N_generation
        self.mutation_s  = mutation_s
        self.crossing_p  = crossing_p

        ## Objects
        self.bandObject = BandStructure(**self.member)
        self.bandObject.march_square = True

        self.condObject = None
        self.admrObject = None

        ## Empty spaces
        self.nb_calls     = 0
        self.json_name   = None
        self.rhozz_data_matrix = None
        self.rzz_data_matrix   = None
        self.Bphi_array   = None
        self.Btheta_array = None
        self.Btheta_data_dict = {}
        self.rhozz_data_dict  = {}
        self.rzz_data_dict    = {}


    def produce_ADMR_object(self):
        ## Update bandObject
        for param_name in self.ranges_dict.keys():
                if hasattr(self.bandObject, param_name):
                    setattr(self.bandObject, param_name, self.member[param_name])
                if param_name in self.bandObject._band_params.keys():
                    self.bandObject[param_name] = self.member["band_params"][param_name]

        ## Adjust the doping if need be
        if self.member["fixdoping"] >=-1 and self.member["fixdoping"] <=1:
            self.bandObject.set_mu_to_doping(self.member["fixdoping"])
            self.member["band_params"]["mu"] = self.bandObject["mu"]

        self.bandObject.runBandStructure()
        self.condObject = Conductivity(self.bandObject, **self.member)
        self.admrObject = ADMR([self.condObject], **self.member)
        self.admrObject.Btheta_array = self.Btheta_array
        self.admrObject.Bphi_array = self.Bphi_array


    def load_Bphi_data(self):
        ## Create array of phi at the selected temperature
        Bphi_array = []
        for t, phi in self.data_dict.keys():
            if (self.member["data_T"] == t) * np.isin(phi, np.array(self.member["Bphi_array"])):
                Bphi_array.append(float(phi)) # put float for JSON
        Bphi_array.sort()
        self.Bphi_array = np.array(Bphi_array)


    def load_Btheta_data(self):
        ## Create Initial Btheta
        Btheta_array = np.arange(self.member["Btheta_min"],
                                 self.member["Btheta_max"] + self.member["Btheta_step"],
                                 self.member["Btheta_step"])
        ## Create Bphi
        self.load_Bphi_data()

        # Cut Btheta_array to theta_cut
        Btheta_cut_array = np.zeros(len(self.Bphi_array))
        for i, phi in enumerate(self.Bphi_array):
            Btheta_cut_array[i] = self.data_dict[self.member["data_T"], phi][3]
        Btheta_cut_min = np.min(Btheta_cut_array)  # minimum cut for Btheta_array

        # New Btheta_array with cut off if necessary
        self.Btheta_array = Btheta_array[Btheta_array <= Btheta_cut_min]


    def load_and_interp_data(self):
        """
        data_dict[data_T,phi] = [filename, col_theta, col_rzz, theta_cut, rhozz_0]
        """
        ## Create Btheta array
        self.load_Btheta_data()
        ## Create array of phi at the selected temperature
        self.load_Bphi_data()

        ## Interpolate data over theta of simulation
        self.rzz_data_matrix = np.zeros((len(self.Bphi_array), len(self.Btheta_array)))
        self.rhozz_data_matrix = np.zeros((len(self.Bphi_array), len(self.Btheta_array)))
        for i, phi in enumerate(self.Bphi_array):
            filename     = self.data_dict[self.member["data_T"], phi][0]
            col_theta    = self.data_dict[self.member["data_T"], phi][1]
            col_rzz      = self.data_dict[self.member["data_T"], phi][2]
            rhozz_0      = self.data_dict[self.member["data_T"], phi][4]

            data = np.loadtxt(filename, dtype="float", comments="#")
            theta = data[:, col_theta]
            rzz   = data[:, col_rzz]

            ## Order data
            index_order = np.argsort(theta)
            theta = theta[index_order]
            rzz   = rzz[index_order]

            rzz  /= np.interp(0, theta, rzz)
            rzz_i = np.interp(self.Btheta_array, theta, rzz) # "i" is for interpolated

            self.rhozz_data_matrix[i, :] = rzz_i * rhozz_0
            self.rzz_data_matrix[i, :] = rzz_i


    def compute_diff(self, pars):
        """Compute diff = sim - data matrix"""

        self.pars = pars

        start_total_time = time.time()

        ## Load data
        self.load_and_interp_data()

        ## Update Btheta & Bphi function of the data
        self.member["Bphi_array"]  = list(self.Bphi_array)
        self.member["Btheta_min"]  = float(np.min(self.Btheta_array)) # float need for JSON
        self.member["Btheta_max"]  = float(np.max(self.Btheta_array))
        self.member["Btheta_step"] = float(self.Btheta_array[1] - self.Btheta_array[0])

        ## Update member with fit parameters
        for param_name in self.ranges_dict.keys():
            if param_name in self.init_member.keys():
                self.member[param_name] = self.pars[param_name].value
            elif param_name in self.init_member["band_params"].keys():
                self.member["band_params"][param_name] = self.pars[param_name].value
            print(param_name + " : " + "{0:g}".format(self.pars[param_name].value))

        ## Compute ADMR ------------------------------------------------------------
        self.produce_ADMR_object()
        self.admrObject.runADMR()

        self.nb_calls += 1
        print("---- call #" + str(self.nb_calls) + " in %.6s seconds ----" % (time.time() - start_total_time))

        ## Compute diff
        diff_matrix = np.zeros_like(self.rzz_data_matrix)
        for i in range(self.Bphi_array.size):
            if self.normalized_data==True:
                diff_matrix[i, :] = self.rzz_data_matrix[i, :] - self.admrObject.rzz_array[i, :]
            else:
                diff_matrix[i, :] = (self.rhozz_data_matrix[i, :] - self.admrObject.rhozz_array[i, :])*1e5

        return diff_matrix.flatten()


    def runFit(self, filename=None):
        ## Initialize parameters

        self.nb_calls = 0

        ## Run fit algorithm
        if self.method=="least_square":
            out = minimize(self.compute_diff, self.pars)
        if self.method=="shgo":
            out = minimize(self.compute_diff, self.pars,
                           method='shgo', sampling_method='sobol', options={"f_tol": 1e-16}, n = 100, iters=20)
        if self.method=="differential_evolution":
            out = minimize(self.compute_diff, self.pars,
                           method='differential_evolution')
        if self.method=="ampgo":
            out = minimize(self.compute_diff, self.pars,
                           method='ampgo')
        else:
            print("This method does not exist in the class")

        ## Display fit report
        report_fit(out)

        ## Export final parameters from the fit
        for param_name in self.ranges_dict.keys():
            if param_name in self.init_member.keys():
                self.member[param_name] = out.params[param_name].value
            elif param_name in self.init_member["band_params"].keys():
                self.member["band_params"][param_name] = out.params[param_name].value

        ## Save BEST member to JSON
        self.save_member_to_json(filename=filename)

        ## Compute the FINAL member
        self.fig_compare(fig_save=True, figname=filename)


    def load_member_from_json(self):
        with open(self.folder + "/" + self.json_name, "r") as f:
            self.member = json.load(f)


    def save_member_to_json(self, filename=None):
        self.produce_ADMR_object()
        if filename==None:
            filename = "data_" + \
            "p" + "{0:.2f}".format(self.member["data_p"]) + "_" + \
            "T" + "{0:.1f}".format(self.member["data_T"]) + "_fit_" + self.admrObject.fileNameFunc()
        path = self.folder + "/" + filename + ".json"
        with open(path, 'w') as f:
            json.dump(self.member, f, indent=4)


    def fig_compare(self, fig_show=True, fig_save=False, figname=None):

        ## Create Btheta array
        self.load_Btheta_data()
        ## Create array of phi at the selected temperature
        self.load_Bphi_data()

        ## Load non-interpolated data ------------------------------------------
        Btheta_cut = np.max(self.Btheta_array)
        for i, phi in enumerate(self.Bphi_array):
            filename     = self.data_dict[self.member["data_T"], phi][0]
            col_theta    = self.data_dict[self.member["data_T"], phi][1]
            col_rzz      = self.data_dict[self.member["data_T"], phi][2]
            rhozz_0      = self.data_dict[self.member["data_T"], phi][4]

            data  = np.loadtxt(filename, dtype="float", comments="#")
            theta = data[:, col_theta]
            rzz   = data[:, col_rzz]

            ## Order data
            index_order = np.argsort(theta)
            theta = theta[index_order]
            rzz   = rzz[index_order]

            rzz   = rzz / np.interp(0, theta, rzz)
            rhozz = rzz * rhozz_0
            self.Btheta_data_dict[phi] = theta[theta<=Btheta_cut]
            self.rhozz_data_dict[phi]  = rhozz[theta<=Btheta_cut]
            self.rzz_data_dict[phi]    = rzz[theta<=Btheta_cut]

        ## Run ADMR
        self.produce_ADMR_object()
        self.admrObject.runADMR()

        ## Plot >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        fig_list = []

        ## Plot Parameters
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
        fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

        if self.normalized_data==True:
            axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

        #############################################
        fig.text(0.84, 0.89, r"$B$ = " + str(self.member["Bamp"]) + " T", fontsize=14)
        fig.text(0.84,0.84, r"$T$ (data) = " + str(self.member["data_T"]) + " K", fontsize=14)
        fig.text(0.84,0.79, r"$T$ (sim) = " + str(self.member["T"]) + " K", fontsize=14)
        fig.text(0.84,0.74, r"$p$ (data) = " + "{0:.2f}".format(self.member["data_p"]), fontsize=14)
        fig.text(0.84,0.69, r"$p$ (sim) = " + "{0:.3f}".format(self.admrObject.total_hole_doping), fontsize=14)
        #############################################

        #############################################
        axes.set_xlim(0, self.admrObject.Btheta_max)
        # axes.set_ylim(1+1.2*(min_y-1),1.2*(max_y-1)+1)
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
        axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)
        #############################################


        colors = ['#000000', '#3B528B', '#FF0000', '#C7E500', '#ff0080', '#dfdf00']

        if self.normalized_data==True:
            axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)
            for i, phi in enumerate(self.Bphi_array):
                line = axes.plot(self.Btheta_data_dict[phi], self.rzz_data_dict[phi], label = r"$\phi$ = " + str(phi))
                plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

            for i, phi in enumerate(self.Bphi_array):
                line = axes.plot(self.Btheta_array, self.admrObject.rzz_array[i,:])
                plt.setp(line, ls ="--", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)
        else:
            axes.set_ylabel(r"$\rho_{\rm zz}$ ( m$\Omega$ cm )", labelpad = 8)
            for i, phi in enumerate(self.Bphi_array):
                line = axes.plot(self.Btheta_data_dict[phi], self.rhozz_data_dict[phi] * 1e5, label = r"$\phi$ = " + str(phi))
                plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

            for i, phi in enumerate(self.Bphi_array):
                line = axes.plot(self.Btheta_array, self.admrObject.rhozz_array[i,:] * 1e5)
                plt.setp(line, ls ="--", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)

        ######################################################
        plt.legend(loc = 0, fontsize = 14, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
        ######################################################

        ##///Set ticks space and minor ticks space ///#
        xtics = 30 # space between two ticks
        mxtics = xtics / 2.  # space between two minor ticks

        majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

        axes.xaxis.set_major_locator(MultipleLocator(xtics))
        axes.xaxis.set_major_formatter(majorFormatter)
        axes.xaxis.set_minor_locator(MultipleLocator(mxtics))

        if self.normalized_data==True:
            axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        else:
            axes.yaxis.set_major_formatter(FormatStrFormatter('%g'))

        fig_list.append(fig)
        #///////////////////////////////////////////////////////////////////////////////

        if fig_show == True:
            plt.show()
        else:
            plt.close(fig)

        ## Figure Parameters //////////////////////////////////////////////////////#
        for iniCondObject in self.admrObject.initialCondObjectDict.values():
            fig_list.append(iniCondObject.figParameters(fig_show=fig_show))

        ## Save figures list --------------
        if fig_save == True:
            if figname==None:
                figname = "data_" + \
                "p" + "{0:.2f}".format(self.member["data_p"]) + "_" + \
                "T" + "{0:.1f}".format(self.member["data_T"]) + "_fit_" + self.admrObject.fileNameFunc()
            path = self.folder + "/" + figname + ".pdf"
            file_figures = PdfPages(path)
            for fig in fig_list:
                file_figures.savefig(fig)
            file_figures.close()








# if __name__ == '__main__':
#     admr = produce_ADMR_object(member)
#     admr.runADMR()
#     chi2 = compute_chi2(admr, data_dict, init_member["data_T"])
#     print('deviation from experiment :',chi2)
#     compare_plot(admr, member["experiment_p"], member["experiment_T"])

