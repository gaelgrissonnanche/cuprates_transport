import json
import numpy as np
import time
from copy import deepcopy
from lmfit import minimize, Parameters, report_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

from gael_modified.bandstructure import BandStructure, PiPiBandStructure, setMuToDoping, doping
from gael_modified.admr import ADMR
from gael_modified.conductivity import Conductivity
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
        self.ranges_dict = deepcopy(ranges_dict)
        self.data_dict   = deepcopy(data_dict)
        self.data_T_list = init_member["data_T"]
        self.folder      = folder
        self.normalized_data = normalized_data
        self.pipi_FSR    = pipi_FSR
        self.weight_rhozz = 0
        self.pars        = Parameters()
        for param_name, param_range in self.ranges_dict.items():
            if param_name in self.init_member.keys() and type(self.init_member[param_name])==dict:
                for T in self.data_T_list:
                    self.pars.add(param_name + "_" + str(T), value = self.init_member[param_name][T], min = param_range[T][0], max = param_range[T][-1])
            elif param_name in self.init_member.keys() and type(self.init_member[param_name])!=dict:
                self.pars.add(param_name, value = self.init_member[param_name], min = param_range[0], max = param_range[-1])
            elif param_name in self.init_member["band_params"].keys():
                self.pars.add(param_name, value = self.init_member["band_params"][param_name], min = param_range[0], max = param_range[-1])
        self.method      = method # "shgo", "differential_evolution", "leastsq"

        ## Create a dictionnary member that contains dictionnaries for each temperatures
        self.member_dict = {}
        for T in self.data_T_list: # first create that list
            self.member_dict[T] = deepcopy(init_member)
            self.member_dict[T]["data_T"] = T
        for param_name, param_value in init_member.items(): # then break the dictionaries of temperatures in member
            if type(param_value)==dict and param_name!="band_params":
                for T in self.data_T_list:
                    self.member_dict[T][param_name] = init_member[param_name][T]

        ## Objects
        if pipi_FSR==False:
            self.bandObject = BandStructure(**self.member_dict[self.data_T_list[0]])
        else:
            self.bandObject = PiPiBandStructure(**self.member_dict[self.data_T_list[0]])

        self.condObject_dict = {}
        self.admrObject_dict = {}

        ## Empty spaces
        self.nb_calls     = 0
        self.json_name   = None
        self.rhozz_data_i_dict = {} # keys=(T), dictionaries of interpolated values for fit
        self.rzz_data_i_dict   = {}  # keys=(T) dictionaries of interpolated values for fit
        self.rhozz_0_data_dict = {} # keys=(T, phi), dictionaries of rhozz(theta=0) data
        self.rhozz_data_dict   = {} # keys=(T, phi) dictionaries of the raw data
        self.rzz_data_dict     = {} # keys=(T, phi) dictionaries of the raw data
        self.Btheta_data_dict  = {} # keys=(T, phi) dictionaries of the raw data
        self.Bphi_dict   = {}
        self.Btheta_dict = {}
        self.rhozz_data_i_dict  = {}
        self.rzz_data_i_dict    = {}


    def produce_ADMR_object(self):
        ## Update bandObject
        for param_name in self.ranges_dict.keys():
                if hasattr(self.bandObject, param_name):
                    setattr(self.bandObject, param_name, self.member_dict[self.data_T_list[0]][param_name])
                if param_name in self.bandObject._band_params.keys():
                    self.bandObject[param_name] = self.member_dict[self.data_T_list[0]]["band_params"][param_name]

        ## Adjust the doping if need be
        if self.member_dict[self.data_T_list[0]]["fixdoping"] >=-1 and self.member_dict[self.data_T_list[0]]["fixdoping"] <=1:
            self.bandObject.setMuToDoping(self.member_dict[self.data_T_list[0]]["fixdoping"])
            for T in self.data_T_list:
                self.member_dict[T]["band_params"]["mu"] = self.bandObject["mu"]

        self.bandObject.runBandStructure()

        for T in self.data_T_list:
            self.condObject_dict[T] = Conductivity(self.bandObject, **self.member_dict[T])
            self.admrObject_dict[T] = ADMR([self.condObject_dict[T]], **self.member_dict[T])
            self.admrObject_dict[T].Btheta_array = self.Btheta_dict[T]
            self.admrObject_dict[T].Bphi_array   = self.Bphi_dict[T]


    def load_Bphi_data(self):
        ## Loop over all temperatures
        for T in self.data_T_list:
            ## Create array of phi at the selected temperature
            Bphi_array = []
            for t_phi, phi in self.data_dict.keys():
                if (T == t_phi) * np.isin(phi, np.array(self.member_dict[T]["Bphi_array"])):
                    Bphi_array.append(float(phi)) # put float for JSON
            Bphi_array.sort()
            self.Bphi_dict[T] = np.array(Bphi_array)


    def load_Btheta_data(self):
        ## Loop over all temperatures
        for T in self.data_T_list:
            ## Create Initial Btheta
            Btheta_array = np.arange(self.member_dict[T]["Btheta_min"],
                                     self.member_dict[T]["Btheta_max"] + self.member_dict[T]["Btheta_step"],
                                     self.member_dict[T]["Btheta_step"])
            ## Create Bphi
            self.load_Bphi_data()
            # Cut Btheta_array to theta_cut
            Btheta_cut_array = np.zeros(len(self.Bphi_dict[T]))
            for i, phi in enumerate(self.Bphi_dict[T]):
                Btheta_cut_array[i] = self.data_dict[T, phi][3]
            Btheta_cut_min = np.min(Btheta_cut_array)  # minimum cut for Btheta_array
            # New Btheta_array with cut off if necessary
            self.Btheta_dict[T] = Btheta_array[Btheta_array <= Btheta_cut_min]


    def load_and_interp_data(self):
        """
        data_dict[data_T,phi] = [filename, col_theta, col_rzz, theta_cut, rhozz_0]
        """
        ## Create Btheta array
        self.load_Btheta_data()
        ## Create array of phi at the selected temperature
        self.load_Bphi_data()

        ## Loop over all temperatures
        for T in self.data_T_list:
            ## Interpolate data over theta of simulation
            rzz_data_i_matrix = np.zeros((len(self.Bphi_dict[T]), len(self.Btheta_dict[T])))
            rhozz_data_i_matrix = np.zeros((len(self.Bphi_dict[T]), len(self.Btheta_dict[T])))
            for i, phi in enumerate(self.Bphi_dict[T]):
                filename     = self.data_dict[T, phi][0]
                col_theta    = self.data_dict[T, phi][1]
                col_rzz      = self.data_dict[T, phi][2]
                Btheta_cut   = self.data_dict[T, phi][3]
                rhozz_0      = self.data_dict[T, phi][4] # Ohm.m

                data = np.loadtxt(filename, dtype="float", comments="#")
                theta = data[:, col_theta]
                rzz   = data[:, col_rzz]

                ## Order data
                index_order = np.argsort(theta)
                theta = theta[index_order]
                rzz   = rzz[index_order]

                rzz  /= np.interp(0, theta, rzz)
                rzz_i = np.interp(self.Btheta_dict[T], theta, rzz) # "i" is for interpolated

                rhozz_data_i_matrix[i, :] = rzz_i * rhozz_0
                rzz_data_i_matrix[i, :] = rzz_i
                self.Btheta_data_dict[T, phi] = theta[theta<=Btheta_cut]
                self.rhozz_data_dict[T, phi]  = rzz[theta<=Btheta_cut] * rhozz_0
                self.rzz_data_dict[T, phi]    = rzz[theta<=Btheta_cut]
                self.rhozz_0_data_dict[T, phi]     = rhozz_0 # Ohm m

            self.rhozz_data_i_dict[T] = rhozz_data_i_matrix
            self.rzz_data_i_dict[T]   = rzz_data_i_matrix


    def compute_diff(self, pars):
        """Compute diff = sim - data matrix"""

        self.pars = pars

        start_total_time = time.time()

        ## Load data
        self.load_and_interp_data()

        ## Update Btheta & Bphi function of the data
        for T in self.data_T_list:
            self.member_dict[T]["Bphi_array"]  = list(self.Bphi_dict[T])
            self.member_dict[T]["Btheta_min"]  = float(np.min(self.Btheta_dict[T])) # float need for JSON
            self.member_dict[T]["Btheta_max"]  = float(np.max(self.Btheta_dict[T]))
            self.member_dict[T]["Btheta_step"] = float(self.Btheta_dict[T][1] - self.Btheta_dict[T][0])

        ## Update member with fit parameters
        for param_name in self.ranges_dict.keys():
            for T in self.data_T_list:
                if param_name in self.init_member.keys() and type(self.init_member[param_name])==dict:
                    self.member_dict[T][param_name] = self.pars[param_name + "_" + str(T)].value
                elif param_name in self.init_member.keys() and type(self.init_member[param_name])!=dict:
                    self.member_dict[T][param_name] = self.pars[param_name].value
                elif param_name in self.init_member["band_params"].keys():
                    self.member_dict[T]["band_params"][param_name] = self.pars[param_name].value

        # Print params
        for param_name in self.ranges_dict.keys():
            if param_name in self.init_member.keys() and type(self.init_member[param_name])==dict:
                        for T in self.data_T_list:
                            print(param_name + "_" + str(T) + " : " + "{0:g}".format(self.pars[param_name + "_" + str(T)].value))
            elif param_name in self.init_member.keys() and type(self.init_member[param_name])!=dict:
                        print(param_name + " : " + "{0:g}".format(self.pars[param_name].value))
            elif param_name in self.init_member["band_params"].keys():
                        print(param_name + " : " + "{0:g}".format(self.pars[param_name].value))

        ## Compute ADMR ------------------------------------------------------------
        self.produce_ADMR_object()
        for T in self.data_T_list:
            self.admrObject_dict[T].runADMR()

        self.nb_calls += 1
        print("---- call #" + str(self.nb_calls) + " in %.6s seconds ----" % (time.time() - start_total_time))

        diff = np.array([])
        for T in self.data_T_list:
            diff_matrix = np.zeros_like(self.rzz_data_i_dict[T])
            for i, phi in enumerate(self.Bphi_dict[T]):
                diff_matrix[i, :] = self.rzz_data_i_dict[T][i, :] - self.admrObject_dict[T].rzz_array[i, :]
            diff = np.append(diff, diff_matrix.flatten())

            if self.normalized_data==False:
                for i, phi in enumerate(self.Bphi_dict[T]):
                    rhozz_0_fit  = np.interp(0, self.admrObject_dict[T].Btheta_array, self.admrObject_dict[T].rhozz_array[i, :])
                    rhozz_0_data = self.rhozz_0_data_dict[T, phi]
                    diff_rhozz_0 = self.weight_rhozz * (rhozz_0_data - rhozz_0_fit) * 1e5 # mOhm.cm
                    diff = np.append(diff, diff_rhozz_0)
        return diff


    def runFit(self):
        ## Initialize parameters

        self.nb_calls = 0

        ## Run fit algorithm
        if self.method=="least_square":
            out = minimize(self.compute_diff, self.pars)
        if self.method=="shgo":
            out = minimize(self.compute_diff, self.pars,
                           method='shgo', sampling_method='sobol', options={"f_tol": 1e-16}, n=100, iters=20)
        if self.method=="ampgo":
            out = minimize(self.compute_diff, self.pars,
                           method='ampgo')
        if self.method=="dual_annealing":
            out = minimize(self.compute_diff, self.pars,
                           method='dual_annealing')
        if self.method=="differential_evolution":
            out = minimize(self.compute_diff, self.pars,
                           method='differential_evolution')
        else:
            print("This method does not exist in the class")

        ## Display fit report
        report_fit(out)

        ## Export final parameters from the fit
        for param_name in self.ranges_dict.keys():
            for T in self.data_T_list:
                if param_name in self.init_member.keys() and type(self.init_member[param_name])==dict:
                    self.member_dict[T][param_name] = out.params[param_name + "_" + str(T)].value
                elif param_name in self.init_member.keys() and type(self.init_member[param_name])!=dict:
                    self.member_dict[T][param_name] = out.params[param_name].value
                elif param_name in self.init_member["band_params"].keys():
                    self.member_dict[T]["band_params"][param_name] = out.params[param_name].value

        ## Save BEST member to JSON
        self.save_member_to_json()

        ## Compute the FINAL member
        self.fig_compare(fig_save=True)


    def load_member_from_json(self):
        with open(self.folder + "/" + self.json_name, "r") as f:
            self.member_dict = json.load(f)


    def save_member_to_json(self):
        self.produce_ADMR_object()
        path = self.folder + "/data_" + "p" + "{0:.2f}".format(self.member_dict[self.data_T_list[0]]["data_p"]) + "_T_"
        for T in self.data_T_list:
            path += "{0:.1f}".format(self.member_dict[T]["data_T"]) + "K_"
        path += "fit_.json"
        with open(path, 'w') as f:
            json.dump(self.member_dict, f, indent=4)


    def fig_compare(self, fig_show=True, fig_save=False):

        ## Load data  ------------------------------------------------------------
        self.load_and_interp_data()

        ## Run ADMR
        self.produce_ADMR_object()
        for T in self.data_T_list:
            self.admrObject_dict[T].runADMR()

        ## Plot >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        fig_list = []

        for T in self.data_T_list:
            fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
            fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

            if self.normalized_data==True:
                axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

            #############################################
            fig.text(0.84, 0.89, r"$B$ = " + str(self.member_dict[T]["Bamp"]) + " T", fontsize=14)
            fig.text(0.84,0.84, r"$T$ (data) = " + str(self.member_dict[T]["data_T"]) + " K", fontsize=14)
            fig.text(0.84,0.79, r"$T$ (sim) = " + str(self.member_dict[T]["T"]) + " K", fontsize=14)
            fig.text(0.84,0.74, r"$p$ (data) = " + "{0:.2f}".format(self.member_dict[T]["data_p"]), fontsize=14)
            fig.text(0.84,0.69, r"$p$ (sim) = " + "{0:.3f}".format(self.admrObject_dict[T].totalHoleDoping), fontsize=14)
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
                for i, phi in enumerate(self.Bphi_dict[T]):
                    line = axes.plot(self.Btheta_data_dict[T, phi], self.rzz_data_dict[T, phi], label = r"$\phi$ = " + str(phi))
                    plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

                for i, phi in enumerate(self.Bphi_dict[T]):
                    line = axes.plot(self.Btheta_dict[T], self.admrObject_dict[T].rzz_array[i,:])
                    plt.setp(line, ls ="--", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)
            else:
                axes.set_ylabel(r"$\rho_{\rm zz}$ ( m$\Omega$ cm )", labelpad = 8)
                for i, phi in enumerate(self.Bphi_dict[T]):
                    line = axes.plot(self.Btheta_data_dict[T, phi], self.rhozz_data_dict[T, phi] * 1e5, label = r"$\phi$ = " + str(phi))
                    plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

                for i, phi in enumerate(self.Bphi_dict[T]):
                    line = axes.plot(self.Btheta_dict[T], self.admrObject_dict[T].rhozz_array[i,:] * 1e5)
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
            for iniCondObject in self.admrObject_dict[T].initialCondObjectDict.values():
                fig_list.append(iniCondObject.figParameters(fig_show=fig_show))

        ## Save figures list --------------
        if fig_save == True:
            path = self.folder + "/data_" + "p" + "{0:.2f}".format(self.member_dict[self.data_T_list[0]]["data_p"]) + "_T_"
            for T in self.data_T_list:
                path += "{0:.1f}".format(self.member_dict[T]["data_T"]) + "K_"
            path += "fit_.pdf"
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

