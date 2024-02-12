import json
import numpy as np
import time
import sys
from copy import deepcopy
from psutil import cpu_count
from multiprocessing import Pool, Value
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import differential_evolution

from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class FittingADMRParallel:
    def __init__(self, init_member, bounds_dict, data_dict, pipi_FSR=False,
                 folder=".",
                 normalized_data=True, popsize=15,
                 **trash):
        ## Initialize
        self.init_member = deepcopy(init_member) # contains all the parameters to calculate ADMR
        self.member      = deepcopy(init_member) # contains all the parameters to calculate ADMR
        self.bounds_dict = bounds_dict
        self.data_dict   = data_dict
        self.init_time   = time.time()
        self.popsize     = popsize # the popsize for the differential evolution
        self.folder      = folder
        self.normalized_data = normalized_data
        self.pipi_FSR    = pipi_FSR

        ## Create the list sorted of the free parameters
        self.pars = {} # dictionnary of free parameters to computre residual
        self.free_pars_name  = sorted(self.bounds_dict.keys())
        self.fixed_pars_name = np.setdiff1d(sorted(self.member.keys()),
                                             self.free_pars_name)
        ## Create tuple of bounds for scipy
        self.bounds  = []
        for free_name in self.free_pars_name:
            self.bounds.append((self.bounds_dict[free_name][0],
                                self.bounds_dict[free_name][1]))
        self.bounds = tuple(self.bounds)

        ## Objects
        self.bandObject = BandStructure(**self.member, parallel=False)
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

        ## Load data
        self.load_and_interp_data()
        ## Update Btheta & Bphi as a function of the angles in the data
        self.member["Bphi_array"]  = list(self.Bphi_array)
        self.member["Btheta_min"]  = float(np.min(self.Btheta_array)) # float need for JSON
        self.member["Btheta_max"]  = float(np.max(self.Btheta_array))
        self.member["Btheta_step"] = float(self.Btheta_array[1] - self.Btheta_array[0])


    def generate_admr(self):
        """Takes care of creating the ADMR object with the member values"""
        ## Update bandObject
        for param_name in self.bounds_dict.keys():
                if hasattr(self.bandObject, param_name):
                    setattr(self.bandObject, param_name, self.member[param_name])
                if param_name in self.bandObject._band_params.keys():
                    self.bandObject[param_name] = self.member["band_params"][param_name]
        ## Adjust the doping if required
        if self.member["fixdoping"] >=-1 and self.member["fixdoping"] <=1:
            self.bandObject.set_mu_to_doping(self.member["fixdoping"])
            self.member["band_params"]["mu"] = self.bandObject["mu"]
        ## Calculate the bandObject
        self.bandObject.runBandStructure()
        ## Create Conductivity object
        self.condObject = Conductivity(self.bandObject, **self.member)
        ## Create ADMR object
        self.admrObject = ADMR([self.condObject], **self.member, show_progress=False)
        self.admrObject.Btheta_array = self.Btheta_array
        self.admrObject.Bphi_array = self.Bphi_array


    def load_Bphi_data(self):
        """Create array of phi angles at the selected temperature"""
        Bphi_array = []
        for t, phi in self.data_dict.keys():
            if (self.member["data_T"] == t) * np.isin(phi, np.array(self.member["Bphi_array"])):
                Bphi_array.append(float(phi)) # put float for JSON
        Bphi_array.sort()
        self.Bphi_array = np.array(Bphi_array)


    def load_Btheta_data(self):
        """Creates the theta angles at the selected temperatures and phi"""
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
            ## Load data
            data = np.loadtxt(filename, dtype="float", comments="#")
            theta = data[:, col_theta]
            rzz   = data[:, col_rzz]
            ## Sort data
            index_order = np.argsort(theta)
            theta = theta[index_order]
            rzz   = rzz[index_order]
            ## Normalize data
            rzz  /= np.interp(0, theta, rzz)
            rzz_i = np.interp(self.Btheta_array, theta, rzz) # "i" is for interpolated
            ## Store data
            self.rhozz_data_matrix[i, :] = rzz_i * rhozz_0
            self.rzz_data_matrix[i, :] = rzz_i


    def compute_diff2(self, x, verbose=False):
        """Compute diff = sim - data matrix"""
        ## Creates the dictionnary of variables with updated values
        for i, pars_name in enumerate(self.free_pars_name):
            self.pars[pars_name] = x[i]
        ## Update member with fit parameters
        for pars_name in self.free_pars_name:
            if pars_name in self.member.keys():
                self.member[pars_name] = self.pars[pars_name]
            elif pars_name in self.member["band_params"].keys():
                self.member["band_params"][pars_name] = self.pars[pars_name]

        ## Compute ADMR ------------------------------------------------------------
        self.generate_admr()
        self.admrObject.runADMR()

        ## Compute diff
        diff_matrix = np.zeros_like(self.rzz_data_matrix)
        for i in range(self.Bphi_array.size):
            if self.normalized_data is True:
                diff_matrix[i, :] = self.rzz_data_matrix[i, :] - self.admrObject.rzz_array[i, :]
            else:
                diff_matrix[i, :] = (self.rhozz_data_matrix[i, :] - self.admrObject.rhozz_array[i, :])*1e5
        self.condObject = None
        self.admrObject = None

        ## Increment the global counter to count generations and member numbers
        ## += operation is not atomic, so we need to get a lock:
        if verbose:
            with globals()['shared_num_member'].get_lock():
                globals()['shared_num_member'].value += 1
            num_member = globals()['shared_num_member'].value
            num_gen = np.floor(num_member / (self.popsize*len(self.bounds))) + 1

            print('Gen ' + str(int(num_gen)) + ' ---- ' +
            'Member ' + str(num_member) + ' ---- ' +
            'Time elapsed ' + " %.6s seconds" % (time.time() - self.init_time) + ' ---- ' +
            'Diff: %.8e' % np.sum(diff_matrix.flatten()**2))
            sys.stdout.flush()

        return np.sum(diff_matrix.flatten()**2)

    def compute_diff(self, x):
        return self.compute_diff2(x, verbose=True)

    def load_member_from_json(self):
        with open(self.folder + "/" + self.json_name, "r") as f:
            self.member = json.load(f)


    def save_member_to_json(self, filename=None):
        self.generate_admr()
        if filename==None:
            filename = "data_" + \
            "p" + "{0:.2f}".format(self.member["data_p"]) + "_" + \
            "T" + "{0:.1f}".format(self.member["data_T"]) + "_fit_" + self.admrObject.file_name_func()
        path = self.folder + "/" + filename + ".json"
        with open(path, 'w') as f:
            json.dump(self.member, f, indent=4)


    def fig_compare(self, fig_show=True, fig_save=False, figname=None):
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
        self.generate_admr()
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
        for iniCondObject in self.admrObject.condObject_dict.values():
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


## Functions for fit -------------------------------------------------------------
global shared_num_member
shared_num_member = None

def init(num_member):
    """store the counter for later use """
    # global shared_num_member
    globals()['shared_num_member'] = num_member

def fit_admr_parallel(init_member, bounds_dict, data_dict,
                    normalized_data=True, filename=None,
                    popsize=15, mutation=(0.5, 1), recombination=0.7,
                    percent_workers=100, num_cpu=None):
    ## Create fitting object for parallel calculations
    fit_object = FittingADMRParallel(init_member=init_member,
                bounds_dict=bounds_dict, data_dict=data_dict, popsize=popsize,
                normalized_data=normalized_data)
    if num_cpu is None:
        num_cpu = cpu_count(logical=False)
    num_workers = int(percent_workers / 100 * num_cpu)
    print("# cpu cores: " + str(num_cpu))
    print("# workers: " + str(num_workers))
    ## Initialize counter
    num_member = Value('i', 0)
    ## Create pool of workers


    pool = Pool(processes=num_workers,
                initializer = init, initargs = (num_member, ))

    global iteration
    iteration = 0
    global time_iter
    time_iter = time.time()
    global best_x
    best_x = None

    def callback():
        def fn(xk, convergence):
            globals()['iteration'] += 1
            text = "Iteration: %d\titer time: %.3f\tconvergence: %.3e" % (globals()['iteration'], (time.time() - globals()['time_iter']), convergence)
            globals()['time_iter'] = time.time()
            if (xk != globals()['best_x']).all():
                globals()['best_x'] = xk
                # obj_val = fit_object.compute_diff2(xk, verbose=False)
                sys.stdout.flush()
                # text += "\tNew best:" + str([round(x, 10) for x in xk]) + "\tchi^2: %.3e" % obj_val
                text += "\tNew best:" + str([round(x, 10) for x in xk])
            print(text)
        return fn

    c = callback()

    ## Differential evolution
    res = differential_evolution(fit_object.compute_diff2, fit_object.bounds,
                                 updating='deferred', workers=pool.map,
                                 popsize=popsize, mutation=mutation,
                                 recombination=recombination, polish=False,
                                 callback=c)

    pool.terminate()
    ## Export final parameters from the fit
    for i, pars_name in enumerate(fit_object.free_pars_name):
        if pars_name in fit_object.member.keys():
            fit_object.member[pars_name] = res.x[i]
        elif pars_name in fit_object.member["band_params"].keys():
            fit_object.member["band_params"][pars_name] = res.x[i]
        print(pars_name + " : " + "{0:g}".format(res.x[i]))
    ## Save BEST member to JSON
    fit_object.save_member_to_json(filename=filename)
    ## Compute the FINAL member
    fit_object.fig_compare(fig_save=True, figname=filename)
    return fit_object.member



## ///////////////////////////////////////////////////////////////////////////////


# if __name__ == '__main__':
#     ## ONE BAND Matt et al. ///////////////////////////////////////////////////////
#     init_member = {
#     "bandname": "LargePocket",
#     "a": 3.75,
#     "b": 3.75,
#     "c": 13.2,
#     "energy_scale": 160,
#     "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06146443},
#     "res_xy": 40,
#     "res_z": 11,
#     "fixdoping": 2,
#     "T" : 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 90,
#     "Btheta_step": 5,
#     "Bphi_array": [0, 15, 30, 45],
#     "gamma_0": 15,
#     "gamma_k": 65.756,
#     "power": 12.21,
#     "data_T": 25,
#     "data_p": 0.24,
#     # "epsilon_z": "- 2*tz*(cos(a*kx) - cos(b*ky))**2*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)" +\
#     #              "- 2*tzp*cos(c*kz)",
#     }
#
#     ## For FIT
#     bounds_dict = {
#         "gamma_0": [7,15],
#         "gamma_k": [0,100],
#         "power": [1, 20],
#     #    "tz": [0.03, 0.09],
#     #    "tzp": [-0.03, 0.03],
#     }
#
#     ## Data Nd-LSCO 0.24  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
#     data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, rhozz_0] # rhozz_0 in SI units
#     data_dict[25, 0] = ["../examples/data/NdLSCO_0p24/0p25_0degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
#     data_dict[25, 15] = ["../examples/data/NdLSCO_0p24/0p25_15degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
#     data_dict[25, 30] = ["../examples/data/NdLSCO_0p24/0p25_30degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
#     data_dict[25, 45] = ["../examples/data/NdLSCO_0p24/0p25_45degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
#
#     data_dict[20, 0] = ["../examples/data/NdLSCO_0p24/0p25_0degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
#     data_dict[20, 15] = ["../examples/data/NdLSCO_0p24/0p25_15degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
#     data_dict[20, 30] = ["../examples/data/NdLSCO_0p24/0p25_30degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
#     data_dict[20, 45] = ["../examples/data/NdLSCO_0p24/0p25_45degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
#
#     data_dict[12, 0] = ["../examples/data/NdLSCO_0p24/0p25_0degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
#     data_dict[12, 15] = ["../examples/data/NdLSCO_0p24/0p25_15degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
#     data_dict[12, 45] = ["../examples/data/NdLSCO_0p24/0p25_45degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
#
#     data_dict[6, 0] = ["../examples/data/NdLSCO_0p24/0p25_0degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
#     data_dict[6, 15] = ["../examples/data/NdLSCO_0p24/0p25_15degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
#     data_dict[6, 45] = ["../examples/data/NdLSCO_0p24/0p25_45degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
#
#     t0 = time.time()
#     fit_admr_parallel(init_member, bounds_dict, data_dict, normalized_data=False, popsize=2, percent_workers=2.5)
#     print("## Total time: ", time.time()-t0, "s")
#
#
