import json
import numpy as np
import os
import sys
from copy import deepcopy
from psutil import cpu_count
from multiprocessing import Pool, Value
from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import differential_evolution

from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## -------------------------------------------------------------------------------
class FitAnglesADMR:
    def __init__(self, T, data_dict):
        self.T = T
        self.data_dict = data_dict
        self.Bphi_array = np.array([])
        self.Btheta_array= np.array([])
        self.Btheta_norm = 0 # theta used for rzz = rhozz / rhozz(theta)

    def create_angles(self):
        """Creates the theta angles at the selected temperatures and phi"""
        ## Create Bphi
        self.Bphi_array = np.array([key[1] for key in self.data_dict.keys() if key[0] == self.T])
        ## Create Btheta_array
        Btheta_min_list = []
        Btheta_max_list = []
        Btheta_steps_list = []
        for i, phi in enumerate(self.Bphi_array):
            Btheta_min_list.append(self.data_dict[self.T, phi][3])
            Btheta_max_list.append(self.data_dict[self.T, phi][4])
            Btheta_steps_list.append(self.data_dict[self.T, phi][5])
        ## Btheta_norm is in principle the same for all phis (here with the last phi)
        self.Btheta_norm = self.data_dict[self.T, phi][6]
        Btheta_min = min(Btheta_min_list)
        Btheta_max = max(Btheta_max_list)
        Btheta_steps = min(Btheta_steps_list)
        N = int((Btheta_max-Btheta_min) / Btheta_steps)
        self.Btheta_array = np.linspace(Btheta_min, Btheta_max, N)
        return self.Bphi_array, self.Btheta_array


## -------------------------------------------------------------------------------
class DataADMR:
    def __init__(self, T, data_dict, Bphi_array, Btheta_array):
        self.data_dict = data_dict
        self.T = T
        self.rhozz_data_matrix = None
        self.rzz_data_matrix   = None
        self.Bphi_array   = Bphi_array
        self.Btheta_array = Btheta_array

    def load_data(self, Btheta_norm=0):
        """
        data_dict[data_T, phi] = [filename, col_theta, col_rhozz, theta_min, theta_max, theta_steps]
        """
        ## Interpolate data over theta of simulation
        self.rzz_data_matrix = np.zeros((len(self.Bphi_array), len(self.Btheta_array)))
        self.rhozz_data_matrix = np.zeros((len(self.Bphi_array), len(self.Btheta_array)))
        for i, phi in enumerate(self.Bphi_array):
            filename  = self.data_dict[self.T, phi][0]
            col_theta = self.data_dict[self.T, phi][1]
            col_rhozz = self.data_dict[self.T, phi][2]
            rhozz_norm = self.data_dict[self.T, phi][7]
            ## Load data
            data  = np.loadtxt(filename, dtype="float", comments="#")
            theta = data[:, col_theta]
            rhozz = data[:, col_rhozz]
            ## Sort data
            index_order = np.argsort(theta)
            theta = theta[index_order]
            rhozz = rhozz[index_order]
            ## Normalize data
            rhozz_i = np.interp(self.Btheta_array, theta, rhozz) # "i" is for interpolated
            rzz_i   = rhozz_i / np.interp(Btheta_norm, theta, rhozz) # rhozz / rhozz(theta=theta_norm)
            rhozz_i *= rhozz_norm
            ## Store data
            self.rhozz_data_matrix[i, :] = rhozz_i
            self.rzz_data_matrix[i, :] = rzz_i


## -------------------------------------------------------------------------------
class SimADMR:
    def __init__(self, params_dict, Bphi_array, Btheta_array):
        self._params_dict = deepcopy(params_dict)
        self.bands_list = [key for key in self.params_dict.keys()]
        self.Bphi_array = Bphi_array
        self.Btheta_array = Btheta_array
        self.rhozz_sim_matrix  = None
        self.rzz_sim_matrix    = None

        condObject_list = []
        for band in self.bands_list:
            bandObject = BandStructure(**self.params_dict[band], parallel=False)
            bandObject.march_square = True
            condObject_list.append(Conductivity(bandObject, **self.params_dict[band]))
        self.admrObject = ADMR(condObject_list, show_progress=False)
        self.admrObject.Bphi_array = self.Bphi_array
        self.admrObject.Btheta_array = self.Btheta_array

    # Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    def _get_params_dict(self):
        return self._params_dict

    def _set_params_dict(self, params_dict):
        self._params_dict = deepcopy(params_dict)
        self.update_params()
    params_dict = property(_get_params_dict, _set_params_dict)

    def update_params(self):
        for i, band in enumerate(self.bands_list):
            band_params = self.params_dict[band]["band_params"]
            scattering_params = self.params_dict[band]["scattering_params"]
            self.admrObject.condObject_list[i].bandObject.band_params =  band_params
            self.admrObject.condObject_list[i].scattering_params = scattering_params

    def compute_rhozz(self, Btheta_norm=0):
        """Calculate the simulated rhozz from ADMR object"""
        for i in range(len(self.bands_list)):
            self.admrObject.condObject_list[i].bandObject.runBandStructure()
        self.admrObject.Bphi_array = self.Bphi_array
        self.admrObject.Btheta_array = self.Btheta_array
        self.admrObject.Btheta_norm = Btheta_norm
        self.admrObject.runADMR()
        self.rhozz_sim_matrix = self.admrObject.rhozz_array
        self.rzz_sim_matrix   = self.admrObject.rzz_array



## -------------------------------------------------------------------------------
class Fitness:
    def __init__(self, data_dict, params_dict, bounds_dict, normalized_data=True):
        ## Initialize
        self.params_dict = deepcopy(params_dict) # contains all the parameters to calculate ADMR
        self.bounds_dict = deepcopy(bounds_dict)
        self.data_dict   = deepcopy(data_dict)
        self.T_list = [T for T in self.bounds_dict.keys() if T!="all"]
        self.normalized_data = normalized_data
        ## Create list of free parameters
        self.free_params_labels = []
        for T in self.bounds_dict.keys():
            for band in self.bounds_dict[T].keys():
                for params_name in self.bounds_dict[T][band].keys():
                    self.free_params_labels.append([T, band, params_name])
        ## Create tuple of bounds for differential evolution
        self.bounds  = []
        for label in self.free_params_labels:
            self.bounds.append((self.bounds_dict[label[0]][label[1]][label[2]][0],
                                self.bounds_dict[label[0]][label[1]][label[2]][1]))
        self.bounds = tuple(self.bounds)
        ## Create phi, theta, rzz_data, rhozz_data arrays
        self.phis_dict   = {}
        self.thetas_dict = {}
        self.rzz_data_dict = {}
        self.rhozz_data_dict = {}
        self.rzz_sim_dict = {}
        self.rhozz_sim_dict = {}
        self.sim_obj_dict = {}
        self.angles_obj_dict = {}
        for T in self.T_list:
            ## Angles
            angles_obj = FitAnglesADMR(T, self.data_dict)
            phis, thetas = angles_obj.create_angles()
            self.angles_obj_dict[T] = angles_obj
            ## Data
            data_obj = DataADMR(T, data_dict, phis, thetas)
            data_obj.load_data(angles_obj.Btheta_norm)
            rzz = data_obj.rzz_data_matrix  # normalized rhozz/rhozz(B=constant)
            rhozz = data_obj.rhozz_data_matrix
            self.phis_dict[T] = phis
            self.thetas_dict[T] = thetas
            self.rzz_data_dict[T] = rzz
            self.rhozz_data_dict[T] = rhozz
            # Simulation
            self.sim_obj_dict[T] = SimADMR(self.params_dict[T],
                                           self.phis_dict[T],
                                           self.thetas_dict[T])

    def update_conditions(self, T, band, param, x_i):
        """Update parameters"""
        if param in self.params_dict[T][band].keys():
            self.params_dict[T][band][param] = x_i
        elif param in self.params_dict[T][band]["band_params"].keys():
            self.params_dict[T][band]["band_params"][param] = x_i
        elif param in self.params_dict[T][band]["scattering_params"].keys():
            self.params_dict[T][band]["scattering_params"][param] = x_i
        else:
            print(str(param)+" does not exist in params_dict["+str(T)+"]["+str(band)+"]")

    def update_parameters(self, x):
        for i, label in enumerate(self.free_params_labels):
            T     = label[0]
            band  = label[1]
            param = label[2]
            if T!="all": # for parameters that are different at different T
                self.update_conditions(T, band, param, x[i])
            elif T=="all": # for parameters that are the same at all T
                for T in self.T_list:
                    self.update_conditions(T, band, param, x[i])

    def compute_fitness(self, x=np.array([0])):
        """Compute chi2 = sum((sim[i] - data[i])**2)"""
        ## Update the params_dict with fit values
        if x.all()!=0:
            self.update_parameters(x)
        ## Compute sim, then diff
        diff_array = np.empty(0)
        self.rzz_sim_dict = {}
        self.rhozz_sim_dict = {}
        for T in self.T_list:
            # Compute sim
            self.sim_obj_dict[T].params_dict = self.params_dict[T]
            self.sim_obj_dict[T].compute_rhozz(self.angles_obj_dict[T].Btheta_norm)
            self.rzz_sim_dict[T] = self.sim_obj_dict[T].rzz_sim_matrix
            self.rhozz_sim_dict[T] = self.sim_obj_dict[T].rhozz_sim_matrix
            # Compute diff
            if self.normalized_data is True:
                diff = self.rzz_data_dict[T].flatten() - self.rzz_sim_dict[T].flatten()
            else:
                diff = self.rhozz_data_dict[T].flatten() - self.rhozz_sim_dict[T].flatten()
            diff_array = np.append(diff_array, diff)
        ## Compute chi2
        chi2 = np.sum(diff_array**2)
        return chi2

    def fig_compare(self, fig_show=True, fig_save=False, folder= "", figname="figure"):
        "Plot Data vs Sim"
        ## Plot >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        fig_list = []
        ## Plot function
        def plot(T):
            ## Plot Parameters
            fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
            fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)
            if self.normalized_data==True:
                axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)
            ## Text labels
            # fig.text(0.84, 0.89, r"$B$ = " + str(self.params_dict[T] ["Bamp"]) + " T", fontsize=14)
            fig.text(0.84,0.84, r"$T$ = " + str(T) + " K", fontsize=14)
            # fig.text(0.84,0.69, r"$p$ (sim) = " + "{0:.3f}".format(self.admrObject.totalHoleDoping), fontsize=14)
            # Colors
            colors = ['#000000', '#3B528B', '#FF0000', '#C7E500', '#ff0080', '#dfdf00']
            ## Select if normalized or not
            if self.normalized_data==True:
                axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)
                data = self.rzz_data_dict[T]
                sim = self.rzz_sim_dict[T]
            else:
                data = self.rhozz_data_dict[T]
                sim = self.rhozz_sim_dict[T]
                axes.set_ylabel(r"$\rho_{\rm zz}$ ( m$\Omega$ cm )", labelpad = 8)
            ## Plotting
            for i, phi in enumerate(self.phis_dict[T]):
                line = axes.plot(self.thetas_dict[T], data[i,:], label = r"$\phi$ = " + str(phi))
                plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)
                line = axes.plot(self.thetas_dict[T], sim[i,:])
                plt.setp(line, ls ="--", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)
            ## Labels
            axes.set_xlim(np.min(self.thetas_dict[T]), np.max(self.thetas_dict[T]))
            axes.tick_params(axis='x', which='major', pad=7)
            axes.tick_params(axis='y', which='major', pad=8)
            axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
            ## Legend
            plt.legend(loc = 0, fontsize = 14, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
            ## Set ticks space and minor ticks space
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
            return fig

        ## Generate plots
        for T in self.T_list:
            fig = plot(T)
            fig_list.append(fig)
            for condObject in self.sim_obj_dict[T].admrObject.condObject_list:
                fig_list.append(condObject.figParameters(fig_show=fig_show))
            if fig_show == True:
            ## Show figures
                plt.show()
            else:
                plt.close(fig)
        ## Save figures list --------------
        if fig_save == True:
            path = os.path.join(folder, figname + ".pdf")
            file_figures = PdfPages(path)
            for fig in fig_list:
                file_figures.savefig(fig)
            file_figures.close()

    def save_member_to_json(self, filename="file", folder=""):
        path = os.path.join(folder, filename + ".json")
        with open(path, 'w') as f:
            json.dump(self.params_dict, f, indent=4)




## Functions for fit -------------------------------------------------------------
global shared_num_member
shared_num_member = None

def init(num_member):
    """store the counter for later use """
    # global shared_num_member
    globals()['shared_num_member'] = num_member

def fit_admr_parallel(params_dict, bounds_dict, data_dict,
                    normalized_data=True, filename="fit_results", folder="",
                    popsize=15, mutation=(0.5, 1), recombination=0.7,
                    percent_workers=100, num_cpu=None):
    ## Create fitness object for parallel calculations
    fitness_obj = Fitness(data_dict, params_dict, bounds_dict, normalized_data)
    ## Initialize workers
    if num_cpu is None:
        num_cpu = cpu_count(logical=False)
    num_workers = int(percent_workers / 100 * num_cpu)
    print("# Parallelization | CPU cores: " + str(num_cpu) + " | Workers: " + str(num_workers) )
    ## Initialize counter
    num_member = Value('i', 0)
    ## Create pool of workers
    pool = Pool(processes=num_workers, initializer = init, initargs = (num_member, ))
    ## Variables for Callback function
    global iteration
    iteration = 0
    global time_iter
    time_iter = time()
    global best_x
    best_x = None
    ## Callback function to print the evolution of differential evolution
    def callback(xk, convergence):
        globals()['iteration'] += 1
        text = "gen %d | %.1f s | convergence: %.3f" % (globals()['iteration'], (time() - globals()['time_iter']), convergence) + "/1 |"
        globals()['time_iter'] = time()
        if (xk != globals()['best_x']).all():
            globals()['best_x'] = xk
            # obj_val = fit_object.compute_diff2(xk, verbose=False)
            # text += "\tNew best:" + str([round(x, 10) for x in xk]) + "\tchi^2: %.3e" % obj_val
            text += " New best:" + str([round(x, 2) for x in xk])
        print(text)
        sys.stdout.flush()
    ## Differential evolution
    res = differential_evolution(fitness_obj.compute_fitness, fitness_obj.bounds,
                                updating='deferred', workers=pool.map,
                                popsize=popsize, mutation=mutation,
                                recombination=recombination, polish=False,
                                callback=callback)
    # res = shgo(fitness_obj.compute_fitness, fitness_obj.bounds,
    #                              workers=pool.map, callback=callback)
    pool.terminate()
    ## Export final parameters from the fit
    fitness_obj.update_parameters(res.x)
    fitness_obj.compute_fitness()
    return fitness_obj

## -------------------------------------------------------------------------------

