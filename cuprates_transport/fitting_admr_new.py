import os
import sys
import json
import argparse
import numpy as np
import random
from copy import deepcopy
from lmfit import minimize, Parameters, fit_report
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class FittingADMR:
    def __init__(self, init_member, ranges_dict, data_dict,
                 folder="",
                 population=100, N_generation=20, mutation_s=0.1, crossing_p=0.9,
                 normalized_data=True,
                 **trash):
        self.member      = init_member
        self.best_member = None
        self.ranges_dict = ranges_dict
        self.data_dict   = data_dict
        self.folder      = folder
        self.population  = population
        self.N_generation= N_generation
        self.mutation_s  = mutation_s
        self.crossing_p  = crossing_p
        self.normalized_data = normalized_data
        self.json_name   = "member.json"
        self.rhozz_data_matrix = None
        self.rzz_data_matrix   = None
        self.Bphi_array   = None
        self.Btheta_array = None
        self.Btheta_data_dict = {}
        self.rhozz_data_dict  = {}
        self.rzz_data_dict    = {}

    ## Genetic algorithm ///////////////////////////////////////////////////////
    def genetic_search(self):
        """init_params_dic is the inital set where we want the algorthim to start from"""

        ## Randomize the radom generator at first
        np.random.seed(self.member['seed'])

        ## INITIAL SET :::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        ## Compute the ADMR for the initial set of parameters
        print('\nINITIAL PARAMETER SET')
        self.compute_chi2()
        print('Initial chi2 = ' + "{0:.3e}".format(self.member["chi2"]))

        ## Initialize the BEST member
        self.best_member = deepcopy(self.member)
        best_member_path = self.save_member_to_json(best_member, folder=folder)
        utils.fig_compare(best_member, data_dict, folder=folder, fig_show=False, normalized_data=normalized_data)
        n_improvements = 0


        ## GENERATION 0 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        ## Create GENERATION_0 from the initial set of parameters
        ## and mutating the ones that should variate in the proposed search range
        print('\nINITIAL POPULATION')
        generation_0_list = []

        ## Loop over the MEMBERs of GENERATION_0
        for ii in range(population):
            print('\nGENERATION 0 MEMBER '+str(ii+1))
            this_member = deepcopy(init_member)

            ## Loop over the genes of this MEMBER picked within the range
            for gene_name, gene_range in ranges_dict.items():
                this_member[gene_name] = random.uniform(gene_range[0], gene_range[-1])

            ## Compute the fitness of this MEMBER
            this_member = utils.compute_chi2(this_member, data_dict, normalized_data)[0]
            print('this chi2 = ' + "{0:.3e}".format(this_member["chi2"]))

            ## Just display if this member has a better ChiSquare than the Best Member so far
            ## Also erase BestMember if this new Member is better
            generation_0_list.append(this_member)
            if this_member["chi2"] < best_member["chi2"]:
                best_member = deepcopy(this_member)
                n_improvements += 1
                print(str(n_improvements)+' IMPROVEMENTS SO FAR!')
                ## Save BEST member to JSON
                os.remove(best_member_path) # remove previous json
                os.remove(best_member_path[:-4] + "pdf") # remove previous figure
                best_member_path = utils.save_member_to_json(best_member, folder=folder)
                utils.fig_compare(best_member, data_dict, folder=folder, fig_show=False, normalized_data=normalized_data)
            print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))


        ## NEXT GENERATIONS --------------------------------------------------------

        generations_list = []
        last_generation = generation_0_list # is the list of all the MEMBERs previous GENERATION

        ## Loop over ALL next GENERATIONs
        for n in range(N_generation):
            print('\n\nMUTATION CYCLE '+str(n+1))
            next_gen = []

            ## Loop over the MEMBERs of this GENERATION
            for ii in range(population):
                print('\nGENERATION '+str(n+1)+' MEMBER '+str(ii+1))
                parent = last_generation[ii] # take i^th member of the last computed generation
                child = deepcopy(parent)

                ## Loop over the different genes that will either cross or mutate
                for gene_name, gene_range in ranges_dict.items():
                    # crossing
                    if random.uniform(0,1) > crossing_p:
                        # within the probability of crossing "p", keep the gene of the parent
                        # crossing : keep the same gene (gene_name) value for the child as the parent without mutation.
                        child[gene_name] = parent[gene_name]
                    # mutation
                    else:
                        parent_1 = random.choice(last_generation) # choose a random member in the last generation
                        parent_2 = random.choice(last_generation)
                        parent_3 = random.choice(last_generation)
                        new_gene = parent_1[gene_name] + mutation_s*(parent_2[gene_name]-parent_3[gene_name])
                        ## Is the mutated gene within the range of the gene?
                        child[gene_name] = np.clip(new_gene, gene_range[0], gene_range[-1])

                ## Compute the fitness of the CHILD
                child = utils.compute_chi2(child, data_dict, normalized_data)[0]
                print('this chi2 = ' + "{0:.3e}".format(child["chi2"]))

                # If the CHILD has a better fitness than the PARENT, then keep CHILD
                if child['chi2']<last_generation[ii]['chi2']:
                    next_gen.append(child)
                # If the CHILD is not better than the PARENT, then keep the PARENT
                else:
                    next_gen.append(last_generation[ii])

                # Erase the best chi2 member if found better
                if child['chi2'] < best_member['chi2']:
                    best_member = deepcopy(child)
                    n_improvements += 1
                    print(str(n_improvements)+' IMPROVEMENTS SO FAR!')
                    ## Save BEST member to JSON
                    os.remove(best_member_path) # remove previous json
                    os.remove(best_member_path[:-4] + "pdf") # remove previous figure
                    best_member_path = utils.save_member_to_json(best_member, folder=folder)
                    utils.fig_compare(best_member, data_dict, folder=folder, fig_show=False, normalized_data=normalized_data)
                print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))

            generations_list.append(deepcopy(last_generation))
            last_generation = next_gen


        ## The End of time ---------------------------------------------------------

        print('THE END OF TIME')

        ## Save BEST member to JSON
        os.remove(best_member_path)
        utils.save_member_to_json(best_member, folder=folder)

        ## Print and Compute the BEST member
        plt.clf()
        plt.cla()
        plt.close()
        print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))
        utils.fig_compare(best_member, data_dict, folder=folder, normalized_data=normalized_data)


    def load_member_from_json(self):
        with open(self.folder + "/" + self.json_name, "r") as f:
            self.member = json.load(f)

    def save_member_to_json(self, member):
        admr = self.produce_ADMR()
        path = self.folder + "/data_" + \
            "p" + "{0:.2f}".format(member["data_p"]) + "_" + \
            "T" + "{0:.1f}".format(member["data_T"]) + "_fit_" + admr.fileNameFunc() + ".json"
        with open(path, 'w') as f:
            json.dump(member, f, indent=4)
        return path



    def produce_ADMR(self):
        self.bandObject = BandStructure(**self.member)

        if self.member["fixdoping"] >=-1 and self.member["fixdoping"] <=1  :
            self.bandObject.setMuToDoping(self.member["fixdoping"])

        self.bandObject.discretize_FS(PrintEnding=False)
        self.bandObject.dos_k_func()
        self.bandObject.doping(printDoping=False)

        self.condObject = Conductivity(self.bandObject, **self.member)

        self.condObject.solveMovementFunc()
        admr = ADMR([self.condObject], **self.member)

        return admr


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
        data_dict[data_T,phi] = [filename, col_theta, col_rzz, theta_cut]
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
            factor_to_SI = self.data_dict[self.member["data_T"], phi][4]

            data = np.loadtxt(filename, dtype="float", comments="#")
            theta = data[:, col_theta]
            rhozz = data[:, col_rzz] * factor_to_SI
            rhozz_i = np.interp(self.Btheta_array, theta, rhozz) # "i" is for interpolated

            self.rhozz_data_matrix[i, :] = rhozz_i
            self.rzz_data_matrix[i, :] = rhozz_i / rhozz_i[0]


    def compute_chi2(self):
        """Compute chi^2"""

        ## Load data
        self.load_and_interp_data()

        ## Update Btheta & Bphi function of the data
        self.member["Bphi_array"]  = list(self.Bphi_array)
        self.member["Btheta_min"]  = float(np.min(self.Btheta_array)) # float need for JSON
        self.member["Btheta_max"]  = float(np.max(self.Btheta_array))
        self.member["Btheta_step"] = float(self.Btheta_array[1] - self.Btheta_array[0])

        ## Compute ADMR ------------------------------------------------------------
        admr = self.produce_ADMR()
        admr.runADMR()
        self.member["mu"] = admr.initialCondObjectDict[admr.bandNamesList[0]].bandObject.mu
        print(admr.fileNameFunc())

        ## Compute Chi^2
        if self.normalized_data==True:
            chi2 = np.sum((admr.rzz_array.flatten() - self.rzz_data_matrix.flatten())**2)
        else:
            chi2 = np.sum((admr.rhozz_array.flatten() - self.rhozz_data_matrix.flatten())**2)

        self.member['chi2'] = float(chi2)
        return admr


    # def compute_diff(self, pars, member, ranges_dict, data_dict, normalized_data=True):
    #     """Compute diff = sim - data matrix"""

    #     ## Load data
    #     Bphi_array, Btheta_array, rhozz_data_matrix, rzz_data_matrix = load_and_interp_data(member, data_dict)

    #     ## Update Btheta & Bphi function of the data
    #     member["Bphi_array"]  = list(Bphi_array)
    #     member["Btheta_min"]  = float(np.min(Btheta_array)) # float need for JSON
    #     member["Btheta_max"]  = float(np.max(Btheta_array))
    #     member["Btheta_step"] = float(Btheta_array[1] - Btheta_array[0])

    #     ## Update member with fit parameters
    #     for param_name in ranges_dict.keys():
    #             member[param_name] = pars[param_name].value
    #             print(param_name + " : " + "{0:g}".format(pars[param_name].value))

    #     ## Compute ADMR ------------------------------------------------------------
    #     admr = produce_ADMR(member)
    #     admr.runADMR()

    #     ## Compute diff
    #     diff_matrix = np.zeros_like(rzz_data_matrix)
    #     for i in range(Bphi_array.size):
    #         if normalized_data==True:
    #             diff_matrix[i, :] = rzz_data_matrix[i, :] - admr.rzz_array[i, :]
    #         else:
    #             diff_matrix[i, :] = rhozz_data_matrix[i, :] - admr.rhozz_array[i, :]

    #     return diff_matrix


    def fig_compare(self, fig_show=True, fig_save=True):
        ## Run ADMR from member parameters -----------------------------------------
        admr = self.compute_chi2()
        print('this chi2 = ' + "{0:.3e}".format(self.member["chi2"]))

        ## Load data ---------------------------------------------------------------
        Btheta_cut = np.max(self.Btheta_array)
        for i, phi in enumerate(self.Bphi_array):
            filename     = self.data_dict[self.member["data_T"], phi][0]
            col_theta    = self.data_dict[self.member["data_T"], phi][1]
            col_rzz      = self.data_dict[self.member["data_T"], phi][2]
            factor_to_SI = self.data_dict[self.member["data_T"], phi][4]

            data  = np.loadtxt(filename, dtype="float", comments="#")
            theta = data[:, col_theta]
            rhozz = data[:, col_rzz] * factor_to_SI
            rzz   = data[:, col_rzz] / data[0, col_rzz]
            self.Btheta_data_dict[phi] = theta[theta<=Btheta_cut]
            self.rhozz_data_dict[phi]  = rhozz[theta<=Btheta_cut]
            self.rzz_data_dict[phi]    = rzz[theta<=Btheta_cut]

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
        fig.text(0.84,0.69, r"$p$ (sim) = " + "{0:.3f}".format(admr.totalHoleDoping), fontsize=14)
        fig.text(0.84,0.59, r"$\chi^{\rm 2}$ = " + "{0:.3e}".format(self.member["chi2"]), fontsize=14)
        #############################################

        #############################################
        axes.set_xlim(0, 90)
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
                line = axes.plot(admr.Btheta_array, admr.rzz_array[i,:])
                plt.setp(line, ls ="--", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 5, mec = colors[i], mew= 0)
        else:
            axes.set_ylabel(r"$\rho_{\rm zz}$ ( m$\Omega$ cm )", labelpad = 8)
            for i, phi in enumerate(self.Bphi_array):
                line = axes.plot(self.Btheta_data_dict[phi], self.rhozz_data_dict[phi] * 1e5, label = r"$\phi$ = " + str(phi))
                plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)

            for i, phi in enumerate(self.Bphi_array):
                line = axes.plot(admr.Btheta_array, admr.rhozz_array[i,:] * 1e5)
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
        for iniCondObject in admr.initialCondObjectDict.values():
            fig_list.append(iniCondObject.figParameters(fig_show=fig_show))

        ## Save figures list --------------
        if fig_save == True:
            file_figures = PdfPages(self.folder + "/data_" + \
            "p" + "{0:.2f}".format(self.member["data_p"]) + "_" + \
            "T" + "{0:.1f}".format(self.member["data_T"]) + "_fit_" + admr.fileNameFunc() + ".pdf")
            for fig in fig_list:
                file_figures.savefig(fig)
            file_figures.close()











# if __name__ == '__main__':
#     admr = produce_ADMR(member)
#     admr.runADMR()
#     chi2 = compute_chi2(admr, data_dict, init_member["data_T"])
#     print('deviation from experiment :',chi2)
#     compare_plot(admr, member["experiment_p"], member["experiment_T"])



## Fit search //////////////////////////////////////////////////////////////////

# def fit_search(init_member, ranges_dict, data_dict, folder="",
#                normalized_data=True):

#     ## Initialize
#     pars = Parameters()
#     final_member = deepcopy(init_member)

#     for param_name, param_range in ranges_dict.items():
#         pars.add(param_name, value = init_member[param_name], min = param_range[0], max = param_range[-1])

#     ## Run fit algorithm
#     out = minimize(utils.compute_diff, pars, args=(init_member, ranges_dict, data_dict, normalized_data))#, method="differential_evolution")

#     ## Display fit report
#     print(fit_report(out.params))

#     ## Export final parameters from the fit
#     for param_name in ranges_dict.keys():
#             final_member[param_name] = out.params[param_name].value

#     ## Save BEST member to JSON
#     utils.save_member_to_json(final_member, folder=folder)

#     ## Compute the FINAL member
#     utils.fig_compare(final_member, data_dict, folder=folder, normalized_data=normalized_data)









# class ObjectView():
#     def __init__(self,dictionary):
#         self.__dict__.update(dictionary)