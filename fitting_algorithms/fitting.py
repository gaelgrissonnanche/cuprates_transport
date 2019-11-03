import numpy as np
import random
from copy import deepcopy
import fitting_utils as utils
from lmfit import minimize, Parameters, fit_report
import os
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



## Genetic algorithm ///////////////////////////////////////////////////////////

def genetic_search(init_member, ranges_dict, data_dict, folder="",
        population_size=100, N_generation=20, mutation_s=0.1, crossing_p=0.9):
    """init_params_dic is the inital set where we want the algorthim to start from"""

    ## Randomize the radom generator at first
    np.random.seed(init_member['seed'])


    ## INITIAL SET -------------------------------------------------------------

    ## Compute the ADMR for the initial set of parameters
    print('\nINITIAL PARAMETER SET')
    init_member = utils.compute_chi2(init_member, data_dict)[0]
    print('Initial chi2 = ' + "{0:.3e}".format(init_member["chi2"]))

    ## Initialize the BEST member
    best_member = deepcopy(init_member)
    best_member_path = utils.save_member_to_json(best_member, folder=folder)
    n_improvements = 0


    ## GENERATION 0 ------------------------------------------------------------

    ## Create GENERATION_0 from the initial set of parameters
    ## and mutating the ones that should variate in the proposed search range
    print('\nINITIAL POPULATION')
    generation_0_list = []

    ## Loop over the MEMBERs of GENERATION_0
    for ii in range(population_size):
        print('\nGENERATION 0 MEMBER '+str(ii+1))
        this_member = deepcopy(init_member)

        ## Loop over the genes of this MEMBER picked within the range
        for gene_name, gene_range in ranges_dict.items():
            this_member[gene_name] = random.uniform(gene_range[0], gene_range[-1])

        ## Compute the fitness of this MEMBER
        this_member = utils.compute_chi2(this_member, data_dict)[0]
        print('this chi2 = ' + "{0:.3e}".format(this_member["chi2"]))

        ## Just display if this member has a better ChiSquare than the Best Member so far
        ## Also erase BestMember if this new Member is better
        generation_0_list.append(this_member)
        if this_member["chi2"] < best_member["chi2"]:
            best_member = deepcopy(this_member)
            n_improvements += 1
            print(str(n_improvements)+' IMPROVEMENTS SO FAR!')
            ## Save BEST member to JSON
            os.remove(best_member_path)
            best_member_path = utils.save_member_to_json(best_member, folder=folder)
        print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))


    ## NEXT GENERATIONS --------------------------------------------------------

    generations_list = []
    last_generation = generation_0_list # is the list of all the MEMBERs previous GENERATION

    ## Loop over ALL next GENERATIONs
    for n in range(N_generation):
        print('\n\nMUTATION CYCLE '+str(n+1))
        next_gen = []

        ## Loop over the MEMBERs of this GENERATION
        for ii in range(population_size):
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
            child = utils.compute_chi2(child, data_dict)[0]
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
                os.remove(best_member_path)
                best_member_path = utils.save_member_to_json(best_member, folder=folder)
            print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))

        generations_list.append(deepcopy(last_generation))
        last_generation = next_gen


    ## The End of time ---------------------------------------------------------

    print('THE END OF TIME')

    ## Save BEST member to JSON
    os.remove(best_member_path)
    utils.save_member_to_json(best_member, folder=folder)

    ## Print and Compute the BEST member
    print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))
    utils.fig_compare(best_member, data_dict, folder=folder)





## Fit search //////////////////////////////////////////////////////////////////

def fit_search(init_member, ranges_dict, data_dict, folder=""):

    ## Initialize
    pars = Parameters()
    final_member = deepcopy(init_member)

    for param_name, param_range in ranges_dict.items():
        pars.add(param_name, value = init_member[param_name], min = param_range[0], max = param_range[-1])

    ## Run fit algorithm
    out = minimize(utils.compute_diff, pars, args=(init_member, ranges_dict, data_dict))

    ## Display fit report
    print(fit_report(out.params))

    ## Export final parameters from the fit
    for param_name in ranges_dict.keys():
            final_member[param_name] = out.params[param_name].value

    ## Save BEST member to JSON
    utils.save_member_to_json(final_member, folder=folder)

    ## Compute the FINAL member
    utils.fig_compare(final_member, data_dict, folder=folder)









# class ObjectView():
#     def __init__(self,dictionary):
#         self.__dict__.update(dictionary)