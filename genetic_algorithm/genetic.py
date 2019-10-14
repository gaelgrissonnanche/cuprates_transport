import numpy as np
import random
from copy import deepcopy
import genetic_utils as utils
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

init_member = {
    "bandname": "LargePocket",
    "a": 3.74767,
    "b": 3.74767,
    "c": 13.2,
    "t": 190,
    "tp": -0.14,
    "tpp": 0.07,
    "tz": 0.07,
    "tz2": 0.00,
    "mu": -0.826,
    "fixdoping": -2,
    "numberOfKz": 7,
    "mesh_ds": 0.15707963267948966,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 15, 30, 45],
    "gamma_0": 15,
    "gamma_k": 66,
    "gamma_dos_max": 0,
    "power": 12,
    "seed": 72,
    "data_T": 25,
    "data_p": 0.24,
}


ranges_dict = {
    "t": [150.0,250.0],
    "tp": [-0.16,-0.12],
    "tpp": [0.05,0.09],
    "tz": [0.05,0.09],
    "mu": [-0.85,-0.8],
    "gamma_0": [5,25],
    "gamma_k": [20,100],
    # "gamma_dos_max": [10.0,300.0],
    # "power":[1, 20],
}

## Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

data_dict = {}  # keys (T, phi), content [filename, theta, rzz, theta_cut]
data_dict[25, 0] = ["../data_NdLSCO_0p25/0p25_0degr_45T_25K.dat", 0, 1, 90]
data_dict[25, 15] = ["../data_NdLSCO_0p25/0p25_15degr_45T_25K.dat", 0, 1, 90]
data_dict[25, 30] = ["../data_NdLSCO_0p25/0p25_30degr_45T_25K.dat", 0, 1, 90]
data_dict[25, 45] = ["../data_NdLSCO_0p25/0p25_45degr_45T_25K.dat", 0, 1, 90]

data_dict[20, 0] = ["../data_NdLSCO_0p25/0p25_0degr_45T_20K.dat", 0, 1, 90]
data_dict[20, 15] = ["../data_NdLSCO_0p25/0p25_15degr_45T_20K.dat", 0, 1, 90]
data_dict[20, 30] = ["../data_NdLSCO_0p25/0p25_30degr_45T_20K.dat", 0, 1, 90]
data_dict[20, 45] = ["../data_NdLSCO_0p25/0p25_45degr_45T_20K.dat", 0, 1, 90]

data_dict[12, 0] = ["../data_NdLSCO_0p25/0p25_0degr_45T_12K.dat", 0, 1, 83.5]
data_dict[12, 15] = ["../data_NdLSCO_0p25/0p25_15degr_45T_12K.dat", 0, 1, 83.5]
data_dict[12, 45] = ["../data_NdLSCO_0p25/0p25_45degr_45T_12K.dat", 0, 1, 83.5]

data_dict[6, 0] = ["../data_NdLSCO_0p25/0p25_0degr_45T_6K.dat", 0, 1, 73.5]
data_dict[6, 15] = ["../data_NdLSCO_0p25/0p25_15degr_45T_6K.dat", 0, 1, 73.5]
data_dict[6, 45] = ["../data_NdLSCO_0p25/0p25_45degr_45T_6K.dat", 0, 1, 73.5]




def genetic_search(init_member, ranges_dict, data_dict,
        population_size=6, N_generation=20, mutation_s=0.1, crossing_p=0.9):
    """init_params_dic is the inital set where we want the algorthim to start from"""

    ## Randomize the radom generator at first
    np.random.seed(init_member['seed'])


    ## INITIAL SET -------------------------------------------------------------

    ## Compute the ADMR for the initial set of parameters
    print('\nINITIAL PARAMETER SET')
    init_member, admr = utils.compute_chi2(init_member, data_dict)
    print('this chi2 = ' + "{0:.3e}".format(init_member["chi2"]))

    ## Initialize the BEST member
    best_member = deepcopy(init_member)
    print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))
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
        this_member, admr = utils.compute_chi2(this_member, data_dict)
        print('this chi2 = ' + "{0:.3e}".format(this_member["chi2"]))

        ## Just display if this member has a better ChiSquare than the Best Member so far
        ## Also erase BestMember if this new Member is better
        generation_0_list.append(this_member)
        if this_member["chi2"] < best_member["chi2"]:
            best_member = deepcopy(this_member)
            n_improvements += 1
            print(str(n_improvements)+' IMPROVEMENTS SO FAR!')
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
            child, admr = utils.compute_chi2(child, data_dict)
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
            print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))

        generations_list.append(deepcopy(last_generation))
        last_generation = next_gen


    ## The End of time ---------------------------------------------------------

    print('THE END OF TIME')

    ## Save BEST member to JSON
    utils.save_member_to_json(best_member, folder="../data_NdLSCO_0p25")

    ## Print and Compute the BEST member
    print('BEST CHI2 = ' + "{0:.3e}".format(best_member["chi2"]))
    utils.fig_compare(best_member, data_dict, folder="../data_NdLSCO_0p25")





## Play
genetic_search(init_member,ranges_dict, data_dict, population_size=100, N_generation=20, mutation_s=0.3, crossing_p=0.5)


# utils.save_member_to_json(init_member, folder="../data_NdLSCO_0p25")
# utils.fig_compare(init_member, data_dict, folder="../data_NdLSCO_0p25")

# class ObjectView():
#     def __init__(self,dictionary):
#         self.__dict__.update(dictionary)