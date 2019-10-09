import numpy as np
import random
from copy import deepcopy
import genetic_utils as utils
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

init_member = {
    "bandname":"LargePocket",
    "a":3.74767,
    "b":3.74767,
    "c":13.2,
    "t":190,
    "tp":-0.14,
    "tpp":0.07,
    "tz":0.07,
    "tz2":0.00,
    "mu":-0.826,
    "fixdoping":-2,
    "numberOfKz":7,
    "mesh_ds":0.15707963267948966,
    "Bamp":45,
    "gamma_0":0,
    "gamma_k":0,
    "gamma_dos_max":275,
    "power":12,
    "seed":72,
    "experiment_doping":25,
    "experiment_temperature":25
}

ranges_dict = {
    "t": [150.0,250.0],
    "tp": [-0.16,-0.12],
    "tpp": [0.05,0.09],
    "tz": [0.05,0.09],
    "mu": [-0.85,-0.8],
    "gamma_0": [1.0,20.0],
    # "gamma_k": [60.0,70.0],
    "gamma_dos_max": [10.0,300.0],
    # "power":[2,10,12,14]
}



def genetic_search(init_member, ranges_dict,
        population_size=6, N_generation=20, mutation_s=0.1, crossing_p=0.7):
    """init_params_dic is the inital set where we want the algorthim to start from"""

    ## Randomize the radom generator at first
    np.random.seed(init_member['seed'])


    ## INITIAL SET -------------------------------------------------------------

    ## Compute the ADMR for the initial set of parameters
    print('\nINITIAL PARAMETER SET')
    admr = utils.produce_ADMR(init_member)
    admr.runADMR()
    chi2 = utils.compute_chi2(admr, int(init_member["experiment_doping"]), int(init_member["experiment_temperature"]))
    print(utils.name(init_member))
    print('this chi2 = '+str(chi2))
    init_member['chi2'] = chi2

    ## Initialize the "best chi2" member
    best_chi2_so_far = chi2
    best_dict = init_member
    print('BEST CHI2 = '+str(best_chi2_so_far) )
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
        admr = utils.produce_ADMR(this_member)
        admr.runADMR()
        chi2 = utils.compute_chi2(admr, int(this_member["experiment_doping"]), int(this_member["experiment_temperature"]))
        print(utils.name(this_member))
        print('this chi2 = '+str(chi2))
        this_member['chi2'] = chi2

        ## Just display if this member has a better ChiSquare than the Best Member so far
        ## Also erase BestMember if this new Member is better
        generation_0_list.append(this_member)
        if chi2 < best_chi2_so_far:
            best_chi2_so_far = chi2
            best_dict = this_member
            n_improvements += 1
            print(str(n_improvements)+' IMPROVEMENTS SO FAR!')
        print('BEST CHI2 = '+str(best_chi2_so_far) )


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
                if random.uniform(0,1) < crossing_p:
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
            admr = utils.produce_ADMR(child)
            admr.runADMR()
            chi2 = utils.compute_chi2(admr, child["experiment_doping"], child["experiment_temperature"])
            print(utils.name(child))
            print('this chi2 = '+str(chi2))
            child['chi2'] = chi2

            # If the CHILD has a better fitness than the PARENT, then keep CHILD
            if child['chi2']<last_generation[ii]['chi2']:
                next_gen.append(child)
            # If the CHILD is not better than the PARENT, then keep the PARENT
            else:
                next_gen.append(last_generation[ii])

            # Erase the best chi2 member if found better
            if chi2 < best_chi2_so_far:
                best_chi2_so_far = chi2
                best_dict = deepcopy(child)
                n_improvements += 1
                print(str(n_improvements)+' IMPROVEMENTS SO FAR!')
            print('BEST CHI2 = '+str(best_chi2_so_far) )

        generations_list.append(deepcopy(last_generation))
        last_generation = next_gen

    print('THE END OF TIME')
    print('BEST CHI2 = '+str(best_chi2_so_far) )
    print(utils.name(best_dict))




## Play
genetic_search(init_member,ranges_dict, population_size=6, N_generation=20, mutation_s=0.3, crossing_p=0.5)



# class ObjectView():
#     def __init__(self,dictionary):
#         self.__dict__.update(dictionary)