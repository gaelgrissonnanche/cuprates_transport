import numpy as np
import random
from copy import deepcopy
import genetic_utils as utils

init_args_dict = {
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

search_ranges = {
    "t": [150.0,250.0],
    "tp": [-0.16,-0.12],
    "tpp": [0.05,0.09],
    "tz": [0.05,0.09],
    "mu": [-0.85,-0.8],
    "gamma_0": [1.0,20.0],
    # "gamma_k": [60.0,70.0],
    "gamma_dos_max": [10.0,300.0],
    "power":[2,10,12,14]
}

class ObjectView():
    def __init__(self,dictionary):
        self.__dict__.update(dictionary)

def genetic_search(args_dict, ranges,
        population_size=20, N_generation=20, mutation_s=0.1, cross_p=0.7):
    np.random.seed(args_dict['seed'])

    print('\nINITIAL PARAMETER SET')
    args = ObjectView(args_dict)
    admr = utils.produce_ADMR(args)
    admr.runADMR()
    chi2 = utils.compute_xp_dev(admr, int(args.experiment_doping), int(args.experiment_temperature))
    print(utils.name(args))
    print('this chi2 = '+str(chi2))
    args_dict['chi2'] = chi2

    best_chi2_so_far = chi2
    best_dict = args_dict
    print('BEST CHI2 = '+str(best_chi2_so_far) )
    n_improvements = 0

    print('\nINITIAL POPULATION')
    gen0 = []
    for ii in range(population_size):
        print('\nGENERATION 0 MEMBER '+str(ii+1))
        this_args_dict = deepcopy(args_dict)
        for key, ran in ranges.items():
            if len(ran)>2:
                value = random.choice(ran)
            elif type(ran[0])==int and type(ran[1])==int:
                value = random.randint(ran[0],ran[1])
            else:
                value = random.uniform(ran[0],ran[1])
            this_args_dict[key] = value   
        
        ## initial fitness
        args = ObjectView(this_args_dict)
        admr = utils.produce_ADMR(args)
        admr.runADMR()
        chi2 = utils.compute_xp_dev(admr, int(args.experiment_doping), int(args.experiment_temperature))
        print(utils.name(args))
        print('this chi2 = '+str(chi2))
        this_args_dict['chi2'] = chi2

        gen0.append(this_args_dict)
        if chi2 < best_chi2_so_far:
            best_chi2_so_far = chi2
            best_dict = this_args_dict
            n_improvements += 1
            print(str(n_improvements)+' IMPROVEMENTS SO FAR!')
        print('BEST CHI2 = '+str(best_chi2_so_far) )
        

    generations_list = []
    last_gen = gen0
    for n in range(N_generation):
        next_gen = []

        print('\n\nMUTATION CYCLE '+str(n+1))
        for ii in range(population_size):
            print('\nGENERATION '+str(n+1)+' MEMBER '+str(ii+1))

            this_dict = last_gen[ii]
            muta_dict = deepcopy(init_args_dict)
            for key, ran in search_ranges.items():
                if random.uniform(0,1) < cross_p:
                    # crossing
                    muta_dict[key] = this_dict[key]
                else:
                    # mutation
                    if len(ran)>2: #then the values are discrete
                        muta_dict[key] = random.choice(ran)
                    else :
                        dict1 = random.choice(last_gen)
                        dict2 = random.choice(last_gen)
                        dict3 = random.choice(last_gen)
                        muta_dict[key] = dict1[key] + mutation_s*(dict2[key]-dict3[key]) 
            
            #fitness
            args = ObjectView(muta_dict)
            admr = utils.produce_ADMR(args)
            admr.runADMR()
            chi2 = utils.compute_xp_dev(admr, args.experiment_doping, args.experiment_temperature)
            print(utils.name(args))
            print('this chi2 = '+str(chi2))
            muta_dict['chi2'] = chi2
        
            if muta_dict['chi2']<last_gen[ii]['chi2']:
                next_gen.append(muta_dict)
            else:
                next_gen.append(last_gen[ii])
            
            if chi2 < best_chi2_so_far:
                best_chi2_so_far = chi2
                best_dict = muta_dict
                n_improvements += 1
                print(str(n_improvements)+' IMPROVEMENTS SO FAR!')
            print('BEST CHI2 = '+str(best_chi2_so_far) )
        
        generations_list.append(deepcopy(last_gen))
        last_gen = next_gen

    print('THE END OF TIME')
    print('BEST CHI2 = '+str(best_chi2_so_far) )
    print(utils.name(ObjectView(best_dict)))

genetic_search(init_args_dict,search_ranges, 6, 20, 0.3, 0.5)