import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# to get the parent dir in the path
from inspect import getsourcefile
current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])

from bandstructure import BandStructure, Pocket, setMuToDoping, doping
from admr import ADMR
from conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# def parse_command():
'''
You can use this python script to define hyperparameters in three ways:
    - Through argparse style arguments
        ex: python genetic.py --tpp -0.1
    - Through a json file named 'params.json' or custom name
        ex: python genetic.py
        with: 'params.json' containing
            {
                "tpp":-0.1,
                ...
            }
        or: python genetic.py --file anyname.json
        with: 'anyname.json' containing
            {
                "tpp":-0.1,
                ...
            }
    - By passing the parameters as an object containing the arguments as its members:
        ex: python script.py
        with: 'script.py' containing
            import genetic_main as gmain
            params_dict = {
                "tpp":-0.1,
                ...
            }
            class ObjectView():
                def __init__(self,dict):
                    self.__dict__.update(dict)
            params = ObjectView(params_dict)
            print(params.tpp)
            ...
    See the the example 'params.json' for a list of all options, see 'random_search.py' for a script like use.
'''

## IF USING ARGUMENTPARSER FOR JSON FILE
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='params.json', help='defines default parameters from a .json file')
## --file has to be followed by xxxx.json, if not it takes params.json as default value

### LOAD PARAMETERS FROM JSON FILE IF AVAILABLE
json_path = parser.parse_known_args()[0].file # it takes the "--file" value in "parser"
if os.path.exists(json_path):
    with open(json_path) as f:
        params = json.load(f)
else:
    print("WARNING: input file '"+json_path+"' not found")
    params = {}

def either_json(key, or_default):
    """If the value is not in the json file, then it takes the default value below"""
    ## If there is a key missing (a parameter missing in the json, it takes the default below)
    try: return params[key]
    except KeyError: return or_default





parser.add_argument('--bandname', type=str, default=either_json('bandname',"LargePocket"), help='name used for the')
parser.add_argument('--a',  type=float, default=either_json('a',  3.74767), help='lattice constant along x')
parser.add_argument('--b',  type=float, default=either_json('b',  3.74767), help='lattice constant along y')
parser.add_argument('--c',  type=float, default=either_json('c',  13.2000), help='lattice constant along z')
parser.add_argument('--t',  type=float, default=either_json('t',  190),     help='first neighbor hopping')
parser.add_argument('--tp', type=float, default=either_json('tp',-0.14),    help='second neighbor hopping')
parser.add_argument('--tpp',type=float, default=either_json('tpp',0.07),    help='third neighbor hopping')
parser.add_argument('--tz', type=float, default=either_json('tz', 0.07),    help='interplane hopping')
parser.add_argument('--tz2',type=float, default=either_json('tz2',0.0),     help='artificial interplane hopping ')
parser.add_argument('--mu', type=float, default=either_json('mu',-0.826),   help='chemical potential')
parser.add_argument('--fixdoping', type=float, default=either_json('fixdoping', -2), help='try fix the doping to the provided value if between -1 and 1, otherwise use the provided mu' )
parser.add_argument('--numberOfKz',     type=int,   default=either_json('numberOfKz',7),     help='density of points in the kz integral')
parser.add_argument('--mesh_ds',        type=float, default=either_json('mesh_ds',np.pi/20), help='spacing of points in the angular integral')
parser.add_argument('--Bamp',           type=float, default=either_json('Bamp',45),           help='Amplitude of the magnetic field (in Tesla)')
parser.add_argument('--gamma_0',        type=float, default=either_json('gamma_0',0),        help='constant scattering rate (in units of t)')
parser.add_argument('--gamma_k',        type=float, default=either_json('gamma_k',0),        help='k-dependent scattering rate (in units of t) following a cos^power(theta) dependence (see power below)')
parser.add_argument('--gamma_dos_max',  type=float, default=either_json('gamma_dos_max',275),help='I do not know this one, ask Gael :)')
parser.add_argument('--power',          type=int,   default=either_json('power',12),         help='power of the cos dependence of the k-dependent scattering rate')
parser.add_argument('--seed',           type=int,   default=either_json('seed',72),          help='Random generator seed (for reproducibility)')
parser.add_argument('--experiment_doping',      type=float, default=either_json('experiment_doping',25),   help='doping to use to retreive the experimental datafile')
parser.add_argument('--experiment_temperature', type=float, default=either_json('experiment_temperature',6),   help='temperature to use to retreive the experimental datafile')
params = parser.parse_known_args()[0] # please leave this weird syntax, it is necessary to run with vscode jupyter

# return params

def name(member):
    return 'admr_t{}tp{}tpp{}tz{}tz2{}mu{}_nkz{}mds{}_go{}gk{}gd{}pow{}'.format(
                round(member["t"],3),
                round(member["tp"],3),
                round(member["tpp"],3),
                round(member["tz"],3),
                round(member["tz2"],3),
                round(member["mu"],3),
                member["numberOfKz"],
                round(member["mesh_ds"]*np.pi,1),
                round(member["gamma_0"],3),
                round(member["gamma_k"],3),
                member["gamma_dos_max"],
                member["power"],
            )


def dump_params(member):
    with open("params_"+name(member)+".json", 'w') as f:
        json.dump(vars(member), f, indent=4)


def produce_ADMR(member):
    bandObject = BandStructure(
                        bandname=member["bandname"],
                        a=member["a"],
                        b=member["b"],
                        c=member["c"],
                        t=member["t"],
                        tp=member["tp"],
                        tpp=member["tpp"],
                        tz=member["tz"],
                        tz2=member["tz2"],
                        mu=member["mu"],
                        numberOfKz=member["numberOfKz"],
                        mesh_ds=member["mesh_ds"]
                    )

    if member["fixdoping"] >=-1 and member["fixdoping"] <=1  :
        bandObject.setMuToDoping(member["fixdoping"])
    bandObject.discretize_FS()
    bandObject.densityOfState()
    bandObject.doping(printDoping=True)

    condObject = Conductivity(bandObject,
                        Bamp=member["Bamp"],
                        gamma_0=member["gamma_0"],
                        gamma_k=member["gamma_k"],
                        gamma_dos_max=member["gamma_dos_max"],
                        power=member["power"]
                    )

    condObject.solveMovementFunc()
    admr = ADMR([condObject], Bphi_array=[0, 15, 30, 45])
    return admr

def get_available_phis(p, T):
    phis=[]
    if p == 21:
        if   T==16: phis= [0,15,30]
        elif T==18: phis= [0]
        elif T==20: phis= [0,15,30,45]
        elif T==25: phis= [0,15,30,45]
    elif p == 25:
        if   T==6 : phis= [0,15,45]
        elif T==12: phis= [0,15,45]
        elif T==20: phis= [0,15,30,45]
        elif T==25: phis= [0,15,30,45]
    return phis

def get_data(p, phi, T, folder='../'):
    exp_data = None
    if p == 25:
        folder = folder + 'data_NdLSCO_0p25/'
        filename = folder +'0p25_'+str(phi)+'degr_45T_'+str(T)+'K.dat'
        exp_data = np.transpose(np.loadtxt(filename, delimiter='\t'))

    elif p == 21:
        folder = folder +'data_NdLSCO_0p21/'
        filename = folder + 'NdLSCO_0p21_1808A_c_AS_T_'+str(T)+'_H_45_phi_'+str(phi)+'.dat'
        exp_data = np.transpose(np.loadtxt(filename, delimiter=' '))
        exp_data = exp_data[0::2] ## remove middle column
    else:
        raise(ValueError('Nno data for p='+str(p)))
    return exp_data

def compute_chi2(admr, doping, temperature):
    """Compute chi^2"""
    chi2 = 0
    phis = get_available_phis(doping,temperature)
    for phi in phis:
        ii = np.where(admr.Bphi_array == phi )[0]
        exp_data = get_data(doping, phi, temperature)
        data_function = interp1d(exp_data[0], exp_data[1])
        chi2 += np.sum(np.square( admr.rzz_array[ii,-2] - data_function(admr.Btheta_array[:-2]) ))
    return chi2

def compare_plot(admr, doping, temperature, filename=None):
    phis = get_available_phis(doping,temperature)
    for phi in phis:
        ii = np.where(admr.Bphi_array == phi )[0][0]
        exp_data = get_data(doping, phi, temperature)

        colors = ['#000000', '#3B528B', '#FF0000', '#C7E500']

        plt.plot(admr.Btheta_array, admr.rzz_array[ii], 'o', c=colors[ii])
        plt.plot(exp_data[0], exp_data[1], '-', c=colors[ii])
        data_function = interp1d(exp_data[0], exp_data[1])
        plt.plot(admr.Btheta_array[:-2], data_function(admr.Btheta_array[:-2]), '+', c=colors[ii])
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename+'.pdf')
        plt.clf()

if __name__ == '__main__':
    admr = produce_ADMR(params)
    print(params.fixdoping)

    admr.runADMR()
    chi2 = compute_chi2(admr, params.experiment_doping, params.experiment_temperature)
    print('deviation from experiment :',chi2)
    compare_plot(admr,params.experiment_doping, params.experiment_temperature)
