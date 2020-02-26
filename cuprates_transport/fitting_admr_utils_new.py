import os
import sys
import json
import argparse
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import chisquare
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<











# ## Parser ----------------------------------------------------------------------
# # def parse_command():
# '''
# You can use this python script to define hyperparameters in three ways:
#     - Through argparse style arguments
#         ex: python genetic.py --tpp -0.1
#     - Through a json file named 'member.json' or custom name
#         ex: python genetic.py
#         with: 'member.json' containing
#             {
#                 "tpp":-0.1,
#                 ...
#             }
#         or: python genetic.py --file anyname.json
#         with: 'anyname.json' containing
#             {
#                 "tpp":-0.1,
#                 ...
#             }
#     - By passing the parameters as an object containing the arguments as its members:
#         ex: python script.py
#         with: 'script.py' containing
#             import genetic_main as gmain
#             member_dict = {
#                 "tpp":-0.1,
#                 ...
#             }
#             class ObjectView():
#                 def __init__(self,dict):
#                     self.__dict__.update(dict)
#             member = ObjectView(member_dict)
#             print(member.tpp)
#             ...
#     See the the example 'member.json' for a list of all options, see 'random_search.py' for a script like use.
# '''

# ## IF USING ARGUMENTPARSER FOR JSON FILE
# parser = argparse.ArgumentParser()
# parser.add_argument('--file', type=str, default='member.json', help='defines default parameters from a .json file')
# ## --file has to be followed by xxxx.json, if not it takes member.json as default value

# ### LOAD PARAMETERS FROM JSON FILE IF AVAILABLE
# json_path = parser.parse_known_args()[0].file # it takes the "--file" value in "parser"
# if os.path.exists(json_path):
#     with open(json_path) as f:
#         member = json.load(f)
# else:
#     print("WARNING: input file '"+json_path+"' not found")
#     member = {}

# def either_json(key, or_default):
#     """If the value is not in the json file, then it takes the default value below"""
#     ## If there is a key missing (a parameter missing in the json, it takes the default below)
#     try: return member[key]
#     except KeyError: return or_default





# parser.add_argument('--bandname', type=str, default=either_json('bandname',"LargePocket"), help='name used for the')
# parser.add_argument('--a',  type=float, default=either_json('a',  3.74767), help='lattice constant along x')
# parser.add_argument('--b',  type=float, default=either_json('b',  3.74767), help='lattice constant along y')
# parser.add_argument('--c',  type=float, default=either_json('c',  13.2000), help='lattice constant along z')
# parser.add_argument('--t',  type=float, default=either_json('t',  190),     help='first neighbor hopping')
# parser.add_argument('--tp', type=float, default=either_json('tp',-0.14),    help='second neighbor hopping')
# parser.add_argument('--tpp',type=float, default=either_json('tpp',0.07),    help='third neighbor hopping')
# parser.add_argument('--tz', type=float, default=either_json('tz', 0.07),    help='interplane hopping')
# parser.add_argument('--tz2',type=float, default=either_json('tz2',0.0),     help='artificial interplane hopping ')
# parser.add_argument('--mu', type=float, default=either_json('mu',-0.826),   help='chemical potential')
# parser.add_argument('--fixdoping', type=float, default=either_json('fixdoping', -2), help='try fix the doping to the provided value if between -1 and 1, otherwise use the provided mu' )
# parser.add_argument('--numberOfKz',     type=int,   default=either_json('numberOfKz',7),     help='density of points in the kz integral')
# parser.add_argument('--mesh_ds',        type=float, default=either_json('mesh_ds',np.pi/20), help='spacing of points in the angular integral')
# parser.add_argument('--Bamp',           type=float, default=either_json('Bamp',45),           help='Amplitude of the magnetic field (in Tesla)')
# parser.add_argument('--gamma_0',        type=float, default=either_json('gamma_0',0),        help='constant scattering rate (in units of t)')
# parser.add_argument('--gamma_k',        type=float, default=either_json('gamma_k',0),        help='k-dependent scattering rate (in units of t) following a cos^power(theta) dependence (see power below)')
# parser.add_argument('--gamma_dos_max',  type=float, default=either_json('gamma_dos_max',275),help='I do not know this one, ask Gael :)')
# parser.add_argument('--power',          type=int,   default=either_json('power',12),         help='power of the cos dependence of the k-dependent scattering rate')
# parser.add_argument('--seed',           type=int,   default=either_json('seed',72),          help='Random generator seed (for reproducibility)')
# parser.add_argument('--experiment_p',      type=float, default=either_json('experiment_p',25),   help='doping to use to retreive the experimental datafile')
# parser.add_argument('--experiment_T', type=float, default=either_json('experiment_T',6),   help='temperature to use to retreive the experimental datafile')
# member = parser.parse_known_args()[0].__dict__ # please leave this weird syntax, it is necessary to run with vscode jupyter
