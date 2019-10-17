import os
import sys
import json
import argparse
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

# to get the parent dir in the path
from inspect import getsourcefile
current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(os.path.sep)])

from bandstructure import BandStructure, Pocket, setMuToDoping, doping
from admr import ADMR
from conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def load_member_from_json(folder="", filename="member"):
    with open(folder + "/" + filename + ".json", "r") as f:
        member = json.load(f)
    return member


def save_member_to_json(member, folder=""):
    admr = produce_ADMR(member)
    path = folder + "/fit_" + \
           "p" + "{0:.2f}".format(member["data_p"]) + "_" + \
           "T" + "{0:.1f}".format(member["data_T"]) + "_" + admr.fileNameFunc() + ".json"
    with open(path, 'w') as f:
        json.dump(member, f, indent=4)
    return path



def produce_ADMR(member):
    bandObject = BandStructure(**member)

    if member["fixdoping"] >=-1 and member["fixdoping"] <=1  :
        bandObject.setMuToDoping(member["fixdoping"])
    bandObject.discretize_FS(PrintEnding=False)
    bandObject.densityOfState()
    bandObject.doping(printDoping=False)

    condObject = Conductivity(bandObject, **member)

    condObject.solveMovementFunc()
    admr = ADMR([condObject], **member)

    return admr



def load_Bphi_data(member, data_dict):
    ## Create array of phi at the selected temperature
    Bphi_array = []
    for t, phi in data_dict.keys():
        if (member["data_T"] == t) * np.isin(phi, np.array(member["Bphi_array"])):
            Bphi_array.append(float(phi)) # put float for JSON

    Bphi_array.sort()
    Bphi_array = np.array(Bphi_array)

    return Bphi_array



def load_Btheta_data(member, data_dict):
    ## Create Initial Btheta
    Btheta_array = np.arange(member["Btheta_min"],
                             member["Btheta_max"] + member["Btheta_step"],
                             member["Btheta_step"])
    ## Create Bphi
    Bphi_array = load_Bphi_data(member, data_dict)

    # Cut Btheta_array to theta_cut
    Btheta_cut_array = np.zeros(len(Bphi_array))
    for i, phi in enumerate(Bphi_array):
        Btheta_cut_array[i] = data_dict[member["data_T"], phi][3]
    Btheta_cut_min = np.min(Btheta_cut_array)  # minimum cut for Btheta_array

    # New Btheta_array with cut off if necessary
    Btheta_array = Btheta_array[Btheta_array <= Btheta_cut_min]

    return Btheta_array



def load_and_interp_data(member, data_dict):
    """
    data_dict[data_T,phi] = [filename, col_theta, col_rzz, theta_cut]
    """
    ## Create Btheta array
    Btheta_array = load_Btheta_data(member, data_dict)

    ## Create array of phi at the selected temperature
    Bphi_array = load_Bphi_data(member, data_dict)

    ## Interpolate data over theta of simulation
    rzz_data_matrix = np.zeros((len(Bphi_array), len(Btheta_array)))
    for i, phi in enumerate(Bphi_array):
        filename = data_dict[member["data_T"], phi][0]
        col_theta = data_dict[member["data_T"], phi][1]
        col_rzz = data_dict[member["data_T"], phi][2]

        data = np.loadtxt(filename, dtype="float", comments="#")
        theta = data[:, col_theta]
        rzz = data[:, col_rzz]
        rzz_i = np.interp(Btheta_array, theta, rzz) # "i" is for interpolated

        rzz_data_matrix[i, :] = rzz_i

    return Bphi_array, Btheta_array, rzz_data_matrix



def compute_chi2(member, data_dict):
    """Compute chi^2"""

    ## Load data
    Bphi_array, Btheta_array, rzz_data_matrix = load_and_interp_data(member, data_dict)

    ## Update Btheta & Bphi function of the data
    member["Bphi_array"]  = list(Bphi_array)
    member["Btheta_min"]  = float(np.min(Btheta_array)) # float need for JSON
    member["Btheta_max"]  = float(np.max(Btheta_array))
    member["Btheta_step"] = float(Btheta_array[1] - Btheta_array[0])

    ## Compute ADMR ------------------------------------------------------------
    admr = produce_ADMR(member)
    admr.runADMR()
    print(admr.fileNameFunc())

    ## Compute Chi^2
    chi2 = 0
    for i in range(Bphi_array.size):
        chi2 += np.sum(np.square(admr.rzz_array[i,:] - rzz_data_matrix[i,:]))

    member['chi2'] = float(chi2)
    return member, admr



def fig_compare(member, data_dict, fig_show=True, fig_save=True, folder=""):
    ## Run ADMR from member parameters -----------------------------------------
    member, admr = compute_chi2(member, data_dict)
    print('this chi2 = ' + "{0:.3e}".format(member["chi2"]))

    ## Load data ---------------------------------------------------------------
    Bphi_array = admr.Bphi_array
    Btheta_cut = np.max(admr.Btheta_array)
    Btheta_data_dict = {}
    rzz_data_dict = {}
    for i, phi in enumerate(Bphi_array):
        filename = data_dict[member["data_T"], phi][0]
        col_theta = data_dict[member["data_T"], phi][1]
        col_rzz = data_dict[member["data_T"], phi][2]

        data = np.loadtxt(filename, dtype="float", comments="#")
        theta = data[:, col_theta]
        rzz = data[:, col_rzz]
        Btheta_data_dict[phi] = theta[theta<=Btheta_cut]
        rzz_data_dict[phi] = rzz[theta<=Btheta_cut]

    ## Plot >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    fig_list = []

    ## Figure Parameters //////////////////////////////////////////////////////#
    for iniCondObject in admr.initialCondObjectDict.values():
        fig_list.append(iniCondObject.figParameters(fig_show=False))


    ## Plot Parameters
    fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
    fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.95)

    axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

    #############################################
    fig.text(0.84,0.89, r"$T$ = " + str(member["data_T"]) + " K", fontsize=14)
    fig.text(0.84, 0.84, r"$B$ = " + str(member["Bamp"]) + " T", fontsize=14)
    fig.text(0.84,0.79, r"$p$ (data) = " + "{0:.2f}".format(member["data_p"]), fontsize=14)
    fig.text(0.84,0.74, r"$p$ (sim) = " + "{0:.3f}".format(admr.totalHoleDoping), fontsize=14)
    fig.text(0.84,0.59, r"$\chi^{\rm 2}$ = " + "{0:.3e}".format(member["chi2"]), fontsize=14)
    #############################################

    #############################################
    axes.set_xlim(0, 90)
    # axes.set_ylim(1+1.2*(min_y-1),1.2*(max_y-1)+1)
    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
    axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)
    #############################################


    colors_dict = {0:'k', 15:'#3B528B', 30:'r', 45:'#C7E500'}

    for i, phi in enumerate(Bphi_array):
        line = axes.plot(Btheta_data_dict[phi], rzz_data_dict[phi], label = r"$\phi$ = " + str(phi))
        plt.setp(line, ls ="-", c = colors_dict[phi], lw = 2, marker = "", mfc = colors_dict[phi], ms = 7, mec = colors_dict[phi], mew= 0)

    for i, phi in enumerate(Bphi_array):
        line = axes.plot(admr.Btheta_array, admr.rzz_array[i,:])
        plt.setp(line, ls ="", c = colors_dict[phi], lw = 3, marker = "o", mfc = colors_dict[phi], ms = 7, mec = colors_dict[phi], mew= 0)

    ######################################################
    plt.legend(loc = 1, fontsize = 14, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
    ######################################################

    ##///Set ticks space and minor ticks space ///#
    xtics = 30 # space between two ticks
    mxtics = xtics / 2.  # space between two minor ticks

    majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

    axes.xaxis.set_major_locator(MultipleLocator(xtics))
    axes.xaxis.set_major_formatter(majorFormatter)
    axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    fig_list.append(fig)
    #///////////////////////////////////////////////////////////////////////////////

    if fig_show == True:
        plt.show()

    ## Save figures list --------------
    if fig_save == True:
        file_figures = PdfPages(folder + "/" +
                                "fit_" + str(member["data_T"]) + "K_" +
                                admr.fileNameFunc() + ".pdf")
        for fig in fig_list[::-1]:
            file_figures.savefig(fig)
        file_figures.close()











# if __name__ == '__main__':
#     admr = produce_ADMR(member)
#     admr.runADMR()
#     chi2 = compute_chi2(admr, data_dict, init_member["data_T"])
#     print('deviation from experiment :',chi2)
#     compare_plot(admr, member["experiment_p"], member["experiment_T"])









## Parser ----------------------------------------------------------------------
# def parse_command():
'''
You can use this python script to define hyperparameters in three ways:
    - Through argparse style arguments
        ex: python genetic.py --tpp -0.1
    - Through a json file named 'member.json' or custom name
        ex: python genetic.py
        with: 'member.json' containing
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
            member_dict = {
                "tpp":-0.1,
                ...
            }
            class ObjectView():
                def __init__(self,dict):
                    self.__dict__.update(dict)
            member = ObjectView(member_dict)
            print(member.tpp)
            ...
    See the the example 'member.json' for a list of all options, see 'random_search.py' for a script like use.
'''

## IF USING ARGUMENTPARSER FOR JSON FILE
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='member.json', help='defines default parameters from a .json file')
## --file has to be followed by xxxx.json, if not it takes member.json as default value

### LOAD PARAMETERS FROM JSON FILE IF AVAILABLE
json_path = parser.parse_known_args()[0].file # it takes the "--file" value in "parser"
if os.path.exists(json_path):
    with open(json_path) as f:
        member = json.load(f)
else:
    print("WARNING: input file '"+json_path+"' not found")
    member = {}

def either_json(key, or_default):
    """If the value is not in the json file, then it takes the default value below"""
    ## If there is a key missing (a parameter missing in the json, it takes the default below)
    try: return member[key]
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
parser.add_argument('--experiment_p',      type=float, default=either_json('experiment_p',25),   help='doping to use to retreive the experimental datafile')
parser.add_argument('--experiment_T', type=float, default=either_json('experiment_T',6),   help='temperature to use to retreive the experimental datafile')
member = parser.parse_known_args()[0].__dict__ # please leave this weird syntax, it is necessary to run with vscode jupyter
