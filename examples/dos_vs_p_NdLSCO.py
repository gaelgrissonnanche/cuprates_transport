import numpy as np
from scipy.constants import Boltzmann, Avogadro
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import os
from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
meVolt = 1.602e-22 # 1 eV in Joule
Angstrom = 1e-10 # 1 A in meters
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#



## ONE BAND Horio et al. ///////////////////////////////////////////////////////
params = {
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
    "fixdoping": 0.24,
    "numberOfKz": 61,
    "mesh_ds": 1/100,
    "T" : 0,
}

## Molar Volume for NdLSCO
V_cell = params["a"] * params["b"] * params["c"] # in Angstrom^3
V_molar = Avogadro * V_cell / 2 # in Angstrom^3 / mol (divided by 2 because 2 moles of La2CuO4 in V_cell)

def dos_to_gamma(dos, V_molar):
    """
    Input:
       - dos must be in meV^-1.Angstrom^-3
       - V_molar must in Angstrom^3.mol^-1
    Output:
       - gamma will be in mJ/K^2/mol
    """

    gamma = (np.pi**2/3) * Boltzmann**2 * (dos / meVolt) * V_molar * 1e3

    return gamma


## Array of parameters
tz_array = np.array([0, 0.07]) # in units of t
mu_array = np.linspace(-1.2, -0.54, 2000) # in units of t

## Bandstructure
bandObject = BandStructure(**params)

## Empty matrix
p_matrix = np.empty((len(tz_array), len(mu_array)))
dos_epsilon_matrix = np.empty((len(tz_array), len(mu_array)))
gamma_matrix = np.empty((len(tz_array), len(mu_array)))
mc_matrix = np.empty((len(tz_array), len(mu_array)))

for i, tz in enumerate(tqdm(tz_array, ncols=80, unit="tz", desc="total tz")):
    bandObject.tz = tz
    if tz == 0:
        bandObject.numberOfKz = 1
    else:
        bandObject.numberOfKz = params["numberOfKz"]

    for j, mu in enumerate(tqdm(mu_array, ncols=80, unit="mu", desc="tz = " + str(tz), leave=False)):
        bandObject.mu = mu
        bandObject.doping()
        bandObject.discretize_FS()
        bandObject.dos_k_func()
        bandObject.dos_epsilon_func()
        bandObject.mc_func()

        p_matrix[i,j] = bandObject.p

        ## dos_epsilon is initially in meV^-1 Angstrom^-3
        ## V_cell is in Angstrom
        dos_epsilon_matrix[i,j] = bandObject.dos_epsilon * V_cell / 2 * 1e3 # in eV^-1 for one mole of CuO2 in V_cell / 2

        gamma_matrix[i,j] = dos_to_gamma(bandObject.dos_epsilon, V_molar) # in mJ/K^2/mol
        mc_matrix[i,j] = bandObject.mc # in units of m0


    index = np.argsort(p_matrix[i,:])
    p_matrix[i, :] = p_matrix[i, index]
    dos_epsilon_matrix[i, :] = dos_epsilon_matrix[i, index]
    gamma_matrix[i, :] = gamma_matrix[i, index]
    mc_matrix[i, :] = mc_matrix[i, index]




## Save data
file_path = "sim/NdLSCO_0p25/" + os.path.basename(__file__)[0:-3] + ".dat"
Data_list = []
DataHeader = ""
for i, tz in enumerate(tz_array):
    Data_list.append(p_matrix[i, :])
    Data_list.append(dos_epsilon_matrix[i, :])
    Data_list.append(gamma_matrix[i, :])
    Data_list.append(mc_matrix[i, :])
    DataHeader = "p[tz=" + str(tz) + "]\tDOS(eV^-1)\tgamma(mJ/K^2mol)\tmc(per m0)\t"

Data = np.vstack(Data_list)
Data = Data.transpose()
np.savetxt(file_path, Data, fmt='%.7e', header = DataHeader, comments = "#")



##############################################################################
### Plotting #################################################################
##############################################################################

mpl.rcdefaults()
mpl.rcParams['font.size'] = 24. # change the size of the font in every figure
mpl.rcParams['font.family'] = 'Arial' # font Arial in every figure
mpl.rcParams['axes.labelsize'] = 24.
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6
mpl.rcParams['axes.linewidth'] = 0.6 # thickness of the axes lines
mpl.rcParams['pdf.fonttype'] = 3  # Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
                                    # editing the text in illustrator


####################################################
## Plot Parameters #################################

figures_path = "sim/NdLSCO_0p25/" + os.path.basename(__file__)[0:-3] + ".pdf"

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure

fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes2 = axes.twinx()
axes2.set_axisbelow(True)

## Color map
cmap = mpl.cm.get_cmap("viridis", len(tz_array))
colors = cmap(np.arange(len(tz_array)))
colors[-1] = (1, 0, 0, 1)



### Load and Plot Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

for i, tz in enumerate(tz_array):
    line = axes.plot(p_matrix[i,:], gamma_matrix[i,:], label=r"$\gamma$ ($t_{\rm z}$ = " + "{:.3f}".format(tz) + ")")
    plt.setp(line, ls ="-", c = colors[i], lw = 2, marker = "")

    line = axes2.plot(p_matrix[i,:], mc_matrix[i,:], label=r"$m_{\rm c}$")
    plt.setp(line, ls ="--", c = colors[i], lw = 2, marker = "")


#############################################
axes.set_xlim(0.1,0.4)
axes.tick_params(axis='x', which='major', pad=15)
axes.tick_params(axis='y', which='major', pad=8)
axes2.tick_params(axis='y', which='major', pad=8)
axes.set_xlabel(r"$p$", labelpad = 8)
axes.set_ylabel(r"$\gamma$ ( mJ / K$^2$ mol )")
axes.set_ylim(0, 20)
axes2.set_ylim(bottom=0)
axes2.set_ylabel(r"$m_{\rm c}$", labelpad = 40, rotation = 270)
#############################################


axes.legend(loc = 2, fontsize = 14, frameon = False, numpoints=1, markerscale=1.0, handletextpad=0.5)
axes2.legend(loc = 1, fontsize = 14, frameon = False, numpoints=1, markerscale=1.0, handletextpad=0.5)


# ## Set ticks space and minor ticks space ############
# #####################################################.

xtics = 0.1 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks
majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
######################################################


plt.show()
fig.savefig(figures_path, bbox_inches = "tight")




# ### Load and Plot Data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# for i, tz in enumerate(tz_array):
#     line = axes.plot(p_matrix[i,:], dos_epsilon_matrix[i,:], label=r"$t_{\rm z}$ = " + "{:.2f}".format(tz))
#     plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)


# #############################################
# axes.set_xlim(0.1,0.4)
# axes.tick_params(axis='x', which='major', pad=15)
# axes.tick_params(axis='y', which='major', pad=8)
# axes2.tick_params(axis='y', which='major', pad=8)
# axes.set_xlabel(r"$p$", labelpad = 8)
# axes.set_ylabel(r"DOS ( eV$^{-1}$ )")
# axes.set_ylim(0, 7)
# dosmin, dosmax = axes.axis()[2:]
# axes2.set_ylim(0, dos_to_gamma(dosmax / (1e3 * V_cell / 2), V_molar))
# axes2.set_ylabel(r"$\gamma$ ( mJ / K$^2$ mol )", labelpad = 40, rotation = 270)
# #############################################


# axes.legend(loc = 0, fontsize = 14, frameon = False, numpoints=1, markerscale=1.0, handletextpad=0.5)

# # ## Set ticks space and minor ticks space ############
# # #####################################################.

# xtics = 0.1 # space between two ticks
# mxtics = xtics / 2.  # space between two minor ticks
# majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

# axes.xaxis.set_major_locator(MultipleLocator(xtics))
# axes.xaxis.set_major_formatter(majorFormatter)
# axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
# ######################################################


# plt.show()
# fig.savefig(figures_path, bbox_inches = "tight")

