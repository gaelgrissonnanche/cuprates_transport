import time
import numpy as np
from band import BandStructure
from admr import ADMR
from chambers import Conductivity


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# t = 533.616
# bandObject = BandStructure(t=t, mu=-693.7/t, tp=-113.561/t,
#                            tpp=23.2192/t, tz=8.7296719/t, tz2=-0.89335299/t, mesh_ds = np.pi / 28)
bandObject = BandStructure(t=533.6, mu=-1.3, tp=-0.213,
                           tpp=0.044, tz=0.016, tz2=-0.002, mesh_ds = np.pi / 28, numberOfKz=16)
bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()
# bandObject.figDiscretizeFS2D()
# bandObject.figDiscretizeFS3D()

vproduct_dict = {}
Btheta_array = np.arange(0, 100, 10)
for theta in Btheta_array:
    condObject = Conductivity(bandObject, Bamp=45, Btheta=theta, gamma_0=25, gamma_k=70, power=12, gamma_dos=0)
    condObject.solveMovementFunc()
    vproduct_dict[theta] = condObject.VelocitiesProduct(2,2)
    index_k = np.arange(1, bandObject.kf.shape[1]+1, 1)
    ## Save Data
    Data = np.vstack((index_k, bandObject.kf[0,:], bandObject.kf[1,:], bandObject.kf[2,:], vproduct_dict[theta]))
    Data = Data.transpose()

    np.savetxt("test_vproduct/vproduct_theta_" + str(theta) + ".dat", Data, fmt='%.7e',
    header = "index_k\tkx[A]\tky[A]\tkz[A]\tvproduct*hbar^2", comments = "#")






# condObject = Conductivity(bandObject, Bamp=45, Btheta=0, gamma_0=25, gamma_k=70, power=12, gamma_dos=0)
# condObject.solveMovementFunc()
# condObject.figOnekft()


# start_total_time = time.time()
# ADMRObject = ADMR(condObject, muteWarnings=True)
# ADMRObject.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

# ADMRObject.fileADMR()
# ADMRObject.figADMR()


# ## Comparison Yawen trajectories //////////////////////////////////////////////#

# start_total_time = time.time()
# ADMRObject = ADMR(condObject)
# ADMRObject.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

# # ADMRObject.fileADMR()
# # ADMRObject.figADMR()


# ## Plots //////////////////////////////////////////////////////////////////////#
# #///// RC Parameters //////#
# mpl.rcdefaults()
# mpl.rcParams['font.size'] = 24. # change the size of the font in every figure
# mpl.rcParams['font.family'] = 'Arial' # font Arial in every figure
# mpl.rcParams['axes.labelsize'] = 24.
# mpl.rcParams['xtick.labelsize'] = 24
# mpl.rcParams['ytick.labelsize'] = 24
# mpl.rcParams['xtick.direction'] = "in"
# mpl.rcParams['ytick.direction'] = "in"
# mpl.rcParams['xtick.top'] = True
# mpl.rcParams['ytick.right'] = True
# mpl.rcParams['xtick.major.width'] = 0.6
# mpl.rcParams['ytick.major.width'] = 0.6
# mpl.rcParams['axes.linewidth'] = 0.6 # thickness of the axes lines
# mpl.rcParams['pdf.fonttype'] = 3  # Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
#                                     # editing the text in illustrator

# fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
# fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95)

# #############################################
# fig.text(0.79,0.28, r"$\tau_{\rm N}$ = " + "{0:.3f}".format(ADMRObject.tau_0) + " ps", ha = "right")
# fig.text(0.79,0.22, r"$\phi$dep$\tau$ = " + "{0:.0f}".format(ADMRObject.gamma_k) + " THz", ha = "right")
# #############################################

# ## Allow to shift the label ticks up or down with set_pad
# for tick in axes.xaxis.get_major_ticks():
#     tick.set_pad(7)
# for tick in axes.yaxis.get_major_ticks():
#     tick.set_pad(8)



# cmap = mpl.cm.get_cmap("jet", len(ADMRObject.Btheta_array))
# colors = cmap(np.arange(len(ADMRObject.Btheta_array)))
# colors[-1]= (1, 0, 0, 1)

# files_list = [
# "trajectory_ang0._phi_0._grixInd_1.dat",
# "trajectory_ang0.116355_phi_0._grixInd_1.dat",
# "trajectory_ang0.232711_phi_0._grixInd_1.dat",
# "trajectory_ang0.349066_phi_0._grixInd_1.dat",
# "trajectory_ang0.465421_phi_0._grixInd_1.dat",
# "trajectory_ang0.581776_phi_0._grixInd_1.dat",
# "trajectory_ang0.698132_phi_0._grixInd_1.dat",
# "trajectory_ang0.814487_phi_0._grixInd_1.dat",
# "trajectory_ang0.930842_phi_0._grixInd_1.dat",
# "trajectory_ang1.0472_phi_0._grixInd_1.dat",
# "trajectory_ang1.16355_phi_0._grixInd_1.dat",
# "trajectory_ang1.27991_phi_0._grixInd_1.dat",
# "trajectory_ang1.39626_phi_0._grixInd_1.dat",
# "trajectory_ang1.51262_phi_0._grixInd_1.dat",
# "trajectory_ang1.62897_phi_0._grixInd_1.dat",
# "trajectory_ang1.74533_phi_0._grixInd_1.dat",
#             ]

# vproduct_array = np.zeros_like(ADMRObject.Btheta_array)

# for i, theta in enumerate(ADMRObject.Btheta_array):
#     t = ADMRObject.condObject_dict[0,theta].t
#     kft = ADMRObject.condObject_dict[0,theta].kft
#     vft = ADMRObject.condObject_dict[0,theta].vft
#     integral = ADMRObject.condObject_dict[0,theta].tOverTauFunc(kft[:, 0, :], vft[:, 0, :])

#     vproduct_array[i] = ADMRObject.vproduct_dict[0,theta][60]
#     print(ADMRObject.vproduct_dict[0,theta][124])

#     # data = np.loadtxt("trajectory_yawen_2018-11-28/" + files_list[i], skiprows = 1)
#     # t_yawen = data[:,3]
#     # integral_yawen = data[:,4]

#     # line = axes.plot(t, integral)
#     # plt.setp(line, ls ="-", c = colors[i], lw = 3, marker = "", mfc = colors[i], ms = 7, mec = colors[i], mew= 0)
#     # fig.text(0.95,0.87-i*0.03, r"{0:.2f}".format(theta), ha="right",
#     #         color =colors[i], fontsize = 14)

#     # line = axes.plot(t, integral)
#     # plt.setp(line, ls ="--", c = 'k', lw = 1, marker = "", mfc = 'k', ms = 7, mec = 'k', mew= 0)


# # #############################################
# # axes.set_xlim(0, None)
# # axes.set_ylim(0, None)
# # axes.set_xlabel(r"t ( ps )", labelpad = 8)
# # axes.set_ylabel(r"$\int_{0}^{t}$ ${\rm \dfrac{dt'}{\tau ( t' )}}$", labelpad = 8)
# # #############################################

# # plt.show()





## Plots v_product //////////////////////////////////////////////////////////#

theta = 0
phi = 0

#///// RC Parameters //////#
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

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95)

#############################################
fig.text(0.84,0.86, r"$\theta$ =" + "{0:.0f}".format(theta), fontsize = 20)
fig.text(0.84,0.80, r"$\phi$ =" + "{0:.0f}".format(phi), fontsize = 20)
#############################################

## Allow to shift the label ticks up or down with set_pad
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)



files_dict = {}
files_dict[0] = "vproduct_yawen/traject_ang0._phi_0._grixInd_.dat"
files_dict[10] = "vproduct_yawen/traject_ang10._phi_0._grixInd_.dat"
files_dict[20] = "vproduct_yawen/traject_ang20._phi_0._grixInd_.dat"
files_dict[30] = "vproduct_yawen/traject_ang30._phi_0._grixInd_.dat"
files_dict[40] = "vproduct_yawen/traject_ang40._phi_0._grixInd_.dat"
files_dict[50] = "vproduct_yawen/traject_ang50._phi_0._grixInd_.dat"
files_dict[60] = "vproduct_yawen/traject_ang60._phi_0._grixInd_.dat"
files_dict[70] = "vproduct_yawen/traject_ang70._phi_0._grixInd_.dat"
files_dict[80] = "vproduct_yawen/traject_ang80._phi_0._grixInd_.dat"
files_dict[90] = "vproduct_yawen/traject_ang20._phi_0._grixInd_.dat"



data = np.loadtxt(files_dict[theta], skiprows = 1)
vproduct_yawen = data[:,3]

line = axes.plot(vproduct_yawen / np.max(vproduct_yawen), label = 'mathematica')
plt.setp(line, ls ="-", c = 'r', lw = 1, marker = "", mfc = 'r', ms = 7, mec = 'r', mew= 0)

line = axes.plot(vproduct_dict[theta] / np.max(vproduct_dict[theta]), label = 'python')
plt.setp(line, ls ="-", c = 'k', lw = 1, marker = "", mfc = 'k', ms = 7, mec = 'k', mew= 0)

plt.legend(loc = 1, fontsize = 10, frameon = False, numpoints = 1, markerscale = 1, handletextpad = 0.5)

# #############################################
axes.set_xlim(0, None)
# axes.set_ylim(0, None)
axes.set_xlabel(r"index $k$", labelpad = 8)
axes.set_ylabel(r"$v$($k$,$0$) $\int_{-\infty}^{0}$ $v$($k$,$t$) exp($\int_{0}^{t}$ ${\rm \dfrac{dt'}{\tau ( t' )}}$) $dt$", labelpad = 8, fontsize = 16)
# #############################################

plt.show()
