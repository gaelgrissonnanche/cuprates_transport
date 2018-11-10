## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import exp, pi, ones, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from admr_routine import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Parameters //////
c = 13.3 # in Angstrom
a = 3.74 # in Angstrom
b = 3.74 # in Angstrom

t   =  190 # meV
tp  = -0.14 * t
tpp =  0.07 * t
tz  =  0.07 * t
tz2 = - 0 * t
mu  = 0.79 * t # van Hove 0.84

## Life time
gamma_0 = 155 # in THz
gamma_k = 700 # in THz
power   = 12

## Magnetic field
B_amp = 45 # in Tesla

## Discretization
mesh_ds = pi / 20 # resolution on arc length of the FS
mesh_z  = 7 # number of kz

## Magnetic field /////////////////////////////////////////////////////////////#
mesh_B_theta = 23 # number of theta angles for B
B_theta_max  = 90 # theta max for B, in degrees
B_phi_a = np.array([0, 15, 30, 45]) * pi / 180
B_theta_a = np.linspace(0, B_theta_max * pi / 180, mesh_B_theta)

## Array of parameters ////////////////////////////////////////////////////////#
mesh_parameters = np.array([mesh_ds, mesh_z], dtype = np.float64)
band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz, tz2], dtype = np.float64)
tau_parameters = np.array([gamma_0, gamma_k, power], dtype = np.float64)

# rho_zz_a = admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = True)


## Fit >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

from lmfit import minimize, Parameters, fit_report

## Interpolate data over theta of simulation

data = np.loadtxt("data_NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_0.dat", dtype = "float", comments = "#")
x = data[:,0]
y = data[:,2]
rzz_0 = np.interp(B_theta_a*180/pi, x, y)
data = np.loadtxt("data_NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_15.dat", dtype = "float", comments = "#")
x = data[:,0]
y = data[:,2]
rzz_15 = np.interp(B_theta_a*180/pi, x, y)
data = np.loadtxt("data_NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_30.dat", dtype = "float", comments = "#")
x = data[:,0]
y = data[:,2]
rzz_30 = np.interp(B_theta_a*180/pi, x, y)
data = np.loadtxt("data_NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_45.dat", dtype = "float", comments = "#")
x = data[:,0]
y = data[:,2]
rzz_45 = np.interp(B_theta_a*180/pi, x, y)


## Function residual ########
def residualFunc(pars, rzz_0, rzz_15, rzz_30, rzz_45):

    gamma_0 = pars["gamma_0"].value
    gamma_k = pars["gamma_k"].value
    power = pars["power"].value
    mu = pars["mu"].value

    print("gamma_0 = ", gamma_0)
    print("gamma_k = ", gamma_k)
    print("power = ", power)
    print("mu = ", mu)

    tau_parameters = np.array([gamma_0, gamma_k, power], dtype = np.float64)
    band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz, tz2], dtype = np.float64)
    rzz_fit_list = admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = False)

    diff_0  = rzz_0  - rzz_fit_list[0]
    diff_15 = rzz_15 - rzz_fit_list[1]
    diff_30 = rzz_30 - rzz_fit_list[2]
    diff_45 = rzz_45 - rzz_fit_list[3]

    return np.concatenate((diff_0, diff_15, diff_30, diff_45))

pars = Parameters()
pars.add("gamma_0", value = gamma_0, min = 0)
pars.add("gamma_k", value = gamma_k, min = 0)
pars.add("power", value = power, vary = False)
pars.add("mu", value = mu, vary = False)

out = minimize(residualFunc, pars, args=(rzz_0, rzz_15, rzz_30, rzz_45))

print(fit_report(out.params))

gamma_0 = out.params["gamma_0"].value
gamma_k = out.params["gamma_k"].value
power = out.params["power"].value
mu = out.params["mu"].value

tau_parameters = np.array([gamma_0, gamma_k, power], dtype = np.float64)
band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz, tz2], dtype = np.float64)
rzz_fit_list = admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = False)


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
## Figures ////////////////////////////////////////////////////////////////#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

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

T = 25
H = 45
####################################################
## Plot Parameters #################################

fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure

fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

axes.axhline(y = 1, ls ="--", c ="k", linewidth = 0.6)

#############################################
fig.text(0.79,0.86, r"NdLSCO 0.21", ha = "right")

fig.text(0.84,0.89, r"$T$ = " + str(T) + " K", ha = "left")
fig.text(0.84,0.82, r"$H$ = " + str(H) + " T", ha = "left")
#############################################

#############################################
axes.set_xlim(0,90)   # limit for xaxis
axes.set_ylim(0.990,1.008) # leave the ymax auto, but fix ymin
axes.set_xlabel(r"$\theta$ ( $^{\circ}$ )", labelpad = 8)
axes.set_ylabel(r"$\rho_{\rm zz}$ / $\rho_{\rm zz}$ ( 0 )", labelpad = 8)
#############################################

## Allow to shift the label ticks up or down with set_pad
for tick in axes.xaxis.get_major_ticks():
    tick.set_pad(7)
for tick in axes.yaxis.get_major_ticks():
    tick.set_pad(8)

colors = ['k', '#3B528B', 'r', '#C7E500']
line = axes.plot(B_theta_a*180/pi, rzz_0)
plt.setp(line, ls ="", c = colors[0], lw = 3, marker = "o", mfc = colors[0], ms = 7, mec = colors[0], mew= 0)
line = axes.plot(B_theta_a*180/pi, rzz_15)
plt.setp(line, ls ="", c = colors[1], lw = 3, marker = "o", mfc = colors[1], ms = 7, mec = colors[1], mew= 0)
line = axes.plot(B_theta_a*180/pi, rzz_30)
plt.setp(line, ls ="", c = colors[2], lw = 3, marker = "o", mfc = colors[2], ms = 7, mec = colors[2], mew= 0)
line = axes.plot(B_theta_a*180/pi, rzz_45)
plt.setp(line, ls ="", c = colors[3], lw = 3, marker = "o", mfc = colors[3], ms = 7, mec = colors[3], mew= 0)

line = axes.plot(B_theta_a*180/pi, rzz_fit_list[0], label = r"$\phi$ = 0")
plt.setp(line, ls ="-", c = colors[0], lw = 3, marker = "", mfc = colors[0], ms = 7, mec = colors[0], mew= 0)
line = axes.plot(B_theta_a*180/pi, rzz_fit_list[1], label = r"$\phi$ = 15")
plt.setp(line, ls ="-", c = colors[1], lw = 3, marker = "", mfc = colors[1], ms = 7, mec = colors[1], mew= 0)
line = axes.plot(B_theta_a*180/pi, rzz_fit_list[2], label = r"$\phi$ = 30")
plt.setp(line, ls ="-", c = colors[2], lw = 3, marker = "", mfc = colors[2], ms = 7, mec = colors[2], mew= 0)
line = axes.plot(B_theta_a*180/pi, rzz_fit_list[3], label = r"$\phi$ = 45")
plt.setp(line, ls ="-", c = colors[3], lw = 3, marker = "", mfc = colors[3], ms = 7, mec = colors[3], mew= 0)

######################################################
plt.legend(loc = 3, fontsize = 16, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
######################################################


##///Set ticks space and minor ticks space ///#
xtics = 30 # space between two ticks
mxtics = xtics / 2.  # space between two minor ticks

majorFormatter = FormatStrFormatter('%g') # put the format of the number of ticks

axes.xaxis.set_major_locator(MultipleLocator(xtics))
axes.xaxis.set_major_formatter(majorFormatter)
axes.xaxis.set_minor_locator(MultipleLocator(mxtics))
axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.show()
fig.savefig("data_NdLSCO_0p21/fit_NdLSCO_0p21.pdf", bbox_inches = "tight")
plt.close()