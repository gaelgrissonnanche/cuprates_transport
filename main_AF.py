import time
from numpy import pi
from copy import deepcopy

from bandstructure import *
from admr import ADMR
from conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# ## ONE Band AF ////////////////////////////////////////////////////////////////#
# h_bandObject = Pocket(mu=-0.825, M=0.2, mesh_ds = pi/40, numberOfKz = 7)
# ## Discretize >>>>>>>>>>>>>>>>>>>>>>>#
# h_bandObject.discretize_FS()
# h_bandObject.densityOfState()
# h_bandObject.doping()
# # h_bandObject.setMuToDoping(pTarget = 0.2)
# # h_bandObject.discretize_FS()
# # print("mu = " + str(h_bandObject.mu))
# # h_bandObject.figDiscretizeFS2D()
# # h_bandObject.figMultipleFS2D()
# ## Conductivity >>>>>>>>>>>>>>>>>>>>>>>#
# h_condObject = Conductivity(h_bandObject, Bamp=45, gamma_0=15, gamma_k=0, power=12)
# ## ADMR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# start_total_time = time.time()
# ADMRObject = ADMR([h_condObject])
# ADMRObject.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))
# ADMRObject.fileADMR(folder="results_sim")
# ADMRObject.figADMR()

# ## ONE Band AF Yawen Parameters ///////////////////////////////////////////////#
# t = 190
# h_bandObject = Pocket(t=t, tp=-76/t, tpp=13.397/t, tz=13.397/t, tz2=0,
#                       mu = -100/t, M=10/t,
#                       mesh_ds = pi/80, numberOfKz=7)
# ## Discretize >>>>>>>>>>>>>>>>>>>>>>>#
# h_bandObject.discretize_FS()
# h_bandObject.densityOfState()
# h_bandObject.doping()
# # h_bandObject.setMuToDoping(pTarget = 0.2)
# # print(h_bandObject.mu)
# # h_bandObject.discr÷etize_FS()
# h_bandObject.figDiscretizeFS2D()
# # h_bandObject.figMultipleFS2D()
# ## Conductivity >>>>>>>>>>>>>>>>>>>>>>>#
# h_condObject = Conductivity(h_bandObject, Bamp=45, gamma_0=14.28, gamma_k=0, power=12)
# ## ADMR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# start_total_time = time.time()
# ADMRObject = ADMR([h_condObject])
# ADMRObject.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))
# ADMRObject.fileADMR(folder="results_sim")
# ADMRObject.figADMR()


## TWO BAND p=0.19 //////////////////////////////////////////////////////////#
hPocket = Pocket(bandname="hPocket", mu = -0.825, M = 0.05, mesh_ds = pi/15, numberOfKz = 7)
# hPocket.tp = -0.24
ePocket = deepcopy(hPocket)
ePocket.electronPocket=True
# setMuToDoping([hPocket,ePocket],0.15,muStart=-0.9)
hPocket.mu = ePocket.mu = -0.78
ePocket.bandname = "ePocket"
# hPocket.figMultipleFS2D()
# ePocket.figMultipleFS2D()

doping([hPocket, ePocket])

## Discretize >>>>>>>>>>>>>>>>>>>>>>>#
# ePocket.mesh_ds=pi/30
hPocket.discretize_FS()
hPocket.densityOfState()
hPocket.doping()
# ePocket.mesh_ds=pi/30
ePocket.discretize_FS()
ePocket.densityOfState()
ePocket.doping()

hPocketCondObject = Conductivity(hPocket, Bamp=45, gamma_0=20, gamma_k=0, power=12, gamma_dos=0)
ePocketCondObject = Conductivity(ePocket, Bamp=45, gamma_0=10, gamma_k=0, power=12, gamma_dos=0)
start_total_time = time.time()
amro2band = ADMR([hPocketCondObject,ePocketCondObject])
# amro2band = ADMR([hPocketCondObject])
# amro2band = ADMR([ePocketCondObject])
amro2band.runADMR()
print("amro2bands time : %.6s seconds" % (time.time() - start_total_time))
amro2band.fileADMR(folder="results_sim")
amro2band.figADMR(folder="results_sim", fig_save=True)
