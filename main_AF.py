import time
from numpy import pi
from copy import deepcopy

from bandstructure import *
from admr import ADMR
from conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## ONE Band AF ////////////////////////////////////////////////////////////////#
hPocket = Pocket(bandname="hPocket",
                 a=3.74767, b=3.74767, c=13.2,
                 t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
                 M=0.01,
                 mu=-0.77,
                 numberOfKz=7, mesh_ds=pi/40)

# hPocket = Pocket(bandname="hPocket",
#                  a=3.74767, b=3.74767, c=13.2,
#                  t=190, tp=-0.115, tpp=0.095, tz=0.058, tz2=0.00,
#                  M=0.011,
#                  mu=-0.47,
#                  numberOfKz=7, mesh_ds=pi/40)

ePocket = deepcopy(hPocket)
ePocket.bandname = "ePocket"
ePocket.electronPocket=True

# setMuToDoping([hPocket,ePocket],pTarget=0.21,muStart=-0.9)

# hPocket.figMultipleFS2D()
# ePocket.figMultipleFS2D()
doping([hPocket, ePocket])

## Discretize
hPocket.discretize_FS()
hPocket.densityOfState()
hPocket.doping()
# hPocket.setMuToDoping(pTarget = 0.21)
# hPocket.discretize_FS()
# print("mu = " + str(hPocket.mu))
# hPocket.figDiscretizeFS2D()
# hPocket.figMultipleFS2D()

## Conductivity
h_condObject = Conductivity(hPocket, Bamp=45,
                            gamma_0=15, gamma_k=0, power=12, gamma_dos=0)

## ADMR
start_total_time = time.time()
ADMRObject = ADMR([h_condObject])
ADMRObject.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))
ADMRObject.fileADMR(folder="results_sim")
ADMRObject.figADMR(folder="results_sim")

# ## ONE Band AF Yawen Parameters ///////////////////////////////////////////////#
# t = 190
# hPocket = Pocket(t=t, tp=-76/t, tpp=13.397/t, tz=13.397/t, tz2=0,
#                       mu = -100/t, M=10/t,
#                       mesh_ds = pi/80, numberOfKz=7)

# ## Discretize
# hPocket.discretize_FS()
# hPocket.densityOfState()
# hPocket.doping()
# # hPocket.setMuToDoping(pTarget = 0.2)
# # print(hPocket.mu)
# # hPocket.discretize_FS()
# # hPocket.figDiscretizeFS2D()
# # hPocket.figMultipleFS2D()

# ## Conductivity
# h_condObject = Conductivity(hPocket, Bamp=45,
#                             gamma_0=14.28, gamma_k=0, power=12)

# ## ADMR
# start_total_time = time.time()
# ADMRObject = ADMR([h_condObject])
# ADMRObject.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))
# ADMRObject.fileADMR(folder="results_sim")
# ADMRObject.figADMR(folder="results_sim")


# ## TWO BAND p=0.19 //////////////////////////////////////////////////////////#
# hPocket = Pocket(bandname="hPocket", mu = -0.825, M = 0.05, mesh_ds = pi/15, numberOfKz = 7)
# # hPocket.tp = -0.24
# ePocket = deepcopy(hPocket)
# ePocket.electronPocket=True
# # setMuToDoping([hPocket,ePocket],0.15,muStart=-0.9)
# hPocket.mu = ePocket.mu = -0.78
# ePocket.bandname = "ePocket"
# # hPocket.figMultipleFS2D()
# # ePocket.figMultipleFS2D()

# doping([hPocket, ePocket])

# ## Discretize >>>>>>>>>>>>>>>>>>>>>>>#
# # ePocket.mesh_ds=pi/30
# hPocket.discretize_FS()
# hPocket.densityOfState()
# hPocket.doping()
# # ePocket.mesh_ds=pi/30
# ePocket.discretize_FS()
# ePocket.densityOfState()
# ePocket.doping()

# hPocketCondObject = Conductivity(hPocket, Bamp=45, gamma_0=20, gamma_k=0, power=12, gamma_dos=0)
# ePocketCondObject = Conductivity(ePocket, Bamp=45, gamma_0=10, gamma_k=0, power=12, gamma_dos=0)
# start_total_time = time.time()
# amro2band = ADMR([hPocketCondObject,ePocketCondObject])
# # amro2band = ADMR([hPocketCondObject])
# # amro2band = ADMR([ePocketCondObject])
# amro2band.runADMR()
# print("amro2bands time : %.6s seconds" % (time.time() - start_total_time))
# amro2band.fileADMR(folder="results_sim")
# amro2band.figADMR(folder="results_sim", fig_save=True)
