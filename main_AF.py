import time
from numpy import pi
from copy import deepcopy

from bandstructure import *
from admr import ADMR
from conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<><<<<<<<<<<<<<<<<<<<<

# ONE Band AF ///////////////////////////////////////////////////////////////////
hPocket = Pocket(bandname="hPocket",
                 a=3.74767, b=3.74767, c=13.2,
                 t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
                 M=0.0041,
                 mu=-0.494,
                 numberOfKz=7, mesh_ds=pi/40)

# hPocket = Pocket(bandname="hPocket",
#                  a=3.74767, b=3.74767, c=13.2,
#                  t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
#                  M=0.10,
#                  mu=-0.813,
#                  numberOfKz=7, mesh_ds=pi/40)

ePocket = deepcopy(hPocket)
ePocket.bandname = "ePocket"
ePocket.electronPocket=True

# setMuToDoping([hPocket,ePocket],pTarget=0.21,muStart=-0.9)

# hPocket.figMultipleFS2D()
# ePocket.figMultipleFS2D()
doping([hPocket, ePocket], printDoping=True)

## Discretize
hPocket.discretize_FS(mesh_xy_rough=501)
hPocket.densityOfState()
hPocket.doping()
# hPocket.setMuToDoping(pTarget = 0.21)
# hPocket.discretize_FS()
# print("mu = " + str(hPocket.mu))
# hPocket.figDiscretizeFS2D()
# hPocket.figMultipleFS2D()

## Conductivity
h_condObject = Conductivity(hPocket, Bamp=45,
                            gamma_0=24.2, gamma_k=0, power=12, gamma_dos=0)
# h_condObject = Conductivity(hPocket, Bamp=45,
#                             gamma_0=20, gamma_k=0, power=12, gamma_dos=0)
# h_condObject.solveMovementFunc()
# h_condObject.figOnekft()
# h_condObject.figLifeTime()

## ADMR
start_total_time = time.time()
ADMRObject = ADMR([h_condObject])
ADMRObject.totalHoleDoping = doping([hPocket, ePocket])
ADMRObject.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))
ADMRObject.fileADMR(folder="results_sim")
ADMRObject.figADMR(folder="results_sim")




# ## TWO BANDS //////////////////////////////////////////////////////////////////////
# hPocket = Pocket(bandname="hPocket",
#                  a=3.74767, b=3.74767, c=13.2,
#                  t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
#                  M=0.001,
#                  mu=-0.636,
#                  numberOfKz=7, mesh_ds=pi/100)

# ePocket = deepcopy(hPocket)
# ePocket.electronPocket=True
# ePocket.bandname = "ePocket"
# # setMuToDoping([hPocket,ePocket],pTarget=0.21,muStart=-0.9)
# # hPocket.mu = ePocket.mu = -0.78
# # hPocket.figMultipleFS2D()
# # ePocket.figMultipleFS2D()

# doping([hPocket, ePocket], printDoping=True)

# ## Discretize >>>>>>>>>>>>>>>>>>>>>>>#
# hPocket.discretize_FS(mesh_xy_rough=2001)
# hPocket.densityOfState()
# hPocket.doping()

# ePocket.discretize_FS(mesh_xy_rough=2001)
# ePocket.densityOfState()
# ePocket.doping()

# hPocketCondObject = Conductivity(hPocket, Bamp=45, gamma_0=18.7, gamma_k=0, power=12, gamma_dos=0)
# ePocketCondObject = Conductivity(ePocket, Bamp=45, gamma_0=17.9, gamma_k=0, power=12, gamma_dos=0)
# start_total_time = time.time()
# amro2band = ADMR([ePocketCondObject, hPocketCondObject])
# amro2band.totalHoleDoping = doping([hPocket, ePocket])
# # amro2band = ADMR([hPocketCondObject])
# # amro2band = ADMR([ePocketCondObject])
# amro2band.runADMR()
# print("amro2bands time : %.6s seconds" % (time.time() - start_total_time))
# amro2band.fileADMR(folder="results_sim")
# amro2band.figADMR(folder="results_sim")
