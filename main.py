import time

from numpy import pi
from band import BandStructure, Pocket, setMuToDoping, doping
from admr import ADMR
from chambers import Conductivity
from copy import deepcopy
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# #### ONE BAND p=0.21 ############################
# bandObject = BandStructure(mu = -0.825)
# # bandObject.setMuToDoping(0.21)
# bandObject.discretize_FS()
# bandObject.densityOfState()
# bandObject.doping()
# # bandObject.figMultipleFS2D()

# condObject = Conductivity(bandObject, Bamp=45, gamma_0=15, gamma_k=65, power=12, a0=0)
# start_total_time = time.time()
# amro1band = ADMR(condObject)
# amro1band.runADMR()
# print("amro1band time : %.6s seconds" % (time.time() - start_total_time))
# # ADMRObject.fileADMR()
# amro1band.figADMR()


## TWO BAND p=0.19 ############################
hPocket = Pocket(M=0.2)
hPocket.tp = -0.24
ePocket = deepcopy(hPocket)
ePocket.electronPocket=True
# setMuToDoping([hPocket,ePocket],0.18,-0.9)
hPocket.mu = ePocket.mu = -0.967
# hPocket.figMultipleFS2D()
# ePocket.figMultipleFS2D()
hPocket.mesh_ds=pi/30
hPocket.discretize_FS()
hPocket.densityOfState()
hPocket.doping()
ePocket.mesh_ds=pi/30
ePocket.discretize_FS()
ePocket.densityOfState()
ePocket.doping()

hPocketCondObject = Conductivity(hPocket, Bamp=45, gamma_0=15, gamma_k=0, power=12, a0=0)
ePocketCondObject = Conductivity(ePocket, Bamp=45, gamma_0=20, gamma_k=0, power=12, a0=0)
start_total_time = time.time()
amro2band = ADMR([hPocketCondObject,ePocketCondObject],muteWarnings=True)
# amro2band = ADMR([hPocketCondObject])
amro2band.runADMR()
print("amro2bands time : %.6s seconds" % (time.time() - start_total_time))
amro2band.figADMR()




## AF reconstruction //////////////////////////////////////////////////////////#
# holePkt = Pocket()
# holePkt.discretize_FS()
# holePkt.densityOfState()
# holePkt.doping()
# dataPoint = Conductivity(holePkt, 45, 0, 0)#, gamma_0=15, gamma_k=65, power=12, a0=0)
# dataPoint.solveMovementFunc()
# dataPoint.figOnekft()


# bandObject = BandStructure(mu = -0.825)
# # bandObject.setMuToDoping(0.21)
# bandObject.discretize_FS()
# bandObject.densityOfState()
# bandObject.doping()
# # bandObject.figMultipleFS2D()


# start_total_time = time.time()
# ADMRObject = ADMR(bandObject, Bamp=45, gamma_0=15, gamma_k=65, power=12, a0=0)
# ADMRObject.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

# ADMRObject.fileADMR()
# ADMRObject.figADMR()

# # bandObject2 = BandStructure()
# bandObject2 = Pocket(M=0.8)
# bandObject2.mu=-0.9
# bandObject2.mesh_ds=pi/20
# # bandObject2.electronPocket=True
# bandObject2.doping()
# bandObject2.discretize_FS()
# bandObject2.densityOfState()
# # bandObject2.figMultipleFS2D()

# start_total_time = time.time()
# ADMRObject2 = ADMR(bandObject2, Bamp=45, gamma_0=50, gamma_k=0, power=12, a0=0)
# ADMRObject2.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

# ADMRObject2.fileADMR()
# ADMRObject2.figADMR()









