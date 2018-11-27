import time

from band import BandStructure, HolePocket
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


# holePkt = HolePocket()
# holePkt.discretize_FS()
# holePkt.densityOfState()
# holePkt.doping()
# dataPoint = Conductivity(holePkt, 45, 0, 0)#, gamma_0=15, gamma_k=65, power=12, a0=0)
# dataPoint.solveMovementFunc()
# dataPoint.figOnekft()


# bandObject = BandStructure()
# bandObject.setMuToDoping(0.21)
# bandObject.discretize_FS()
# bandObject.densityOfState()
# bandObject.doping()
# bandObject.figMultipleFS2D()


# start_total_time = time.time()
# ADMRObject = ADMR(bandObject, Bamp=45, gamma_0=15, gamma_k=65, power=12, a0=0)
# ADMRObject.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

# ADMRObject.fileADMR()
# ADMRObject.figADMR()


bandObject2 = HolePocket(M=0.5)
bandObject2.mu=-0.9
bandObject2.doping()
bandObject2.discretize_FS()
bandObject2.densityOfState()
# bandObject2.figMultipleFS2D()

start_total_time = time.time()
ADMRObject2 = ADMR(bandObject2, Bamp=45, gamma_0=5, gamma_k=0, power=12, a0=0)
ADMRObject2.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

ADMRObject2.fileADMR()
ADMRObject2.figADMR()









