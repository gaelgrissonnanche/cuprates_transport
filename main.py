import time

from band import BandStructure, HolePocket
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

bandObject = BandStructure()
bandObject.setMuToDoping(0.21)
bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()
bandObject.figMultipleFS2D()


start_total_time = time.time()
ADMRObject = ADMR(bandObject, Bamp=45, gamma_0=15, gamma_k=65, power=12, a0=0)
ADMRObject.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

ADMRObject.fileADMR()
ADMRObject.figADMR()


bandObject2 = HolePocket()
bandObject2.setMuToDoping(0.19)
bandObject2.discretize_FS()
bandObject2.densityOfState()
bandObject2.figMultipleFS2D()

start_total_time = time.time()
ADMRObject2 = ADMR(bandObject2, Bamp=45, gamma_0=15, gamma_k=65, power=12, a0=0)
ADMRObject2.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

ADMRObject2.fileADMR()
ADMRObject2.figADMR()









