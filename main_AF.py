import time
from numpy import pi

from band import BandStructure, HolePocket
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

bandObject = HolePocket(mu = -0.82, M = 0.2, mesh_ds = pi / 40, numberOfKz = 7)
bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()
# bandObject.setMuToDoping(pTarget = 0.2)
# print(bandObject.mu)
# bandObject.discretize_FS()
# bandObject.figDiscretizeFS2D()
# bandObject.figMultipleFS2D()

# condObject = Conductivity(bandObject, Bamp = 45, Bphi = 0, Btheta = 0, gamma_0=1, gamma_k=0, power=12, a0=0)
# condObject.solveMovementFunc()
# condObject.figOnekft(index_kf = 0)

start_total_time = time.time()
ADMRObject = ADMR(bandObject, Bamp=45, gamma_0=25, gamma_k=20, power=12, a0=0)
ADMRObject.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

ADMRObject.fileADMR()
ADMRObject.figADMR()


