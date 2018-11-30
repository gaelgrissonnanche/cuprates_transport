import time
from numpy import pi

from band import BandStructure, Pocket
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

h_bandObject = Pocket(mu = -0.82, M = 0.2, mesh_ds = pi / 40, numberOfKz = 7)
# h_bandObject = BandStructure(mu = -0.82, mesh_ds = pi / 40, numberOfKz = 7)
h_bandObject.discretize_FS()
h_bandObject.densityOfState()
h_bandObject.doping()
# h_bandObject.setMuToDoping(pTarget = 0.2)
# print(h_bandObject.mu)
# h_bandObject.discretize_FS()
# h_bandObject.figDiscretizeFS2D()
# h_bandObject.figMultipleFS2D()

h_condObject = Conductivity(h_bandObject, Bamp=45, gamma_0=30, gamma_k=0, power=12)

start_total_time = time.time()
ADMRObject = ADMR(h_condObject)
ADMRObject.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

ADMRObject.fileADMR()
ADMRObject.figADMR()


