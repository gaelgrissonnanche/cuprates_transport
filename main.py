import time

from band import BandStructure
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

bandObject = BandStructure(mu = -0.825)
bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()
# bandObject.figMultipleFS2D()


start_total_time = time.time()
ADMRObject = ADMR(bandObject, Bamp=45, gamma_0=15, gamma_k=65, power=12, a0=0)
ADMRObject.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

ADMRObject.fileADMR()
ADMRObject.figADMR()











