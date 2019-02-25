import time
from copy import deepcopy

from numpy import pi
from bandstructure import BandStructure, Pocket, setMuToDoping, doping
from admr import ADMR
from conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## ONE BAND Yawen ////////////////////////////////#
# bandObject = BandStructure(t=533.6, mu=-1.3, tp=-0.213,
#                            tpp=0.044, tz=0.016, tz2=-0.002)
# ## ONE BAND p=0.23 /////////////////////////////#
bandObject = BandStructure(mu=-0.891, mesh_ds=pi/20)

## Discretize ////////////////////////////////////#
# bandObject.setMuToDoping(0.21)
bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping()

# bandObject.figDiscretizeFS2D()
# bandObject.figMultipleFS2D()


condObject = Conductivity(bandObject, Bamp=45, gamma_0=15, gamma_dos=0, gamma_k=65, power=12)
start_total_time = time.time()
amro1band = ADMR([condObject], Bphi_array=[0,45])
amro1band.runADMR()
print("amro1band time : %.6s seconds" % (time.time() - start_total_time))
amro1band.fileADMR(folder="results_sim")
amro1band.figADMR(folder="results_sim", fig_save=True)







## AF reconstruction //////////////////////////////////////////////////////////#
# holePkt = Pocket()
# holePkt.discretize_FS()
# holePkt.densityOfState()
# holePkt.doping()
# dataPoint = Conductivity(holePkt, 45, 0, 0)#, gamma_0=15, gamma_k=65, power=12, gamma_dos=0)
# dataPoint.solveMovementFunc()
# dataPoint.figOnekft()


# bandObject = BandStructure(mu = -0.825)
# # bandObject.setMuToDoping(0.21)
# bandObject.discretize_FS()
# bandObject.densityOfState()
# bandObject.doping()
# # bandObject.figMultipleFS2D()


# start_total_time = time.time()
# ADMRObject = ADMR(bandObject, Bamp=45, gamma_0=15, gamma_k=65, power=12, gamma_dos=0)
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
# ADMRObject2 = ADMR(bandObject2, Bamp=45, gamma_0=50, gamma_k=0, power=12, gamma_dos=0)
# ADMRObject2.runADMR()
# print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

# ADMRObject2.fileADMR()
# ADMRObject2.figADMR()









