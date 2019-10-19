import time
from numpy import pi
from bandstructure import BandStructure, Pocket, setMuToDoping, doping
from admr import ADMR
from conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


## ONE BAND Horio et al. /////////////////////////////////////////////////////////
bandObject = BandStructure(bandname="LargePocket",
                           a=3.74767, b=3.74767, c=13.2,
                           t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00,
                           mu=-0.826,
                           numberOfKz=7, mesh_ds=1/20)

## Discretize
# bandObject.setMuToDoping(0.22)
bandObject.doping(printDoping=True)
bandObject.discretize_FS()
bandObject.densityOfState()

# bandObject.figDiscretizeFS2D()
# bandObject.figMultipleFS2D()

## Conductivity
condObject = Conductivity(bandObject, Bamp=45, T=0,
                          gamma_0=15.0, gamma_k=66, power=12, gamma_dos_max=0)
# condObject.figdfdE()
# condObject.solveMovementFunc()
# condObject.figScatteringPhi(kz=0)
# condObject.figScatteringPhi(kz=pi/bandObject.c)
# condObject.figScatteringPhi(kz=2*pi/bandObject.c)
# condObject.figArcs()
# Best fit p = 0.25
# condObject = Conductivity(bandObject, Bamp=45,
#                           gamma_0=15, gamma_k=70, gamma_dos_max=0, power=12)

# condObject = Conductivity(bandObject, Bamp=45,
#                           gamma_0=10, gamma_k=5, gamma_dos_max=50, power=2)


## ADMR
start_total_time = time.time()
amro1band = ADMR([condObject], Bphi_array=[0, 15, 30, 45])
amro1band.runADMR()
print("amro1band time : %.6s seconds" % (time.time() - start_total_time))
amro1band.fileADMR(folder="results_sim")
amro1band.figADMR(folder="results_sim")
