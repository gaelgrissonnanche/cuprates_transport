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
                           mu=-0.784,
                           numberOfKz=7, mesh_ds=pi/20)

## Discretize
# bandObject.setMuToDoping(0.27)
bandObject.discretize_FS()
bandObject.densityOfState()
bandObject.doping(printDoping=True)

# bandObject.figDiscretizeFS2D()
# bandObject.figMultipleFS2D()

## Conductivity

# Arcs
condObject = Conductivity(bandObject, Bamp=45,
                          gamma_0=15, gamma_k=0, gamma_dos=0, power=12, factor_arcs=1000)
condObject.solveMovementFunc()
condObject.figArcs()
# Best fit p = 0.25
# condObject = Conductivity(bandObject, Bamp=45,
#                           gamma_0=15, gamma_k=70, gamma_dos=0, power=12)

# condObject = Conductivity(bandObject, Bamp=45,
#                           gamma_0=10, gamma_k=5, gamma_dos=50, power=2)
# condObject.figLifeTime()

## ADMR
start_total_time = time.time()
amro1band = ADMR([condObject], Bphi_array=[0, 15, 30, 45])
amro1band.runADMR()
print("amro1band time : %.6s seconds" % (time.time() - start_total_time))
amro1band.fileADMR(folder="results_sim")
amro1band.figADMR(folder="results_sim")


