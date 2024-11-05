from scipy import optimize
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


## Functions to compute the doping of a two bands system and more ---------------#

def doping(bandIterable, printDoping=False):
    totalFilling=0
    if printDoping is True:
        print("------------------------------------------------")
    for band in bandIterable:
        band.update_filling()
        totalFilling += band.n
        if printDoping is True:
            print(band.band_name + ": band filling = " + "{0:.3f}".format(band.n))
    doping = 1-totalFilling
    if printDoping is True:
        print("total hole doping = " + "{0:.3f}".format(doping))
        print("------------------------------------------------")
    return doping

def dopingCondition(mu,p_target,bandIterable):
    print("mu = " + "{0:.3f}".format(mu))
    for band in bandIterable:
        band["mu"] = mu
    return doping(bandIterable) - p_target

def set_mu_to_doping(bandIterable, p_target, ptol=0.001):
    print("Computing mu for hole doping = " + "{0:.3f}".format(p_target))
    mu = optimize.brentq(dopingCondition, -10, 10, args=(p_target ,bandIterable), xtol=ptol)
    for band in bandIterable:
        band["mu"] = mu
