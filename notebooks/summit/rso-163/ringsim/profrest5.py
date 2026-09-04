#Notes:
    #Changed output from prof1 (line-of-sight) to prof (altitude), also scaled prof since only prof1 was scaled for the output
    #Fixed jwind and therefore v2, before was using scaled prof to calc but now uses normal (prof is scaled so is easy to read when output)

import numpy as np
from scipy import optimize
import json
import codecs
import sys


# read a json file, return a dictionary; returns None if fails
def read_json(filename):
    try:
        file = open(filename, "r")
        p = json.load(file)
        file.close()
        return p
    except FileNotFoundError as err:
        print(err)
        return


# Profile restoration. Inputs: dictionaries of parameters, data, weights, and Z-matrix
# Output: dictionary of profile parameters
# Before calling Restore, run getzen.py to define the zenith distance and star color in data
#
def Restore(par, data, weight, zmat):
    #    print(data["image"]["impar"])
    var = data["moments"]["var"]  # variance of a-coefficients
    cov = data["moments"]["cov"]  # covariance of a-coefficients
    impar = data["image"]["impar"]  # ring parameters
    noisepar = data["image"]["noisepar"]  # noise parameters
    starpar = data["starpar"]  # star parameters
    zen = starpar["zen"]
    bv = starpar["BV"]
    z0 = par["profrest"]["zgrid"]  # grid of heights representing the turbulence profile

    z = np.array(weight["z"])  # nominal distance grid of weights
    nz = len(weight["z"])  # number of layers in weights
    nm = len(weight["wt0"][0])  # number of coefficints per layer
    wt = np.reshape(np.array(weight["wt0"]), (nz, nm))  # wt.shape = (16,21)
    wt += bv * np.reshape(np.array(weight["wtslope"]), (nz, nm))  # must be >0!

    # interpolate weight to the Z0 grid, trim to mmax
    cosz = np.cos(zen / 180 * np.pi)  # cos(z)
    mmax = 15  # hard-coded number of terms to use
    nz0 = len(z0)
    z00 = np.array(z0) / cosz  # stretched array of nominal heights
    wt1 = np.zeros((nz0, mmax))  # interpolated and trimmed weights
    for m in range(0, mmax):
        wt1[:, m] = np.interp(z00, z, wt[:, 1 + m])
    wtsect = np.interp(z00, z, wt[:, 0])  # sector-motion weight on z0 grid

    # Check that moments and weights have the same number of coefficients
    mcoef = len(var)  # 21 for mmax=20
    if mcoef != nm:
        print("Error! Mismatching number of coefficients: ", mcoef, nm)
        return

    # noise bias, see allcubes5.pro => noisecubes
    gain = data["cubepar"]["gain"]  # electrons per ADU
    #eladu = 3.60 * pow(10, -gain / 200)
    eladu = 0.3 #Our hardcoded for gain of 200
    noisepar = data["image"]["noisepar"]  # list of 4 numbers
    fluxadu = float(data["image"]["impar"]["flux"])
    #    print(fluxadu)
    flux = eladu * fluxadu  # flux in electrons

    anoise = float(noisepar[0]) / flux + float(noisepar[1]) * pow(par["telescope"]["ron"] / flux,
                                                                  2)  # noise variance of a-coef
    rnoise = 2. * (float(noisepar[2]) / flux + float(noisepar[3]) * pow(par["telescope"]["ron"] / flux,
                                                                        2) / 8)  # noise on radius, pix^2

    #    anoise = noisepar[0]/flux + noisepar[1]*pow(par["telescope"]["ron"]/flux,2) # noise variance of a-coef
    #    rnoise = 2.*(noisepar[2]/flux + noisepar[3]*pow(par["telescope"]["ron"]/flux,2)/8) # noise on radius, pix^2

    # Select max frequency used in the restoration
    var1 = np.array(var[1:mmax + 1], float) - anoise
    var1 *= (var1 > 0)  # prevent negative (sets negative values to zero)
    cov1 = np.array(cov[1:mmax + 1], float)
    rho = cov1 / var1
    varcorr = var1 / (0.8 + 0.2 * rho)  # correction for finite exposure time
    totvar = np.sum(np.array(var, float))  # full scintillation power

    # Z-matrix correction. Avoid negative values!
    z1 = np.array(zmat)
    ncol = z1.shape[1]
    var2 = np.array(var[1:ncol + 1], float)  # indices used for correction
    varz = varcorr / (1. + np.dot(z1[0:mmax, :], var2))  # correct for saturation

    # weighted nnls. Weight is proportional to 1/var (before noise subtraction, non-negative)
    varwt = np.power(np.array(var[1:mmax + 1], float), -1)
    a2 = np.transpose(wt1.copy())  # matrix of (mmax-1,nz) dimension
    for i in range(0, mmax):
        a2[i, :] *= varwt[i]
    varz2 = varz * varwt
    prof, resvar = optimize.nnls(a2, varz2)  # turbulence intergals in  [m^1/3] in z0 layers

    varmod = np.dot(a2, prof) / varwt
    erms = np.std(1. - varmod / varz)
    #print("RMS residual: {:.3f}".format(erms))

    prof1 = prof * cosz  # zenith-corrected integrals
    jtot = np.sum(prof1)  # turbulence integral
    seeconst = 6.83e-13  # integral for 1" seeing at 500nm
    see = pow(jtot / seeconst, 0.6)  # seeing in arcseconds
    jfree = np.sum(prof1[2:nz0])  # start at 0.5km
    fsee = pow(jfree / seeconst, 0.6)  # FA seeing in arcseconds

    # Compute seeing from radius var.
    d = par["telescope"]["D"]
    pixel = par["telescope"]["pixel"]
    lameff = weight["lameff"]
    lam0 = lameff[0] + bv * (lameff[1] - lameff[0])  # effective wavelength for the star color
    rvar = float(data["moments"]["rvar"]) - rnoise  # radius variance in pix^2, noise-subtracted
    lamd = lam0 / d * 206265.  # lambda/D in arcseconds
    rvarnorm = rvar * pow(pixel / lamd, 2)  # variance in (lam/D^2) units

    wcoef = np.sum(wtsect * prof) / np.sum(prof)  # profile-adjusted weight of sector variance
    jtot2 = rvarnorm / wcoef / 4  # turbulence intergal, m^1/3. Explain factor 4!
    see2 = pow(jtot2 * cosz / seeconst, 0.6)  # seeing at zenith, arcsec
    see2 *= 1. / (1. - 0.4 * totvar)  # saturation correction
    #print("Seeing (sect,tot,FA): {:.3f} {:.3f} {:.3f}".format(see2, see, fsee))

    # Wind measurement
    texp = 1e-3  # hard-coded exposure time
    delta = (var1 - cov1) * (texp ** (-2))
    delta = delta * (delta > 0)
    ucoef = np.array(weight["ucoef0"]) + bv * np.array(weight["ucoefslope"])
    umm = weight["umm"]
    delta = np.array(delta[umm])
    v2mom = np.sum(ucoef * delta)
    jwind = np.sum(prof[1:nz0])  # exclude ground layer, the speed is not zen-corrected
    v2 = pow(v2mom / jwind, 0.5)  # use profile uncorrected for zenith
    #print("Wind speed [m/s]: {:.3f}".format(v2))

    # tau_0 at 500nm
    r0 = pow(6.680e13 * jwind, -0.6)
    tau0 = 310. * r0 / v2  # in [ms]
    # isoplanatic angle at 500nm at zenith
    r0 = 0.101 / see
    tmp = pow(np.array(z0), 1.6667)
    heff = pow(np.sum(tmp * prof1) / np.sum(prof1), 0.6)
    theta0 = 205265. * 0.31 * r0 / heff
    # Output dictionary
    profile = {"z0": z0, "prof": (prof * 1e13).tolist(), "see": see, "fsee": fsee, "see2": see2, "wind": v2, "erms": erms,
               "totvar": totvar, "tau0": tau0, "theta0": theta0}

    return profile


# ### Main module. usage: > python <par> <data>
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python profrest5.py <par-file> <data-file>")
        sys.exit()

    parfile = sys.argv[1]
    # print("Parameter file: " + parfile)
    datafile = sys.argv[2]
    # Ingest all information needed
    par = read_json(parfile)
    if par == None:
        sys.exit()
    weight = read_json(par["profrest"]["weightfile"])
    zmat = read_json(par["profrest"]["zmat"])
    data = read_json(datafile)
    if data == None:
        sys.exit()

    profile = Restore(par, data, weight, zmat)
    if profile == None:
        print("Restoration failed!")
        sys.exit()

    data["profile"] = profile  # add to the data dictionary
    json.dump(data, codecs.open(datafile, 'w', encoding='utf-8'), separators=(',', ':'))
    print("Profile is saved in " + datafile)

    # Output profile
    prof = profile["prof"]  # integrals in 1e-13 m^1/4
    z0 = profile["z0"]  # heights in m
    s = s1 = ""  # print profile in readable form, in 10^(-13) units
    for x in prof:
        s += " {:.2f}".format(x)
    for x in z0:
        s1 += " {:.2f}".format(x * 1e-3)
    print(s1)
    print(s)
