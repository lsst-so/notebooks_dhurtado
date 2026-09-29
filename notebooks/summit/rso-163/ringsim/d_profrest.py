# Profile restoration
#!/usr/bin/env python3
'''
Restores the turbulence profile from the angular moments,
weights, and saturation matrix (profrest5.pro).

Currently not compatible with CLI due to Dict inputs

Auth: A. Tokovinin
Translated: D. Hurtado

'''
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import optimize



def main(par=None, data=None, weight=None,
         zmatfile=os.path.join('andrei', 'zmat.json'),
         zen=0, bv=0, gain=0, display=True, verb=True):

    # Import path: pass par/data/weight dicts (from statmom + computeweight)
    # straight in. CLI path: they are None, so read them from JSON files.
    # (CLI not wired yet -- dict positionals can't come from the command line;
    #  add --datafile/--weightfile/--parfile loaders here when needed.)
    if par is None or data is None or weight is None:
        
        p = argparse.ArgumentParser()
        
        p.add_argument('--datafile', dest='datafile', type=str, default=None,
                       help='JSON with image moments and noise parameters')
        p.add_argument('--weightfile', dest='weightfile', type=str, default=None,
                       help='Weight JSON produced by d_weights.py')
        p.add_argument('--parfile', dest='parfile', type=str, default=None,
                       help='JSON with telescope/profrest parameters')
        p.add_argument('--zmatfile', dest='zmatfile', type=str,
                       default=os.path.join('andrei', 'zmat.json'),
                       help='Saturation (Z-matrix) file, def=andrei/zmat.json')
        p.add_argument('--zen', dest='zen', type=float, default=0,
                       help='Zenith distance of the star, degrees, def=0')
        p.add_argument('--bv', dest='bv', type=float, default=0,
                       help='Star color B-V (0=blue), def=0')
        p.add_argument('--gain', dest='gain', type=float, default=0,
                       help='Detector gain setting, def=0')
        p.add_argument('--display', dest='display', type=bool, default=True,
                       help='Decides if images are displayed, boolean, def=True')
        p.add_argument('--verbose', dest='verb', type=bool, default=True,
                       help='Print text into CLI, def=True')

        args = p.parse_args()

        data = read_json(args.datafile)
        weight = read_json(args.weightfile)
        par = read_json(args.parfile)
        zmatfile = args.zmatfile
        zen = args.zen
        bv = args.bv
        gain = args.gain
        display = args.display
        verb = args.verb

    zmat = read_json(zmatfile)   # (20,5) saturation matrix, La Serena

    # Heights [m] for the restored profile (same as testsimul.pro z0 / par-laserena zgrid)
    par['profrest']['zgrid'] = [0, 250, 500, 1000, 2000, 4000, 8000, 16000]

    # Observing geometry of the simulated star
    data['starpar'] = {'zen': zen, 'BV': bv}    # simulated at zenith
    data['cubepar'] = {'gain': gain}            # detector gain from the sim1.par cell

    profile = restore(par, data, weight, zmat, display, verb)

    prof = np.array(profile['prof'], float) / 1e13
    wind = float(profile['wind'])
    erms = float(profile['erms'])
    if verb: print('Profile restored!')

    return profile


def read_json(filename): # read a json file, return a dictionary; returns None if fails
    try:
        file = open(filename, 'r')
        p = json.load(file)
        file.close()
        return p
    except FileNotFoundError as err:
        print(err)
        return


# Profile restoration. Inputs: dictionaries of parameters, data, weights, and Z-matrix
# Output: dictionary of profile parameters
# Before calling Restore, run getzen.py to define the zenith distance and star color in data
def restore(par, data, weight, zmat, display=True, verb=True):
    
    if verb: print(data['image']['impar'])
    
    var =      data['moments']['var']  # variance of a-coefficients
    cov =      data['moments']['cov']  # covariance of a-coefficients
    impar =    data['image']['impar']  # ring parameters
    noisepar = data['image']['noisepar']  # noise parameters
    starpar =  data['starpar']  # star parameters
    zen =      starpar['zen']
    bv =       starpar['BV']
    z0 =       par['profrest']['zgrid']  # grid of heights representing the turbulence profile
    
    z = np.array(weight['z'])   # nominal distance grid of weights
    nz = len(weight['z'])       # number of layers in weights
    nm = len(weight['wt0'][0])  # number of coefficints per layer
    wt = np.reshape(np.array(weight['wt0']), (nz, nm))  # wt.shape = (16,21)
    wt += bv * np.reshape(np.array(weight['wtslope']), (nz, nm))  # must be >0!
    
    # interpolate weight to the Z0 grid, trim to mmax
    cosz = np.cos(zen / 180 * np.pi)  # cos(z)
    mmax = 20  # hard-coded number of terms to use
    nz0 = len(z0)
    z00 = np.array(z0) / cosz  # stretched array of nominal heights
    wt1 = np.zeros((nz0, mmax))  # interpolated and trimmed weights
    for m in range(0, mmax):
        wt1[:, m] = np.interp(z00, z, wt[:, 1 + m])
    wtsect = np.interp(z00, z, wt[:, 0])  # sector-motion weight on z0 grid
    
    # Check that moments and weights have the same number of coefficients
    mcoef = len(var)  # 21 for mmax=20
    if mcoef != nm:
        raise ValueError('Error! Mismatching number of coefficients: ', mcoef, nm)
    
    # noise bias, see allcubes5.pro => noisecubes
    gain = data['cubepar']['gain']  # electrons per ADU
    #eladu = 3.60 * pow(10, -gain / 200)
    eladu = 1 # sim counts photo-electrons directly (gain=0 in sim1.par => ADU == electrons); was 0.3 for a real gain=200 camera
    noisepar = data['image']['noisepar']  # list of 4 numbers
    fluxadu = float(data['image']['impar']['flux'])
    if verb: print(f'Fluxadu: {fluxadu}')
    flux = eladu * fluxadu  # flux in electrons
    
    anoise = float(noisepar[0]) / flux + float(noisepar[1]) * pow(par['telescope']['ron'] / flux, 2)  # noise variance of a-coef
    rnoise = 2 * (float(noisepar[2]) / flux + float(noisepar[3]) * pow(par['telescope']['ron'] / flux, 2) / 8)  # noise on radius, pix^2

    #    anoise = noisepar[0]/flux + noisepar[1]*pow(par['telescope']['ron']/flux,2) # noise variance of a-coef
    #    rnoise = 2.*(noisepar[2]/flux + noisepar[3]*pow(par['telescope']['ron']/flux,2)/8) # noise on radius, pix^2
    
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
    if verb: print(f'RMS residual: {erms:.3f}')
    
    if display:
        arg = np.arange(mmax) + 1                         # findgen(mmax)+1
        plt.errorbar(arg, varz, yerr=anoise, fmt='s', capsize=3,
                     label='powcorr')                     # plot, /ylog, psym=6 (squares)
        plt.semilogy(arg, varmod, 'k--', label='powmod')  # oplot, li=2 (dashed)
        plt.yscale('log')                                 # errorbar does not set log y itself
        plt.xlabel('m', fontsize=15)                      # !p.charsize = 1.5
        plt.ylabel('Power', fontsize=15)
        plt.xlim(arg[0], arg[-1])                         # xs=1 (exact x-range)
        plt.ylim(1e-4,1e-2)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('images', 'power_spectrum.jpg'), dpi=300, format='jpg')
        #plt.show()
        plt.close()  # release the figure so repeated runs don't accumulate figures

    prof1 = prof * cosz  # zenith-corrected integrals
    jtot = np.sum(prof1)  # turbulence integral
    seeconst = 6.83e-13  # integral for 1' seeing at 500nm
    see = pow(jtot / seeconst, 0.6)  # seeing in arcseconds
    jfree = np.sum(prof1[2:nz0])  # start at 0.5km
    fsee = pow(jfree / seeconst, 0.6)  # FA seeing in arcseconds
    
    # Compute seeing from radius var.
    d = par['telescope']['D']
    pixel = par['telescope']['pixel']
    lameff = weight['lameff']
    lam0 = lameff[0] + bv * (lameff[1] - lameff[0])  # effective wavelength for the star color
    rvar = float(data['moments']['rvar']) - rnoise  # radius variance in pix^2, noise-subtracted
    lamd = lam0 / d * 206265  # lambda/D in arcseconds
    rvarnorm = rvar * pow(pixel / lamd, 2)  # variance in (lam/D^2) units
    
    wcoef = np.sum(wtsect * prof) / np.sum(prof)  # profile-adjusted weight of sector variance
    jtot2 = rvarnorm / wcoef / 4  # turbulence intergal, m^1/3. Explain factor 4!
    see2 = pow(jtot2 * cosz / seeconst, 0.6)  # seeing at zenith, arcsec
    see2 *= 1 / (1 - 0.4 * totvar)  # saturation correction
    if verb: print(f'Seeing (sect,tot,FA): {see2:.3f} {see:.3f} {fsee:.3f}')
    
    # Wind measurement
    texp = 1e-3  # hard-coded exposure time
    delta = (var1 - cov1) * (texp ** (-2))
    delta = delta * (delta > 0)
    ucoef = np.array(weight['ucoef0']) + bv * np.array(weight['ucoefslope'])
    umm = weight['umm']
    delta = np.array(delta[umm])
    v2mom = np.sum(ucoef * delta)
    jwind = np.sum(prof[1:nz0])  # exclude ground layer, the speed is not zen-corrected
    v2 = pow(v2mom / jwind, 0.5)  # use profile uncorrected for zenith
    if verb: print(f'Wind speed [m/s]: {v2:.3f}')
    
    # tau_0 at 500nm
    r0 = pow(6.680e13 * jwind, -0.6)
    tau0 = 310 * r0 / v2   # in [ms]
    r0 = 0.101 / see       # isoplanatic angle at 500nm at zenith
    tmp = pow(np.array(z0), 1.6667)
    heff = pow(np.sum(tmp * prof1) / np.sum(prof1), 0.6)
    theta0 = 206265 * 0.31 * r0 / heff
    
    # Output dictionary
    profile = {
        'z0':     z0, 
        'prof':   (prof * 1e13).tolist(), 
        'see':    see, 
        'fsee':   fsee, 
        'see2':   see2, 
        'wind':   v2, 
        'erms':   erms,
        'totvar': totvar, 
        'tau0':   tau0, 
        'theta0': theta0
              }

    return profile


if __name__ == '__main__':
    main()