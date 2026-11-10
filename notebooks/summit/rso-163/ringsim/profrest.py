#!/usr/bin/env python3
'''
Restores the turbulence profile from the angular moments,
weights, and saturation matrix (profrest5.pro).

Usage:
    python profrest.py --help         # list all arguments and defaults
    python profrest.py --datafile data.json --weightfile weights.json --parfile par.json
    import profrest; profile = profrest.main(par, data, weight)   # dicts from statmom/weights
    help(profrest.restore)            # docstring for any function

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
         zmatfile=os.path.join('.', 'zmat.json'),
         zen=0, bv=0, gain=0, texp=1e-3 ,display=True, verb=True):
    '''
    Entry point for the profile restoration.
    
    If the par/data/weight dicts are passed in (import path), uses
    them directly; otherwise reads them from JSON files given on the
    CLI. Sets the observing geometry, then calls restore().
    
    Args:
        par: Telescope/profrest parameter dict (from statmom); None
            to read from --parfile.
        data: Image/moments data dict (from statmom); None to read
            from --datafile.
        weight: Weight dict (from weights.py); None to read from
            --weightfile.
        zmatfile: Saturation Z-matrix JSON path ('./zmat.json').
        zen: Zenith distance of the star, degrees (0).
        bv: Star color B-V, 0=blue (0).
        gain: Detector gain setting (0).
        display: Save the diagnostic plot (True).
        verb: Print text to the CLI (True).
    
    Returns:
        The restored profile dict (see restore()).
    '''
    if par is None or data is None or weight is None:
        
        p = argparse.ArgumentParser()
        
        p.add_argument('--datafile', dest='datafile', type=str, default=None,
                       help='JSON with image moments and noise parameters')
        p.add_argument('--weightfile', dest='weightfile', type=str, default=None,
                       help='Weight JSON produced by weights.py')
        p.add_argument('--parfile', dest='parfile', type=str, default=None,
                       help='JSON with telescope/profrest parameters')
        p.add_argument('--zmatfile', dest='zmatfile', type=str,
                       default=os.path.join('.', 'zmat.json'),
                       help='Saturation (Z-matrix) file, def=./zmat.json')
        p.add_argument('--zen', dest='zen', type=float, default=0,
                       help='Zenith distance of the star, degrees, def=0')
        p.add_argument('--bv', dest='bv', type=float, default=0,
                       help='Star color B-V (0=blue), def=0')
        p.add_argument('--gain', dest='gain', type=float, default=0,
                       help='Detector gain setting, def=0')
        p.add_argument('--texp', dest='texp', type=float, default=1e-3,
                       help='Exposure time, used for windspeed, def=1e-3')
        p.add_argument('--display', dest='display', type=bool, default=True,
                       help='Decides if images are displayed, boolean, def=True')
        p.add_argument('--verbose', dest='verb', type=bool, default=True,
                       help='Print text into CLI, def=True')
        
        args = p.parse_args()
        
        data     = read_json(args.datafile)
        weight   = read_json(args.weightfile)
        par      = read_json(args.parfile)
        zmatfile = args.zmatfile
        zen      = args.zen
        bv       = args.bv
        gain     = args.gain
        texp     = args.texp
        display  = args.display
        verb     = args.verb
    
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


def restore(par, data, weight, zmat, display=True, verb=True):
    '''
    Main workhorse of the restoration, out from the
    cube coefficients and weights (calculated earlier or not)
    finds the atmospheric turbulence profile. Takes the
    parameters, data, weights and layers and derives
    wind, seeing and turbulence profile.
    
    Args:
        par: Telescope/profrest parameter dict (holds zgrid, ron).
        data: Image/moments data dict (moments, impar, noisepar,
            starpar, cubepar).
        weight: Weight dict (z, wt0, wtslope, ucoef0, lameff, ...).
        zmat: Saturation Z-matrix (array).
        display: Save the power-spectrum plot (True).
        verb: Print text to the CLI (True).
    
    Returns:
        A dict with the turbulence profile and derived quantities:
        z0, prof, see (total), fsee (free-atm), see2 (sector), wind,
        erms, totvar, tau0, theta0.
    '''
    if verb: print(data['image']['impar'])
    
    var =      data['moments']['var']  # variance of a-coefficients
    cov =      data['moments']['cov']  # covariance of a-coefficients
    impar =    data['image']['impar']  # ring parameters
    noisepar = data['image']['noisepar']  # noise parameters
    starpar =  data['starpar']  # star parameters
    zen =      starpar['zen']
    bv =       starpar['BV']
    z0 =       par['profrest']['zgrid']  # grid of heights representing the turbulence profile
    texp =     par['telescope']['texp']  # exposure time [s], from the FITS header
    
    cosz = np.cos(zen / 180 * np.pi)  # cos(z)
    mmax = 20  # hard-coded number of terms to use, from original code
    seeconst = 6.83e-13  # integral for 1' seeing at 500nm
    
    # Prepare weights and interpolate to the Z0 grid, trim to mmax
    z, nm, wt = prepare_weights(weight, bv)
    wt1, wtsect, nz0 = interpolate(z, wt, z0, cosz, mmax)
    
    # Check that moments and weights have the same number of coefficients
    mcoef = len(var)  # 21 for mmax=20
    if mcoef != nm:
        raise ValueError('Error! Mismatching number of coefficients: ', mcoef, nm)
    
    # Noise bias and power correction
    anoise, rnoise = noise_bias(data, par, verb)
    varz, totvar, var1, cov1 = check_power(var, cov, zmat, anoise, mmax)
    
    # Weighted NNLS inversion
    prof, erms, varmod = solve_profile(wt1, varz, var, mmax, verb)
    
    if display:
        varr = np.array(var, float)
        covr = np.array(cov, float)
        mfull = np.arange(len(varr))                      # m = 0..mcoef-1 (raw moments)
        marg = np.arange(mmax) + 1                        # m = 1..mmax (corrected/modeled)
        plt.figure(figsize=(8, 5))
        plt.semilogy(mfull, np.maximum(varr, 1e-12), color='k', marker='o', ls='-',
                     label='Power (var)')
        plt.semilogy(mfull, np.maximum(covr, 1e-12), color='xkcd:fire engine red', marker='v', ls='-',
                     label='Covariance (cov)')
        plt.axhline(max(anoise, 1e-12), color='xkcd:cobalt blue', ls=':', label='noise (anoise)')
        plt.errorbar(marg, varz, yerr=anoise, fmt='s', capsize=3, color='xkcd:neon purple',
                     label='Powcorr (varz)')          # noise/saturation-corrected power
        plt.semilogy(marg, varmod, color='xkcd:neon purple', ls='--', label='Powmod (varmod)')  # model fit
        plt.yscale('log')                                 # errorbar does not set log y itself
        plt.xlabel('m', fontsize=13)
        plt.ylabel('Power', fontsize=13)
        plt.xlim(0, mmax)
        plt.xticks(np.arange(0, mmax + 1))
        plt.ylim(1e-6, 1e-2)
        plt.title('Angular spectrum: moments, noise and restoration fit')
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.savefig(os.path.join('images', 'angular_power_spectrum.jpg'), dpi=300, format='jpg')
        #plt.show()
        plt.close()  # release the figure so repeated runs don't accumulate figures
    
    # Seeing from the scintillation profile (total + free atmosphere)
    scint_see, free_see, jtot, prof1 = see_profile(prof, cosz, nz0, seeconst)
    
    # Seeing from the radius variance (sector motion)
    sector_see = see_radius(data, par, weight, prof, wtsect, cosz, totvar, rnoise, bv, seeconst)
    if verb: print(f'Seeing (sect,tot,FA): {sector_see:.3f} {scint_see:.3f} {free_see:.3f}')
    
    # Wind measurement
    v2, jwind = wind_speed(var1, cov1, weight, prof, bv, nz0, texp, verb)
    
    # Coherence time and isoplanatic angle
    tau0, theta0 = coherence(jwind, v2, scint_see, z0, prof1)
    
    # Output dictionary
    profile = {
        'z0':     z0, 
        'prof':   (prof * 1e13).tolist(), 
        'see':    scint_see, 
        'fsee':   free_see, 
        'see2':   sector_see, 
        'wind':   v2, 
        'erms':   erms,
        'totvar': totvar, 
        'tau0':   tau0, 
        'theta0': theta0
              }
    
    return profile

# Used functions ---------------------------------------------------------------------------

def read_json(filename):
    '''
    Takes a .json file with data and returns the content.
    
    Args:
        filename: Path to the JSON file to read.
    
    Returns:
        The parsed object, or None if the file is not found.
    '''
    try:
        file = open(filename, 'r')
        p = json.load(file)
        file.close()
        return p
    except FileNotFoundError as err:
        print(err)
        return


def prepare_weights(weight, bv):
    '''
    Takes the calculated weights for each
    layer and prepares them for processing.
    
    Reshapes the weight table to (nz, nm) 
    and applies the star-color (B-V) correction.
    
    Args:
        weight: Weight dict holding 'z', 'wt0', 'wtslope'.
        bv: Star color B-V (0=blue).
    
    Returns:
        (z, nm, wt): the distance grid, number of coefficients per
        layer, and the color-adjusted (nz, nm) weight table.
    '''
    z = np.array(weight['z'])   # nominal distance grid of weights
    nz = len(weight['z'])       # number of layers in weights
    nm = len(weight['wt0'][0])  # number of coefficints per layer
    wt = np.reshape(np.array(weight['wt0']), (nz, nm))  # wt.shape = (16,21)
    wt += bv * np.reshape(np.array(weight['wtslope']), (nz, nm))  # must be >0!
    return z, nm, wt


def interpolate(z, wt, z0, cosz, mmax):
    '''
    Interpolates the weights onto the restoration altitude grid.
    
    Stretches the z0 grid by 1/cos(zen) for the line of sight, then
    interpolates each of the mmax angular weights and the sector
    weight from the nominal grid z onto it.
    
    Args:
        z: Nominal weight-distance grid.
        wt: Color-adjusted (nz, nm) weight table.
        z0: Restoration altitude grid, meters.
        cosz: Cosine of the zenith distance.
        mmax: Number of angular terms to keep.
    
    Returns:
        (wt1, wtsect, nz0): the (nz0, mmax) interpolated angular
        weights, the interpolated sector weight, and len(z0).
    '''
    nz0 = len(z0)
    z00 = np.array(z0) / cosz  # stretched array of nominal heights
    wt1 = np.zeros((nz0, mmax))  # interpolated and trimmed weights
    for m in range(0, mmax):
        wt1[:, m] = np.interp(z00, z, wt[:, 1 + m])
    wtsect = np.interp(z00, z, wt[:, 0])  # sector-motion weight on z0 grid
    return wt1, wtsect, nz0


def noise_bias(data, par, verb=True):
    '''
    Computes measured flux to e- and combines it with noise 
    parameters and read-out noise into the angular (anoise) 
    and radial (rnoise) noise variances.
    
    Args:
        data: Data dict (holds cubepar.gain, image.noisepar,
            image.impar.flux).
        par: Parameter dict (holds telescope.ron).
        verb: Print text to the CLI (True).
    
    Returns:
        (anoise, rnoise): angular-coefficient noise variance and
        radius noise variance (pix^2).
    '''
    gain = data['cubepar']['gain']  # electrons per ADU
    #eladu = 3.60 * pow(10, -gain / 200)
    eladu = 1 # sim counts photo-electrons directly (gain=0 in sim1.par => ADU == electrons); was 0.3 for a real gain=200 camera
    noisepar = data['image']['noisepar']  # list of 4 numbers
    fluxadu = float(data['image']['impar']['flux'])
    if verb: print(f'Fluxadu: {fluxadu}')
    flux = eladu * fluxadu  # flux in electrons
    
    anoise = float(noisepar[0]) / flux + float(noisepar[1]) * pow(par['telescope']['ron'] / flux, 2)  # noise variance of a-coef
    rnoise = 2 * (float(noisepar[2]) / flux + float(noisepar[3]) * pow(par['telescope']['ron'] / flux, 2) / 8)  # noise on radius, pix^2
    return anoise, rnoise


def check_power(var, cov, zmat, anoise, mmax):
    '''
    Subtracts the noise bias from the variance,
    applies the finite-exposure-time correction using the
    variance/covariance ratio, and then the matrix saturation
    correction, giving the corrected power spectrum variables.
    
    Args:
        var: Variance of the a-coefficients (per m).
        cov: 1-frame-lag covariance of the a-coefficients (per m).
        zmat: Saturation Z-matrix (array).
        anoise: Angular-coefficient noise variance (from noise_bias).
        mmax: Max angular order used in the restoration.
    
    Returns:
        (varz, totvar, var1, cov1): the saturation-corrected power,
        the full scintillation power (sum of var), and the
        noise-subtracted var1 and raw cov1 arrays (reused for wind).
    '''
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
    return varz, totvar, var1, cov1


def solve_profile(wt1, varz, var, mmax, verb=True):
    '''
    With the weights and variables in each M layer
    resolves the non-negative least-squares system
    to recover the layer turbulence integrals.
    
    Args:
        wt1: (nz0, mmax) interpolated weight table (from interpolate).
        varz: Saturation-corrected power (from check_power).
        var: Raw variance of the a-coefficients (for the weighting).
        mmax: Max angular order used.
        verb: Print text to the CLI (True).
    
    Returns:
        (prof, erms, varmod): the per-layer turbulence integrals
        [m^(1/3)], the RMS fit residual, and the modeled power.
    '''
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
    return prof, erms, varmod


def seeing(j, seeconst):
    '''
    Evaluates the seeing with the turbulence integral.
    
    Args:
        j: Turbulence integral J, m^(1/3).
        seeconst: Integral giving 1" seeing at 500 nm.
    
    Returns:
        Seeing in arcseconds.
    '''
    return pow(j / seeconst, 0.6)


def seeing_sector(jtot2, cosz, seeconst, totvar):
    '''
    Evaluates sector (DIMM-like) seeing in particular.
    
    Args:
        jtot2: Sector-motion turbulence integral, m^(1/3).
        cosz: Cosine of the zenith distance.
        seeconst: Integral giving 1" seeing at 500 nm.
        totvar: Full scintillation power (for the saturation term).
    
    Returns:
        Sector seeing in arcseconds.
    '''
    sector_see = pow(jtot2 * cosz / seeconst, 0.6)  # seeing at zenith, arcsec
    sector_see *= 1 / (1 - 0.4 * totvar)  # saturation correction
    return sector_see


def see_profile(prof, cosz, nz0, seeconst):
    '''
    Evaluates the profile-dependent zenith-corrected 
    seeing, total (all-layer) and free-atmosphere (>0.5 km).
    
    Args:
        prof: Per-layer turbulence integrals (from solve_profile).
        cosz: Cosine of the zenith distance.
        nz0: Number of layers in the restoration grid.
        seeconst: Integral giving 1" seeing at 500 nm.
    
    Returns:
        (scint_see, free_see, jtot, prof1): total seeing, free-atm
        seeing, total turbulence integral, and the zenith-corrected
        profile.
    '''
    prof1 = prof * cosz  # zenith-corrected integrals
    jtot = np.sum(prof1)  # turbulence integral
    jfree = np.sum(prof1[2:nz0])  # start at 0.5km
    scint_see = seeing(jtot, seeconst)
    free_see = seeing(jfree, seeconst)
    return scint_see, free_see, jtot, prof1


def see_radius(data, par, weight, prof, wtsect, cosz, totvar, rnoise, bv, seeconst):
    '''
    Noise-subtracts and normalizes the radius variance, weights it by
    the profile-adjusted sector weight, and converts to seeing via
    seeing_sector.
    
    Args:
        data: Data dict (holds moments.rvar).
        par: Parameter dict (holds telescope.D, telescope.pixel).
        weight: Weight dict (holds lameff).
        prof: Per-layer turbulence integrals.
        wtsect: Interpolated sector weight (from interpolate).
        cosz: Cosine of the zenith distance.
        totvar: Full scintillation power (for the saturation term).
        rnoise: Radius noise variance (from noise_bias).
        bv: Star color B-V.
        seeconst: Integral giving 1" seeing at 500 nm.
    
    Returns:
        Sector seeing in arcseconds.
    '''
    d = par['telescope']['D']
    pixel = par['telescope']['pixel']
    lameff = weight['lameff']
    lam0 = lameff[0] + bv * (lameff[1] - lameff[0])  # effective wavelength for the star color
    rvar = float(data['moments']['rvar']) - rnoise  # radius variance in pix^2, noise-subtracted
    lamd = lam0 / d * 206265  # lambda/D in arcseconds
    rvarnorm = rvar * pow(pixel / lamd, 2)  # variance in (lam/D^2) units
    
    wcoef = np.sum(wtsect * prof) / np.sum(prof)  # profile-adjusted weight of sector variance
    jtot2 = rvarnorm / wcoef / 4  # turbulence intergal, m^1/3. Explain factor 4!
    return seeing_sector(jtot2, cosz, seeconst, totvar)


def wind_speed(var1, cov1, weight, prof, bv, nz0, texp=1e-3, verb=True):
    '''
    Calcualtes wind speeds out from the exposure time and sector movement
    
    Args:
        var1: Noise-subtracted variance per m (from check_power).
        cov1: 1-frame-lag covariance per m (from check_power).
        weight: Weight dict (holds ucoef0, ucoefslope, umm).
        prof: Per-layer turbulence integrals.
        bv: Star color B-V.
        nz0: Number of layers in the restoration grid.
        verb: Print text to the CLI (True).
    
    Returns:
        (v2, jwind): the turbulence-weighted wind speed [m/s] and the
        wind turbulence integral (ground layer excluded).
    '''
    delta = (var1 - cov1) * (texp ** (-2))
    delta = delta * (delta > 0)
    ucoef = np.array(weight['ucoef0']) + bv * np.array(weight['ucoefslope'])
    umm = weight['umm']
    delta = np.array(delta[umm])
    v2mom = np.sum(ucoef * delta)
    jwind = np.sum(prof[1:nz0])  # exclude ground layer, the speed is not zen-corrected
    v2 = pow(v2mom / jwind, 0.5)  # use profile uncorrected for zenith
    if verb: print(f'Wind speed [m/s]: {v2:.3f}')
    return v2, jwind


def coherence(jwind, v2, see, z0, prof1):
    '''
    Calculates the coherence time and the isoplanatic angle.
    
    The coherence time (where the atmosphere stays static)
    comes from the wind turbulence integral and wind speed. 
    Isoplanatic angle theta0 comes from the total seeing and 
    the effective turbulence height.
    
    Args:
        jwind: Wind turbulence integral (ground layer excluded).
        v2: Turbulence-weighted wind speed, m/s.
        see: Total seeing, arcsec (feeds the isoplanatic angle).
        z0: Restoration altitude grid, meters.
        prof1: Zenith-corrected per-layer turbulence integrals.
    
    Returns:
        (tau0, theta0): coherence time [ms] and isoplanatic angle.
    '''
    # tau_0 at 500nm
    r0 = pow(6.680e13 * jwind, -0.6)
    tau0 = 310 * r0 / v2   # in [ms]
    r0 = 0.101 / see       # isoplanatic angle at 500nm at zenith
    tmp = pow(np.array(z0), 1.6667)
    heff = pow(np.sum(tmp * prof1) / np.sum(prof1), 0.6)
    theta0 = 206265 * 0.31 * r0 / heff
    return tau0, theta0


if __name__ == '__main__':
    main()