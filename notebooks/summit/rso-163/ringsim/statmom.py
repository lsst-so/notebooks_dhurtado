#!/usr/bin/env python3
'''
Computes the statistical moments (statmom.pro) of the per-frame
angular coefficients produced by cubecoef: variances, covariances,
and mean aberration coefficients.

Currently not compatible with CLI due to array input.

Auth: A. Tokovinin
Translated: D. Hurtado

'''
import argparse
import logging  # debugging not supported
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    '''
    CLI/entry wrapper that parses arguments and calls statmom().
    
    Note: impar and coef are a dict and an array passed in memory,
    so this is really driven by import, not the command line (see
    the module docstring).
    '''
    p = argparse.ArgumentParser()

    p.add_argument('impar', type=dict,
                   help='Image parameters')
    p.add_argument('coef', type=np.array,
                   help='Coefficient array from cubecoef')
    p.add_argument('--mmax', dest='mmax', type=int, default=20,
                   help='Max order of angular signals, def=20')
    p.add_argument('--nsect', dest='nsect', type=int, default=8,
                   help='Number of sectors for radius calculation, def=8')
    p.add_argument('--display', dest='display', type=bool, default=True,
                   help='Decides if images are displayed, boolean, def=True')
    p.add_argument('--verbose', dest='verb', type=bool, default=True,
                   help='Print text into CLI, def=True')
    
    args = p.parse_args()
    
    impar = args.impar
    coef = args.coef
    mmax = args.mmax
    nsect = args.nsect
    display = args.display
    verb = args.verb
    
    return statmom(impar, coef, mmax, nsect, display, verb)


def statmom(impar, coef, mmax=20, nsect=8, display=True, verb=True):
    '''
    Computes the statistical moments of the per-frame coefficients.
    
    From the cubecoef output, builds the differential radius
    variance (DIMM-like), the angular power spectrum and covariance,
    and the mean aberration coefficients, then assembles the par and
    data dicts that weights.py and profrest.py consume.
    
    Args:
        impar: Ring-parameter dict (from cubecoef).
        coef: (ncoef, nframes) per-frame coefficient array.
        mmax: Max order of angular signals (20).
        nsect: Number of sectors for the radius calc (8).
        display: Save the angular-spectrum plot (True).
        verb: Print text to the CLI (True).
    
    Returns:
        (par, data): par is the telescope/profrest parameter dict;
        data holds the image and moments dicts.
    '''
    # Calculation of statistical moments
    ncoef, nz = coef.shape # Check shape
    expected_ncoef = 2 * nsect + 2 * mmax + 2
    if verb: print(f'ncoef {ncoef} expect: {expected_ncoef}')
    if ncoef != expected_ncoef:
        raise ValueError(f'Parameters do not match (ncoef={expected_ncoef}, got {ncoef})')
    
    # Differential radius variance and radius noise (DIMM-like signals)
    meanrvar, drnoise = radius_moments(coef, nsect)
    
    # Variance and covariance of angular coefficients
    power, cov, meancoef = angular_moments(coef, nsect, mmax, nz)
    
    # Mean coefficients for aberrations (mean radii and m=1,2,3 cos/sine terms)
    mcoef = aberration_coef(coef, meancoef, nsect, mmax)
    
    if display:
        save_image(power, cov, mmax)
    
    # Assemble output dictionaries (same structure as cube2.py)
    moments = {
        'var': power.tolist(),
        'cov': cov.tolist(),
        'rnoise': float(drnoise[0]),
        'rvar': meanrvar,
        'mcoef': mcoef.tolist()
                }
    
    # Assemble <par> for what weights.py expects.
    par = {
        'telescope': {
            'D': impar['d'],
            'eps': impar['eps'],
            'pdist': impar['pdist'],
            'pixel': impar['asperpix'],     # detector plate scale [arcsec/pix]
            'ringradpix': impar['rad'],     # ring radius MEASURED from this cube
            'ron': impar['ron'],
            'texp': impar['texp'],
                    },
        'profrest': {
            'mmax': mmax,
            'wavelen': [impar['wavelen']],
            'sp':  [1.0],                   # flat single-wavelength response
            'weightfile': 'weights.json',
            'nsect': nsect
                    }
            }
    
    # Assemble <data> for profrest.py profile restoration.
    data = {
        'image': {
            'impar': impar, 
            'noisepar': impar['noisepar']},
        'moments': moments,
            }
    
    if verb: print('Par file created with',
          'D:', par['telescope']['D'], 'eps:', par['telescope']['eps'],
          'pdist:', par['telescope']['pdist'],
          '\npixel (asperpix):', par['telescope']['pixel'],
          'ringradpix:', par['telescope']['ringradpix'], 'ron:', par['telescope']['ron'])
    
    return par, data


# Used functions ---------------------------------------------------------------------------

def radius_moments(coef, nsect):
    '''
    Computes the differential radius variance and its noise.
    
    Args:
        coef: (ncoef, nframes) per-frame coefficient array.
        nsect: Number of angular sectors.
    
    Returns:
        (meanrvar, drnoise): mean differential radius variance
        (pix^2) and the per-pair noise variances (pix^2).
    '''
    # Differential radius variance across opposite sector pairs (DIMM-like signals)
    half_sect = nsect // 2
    dr = coef[:half_sect, :] + coef[half_sect:nsect, :]  # (half_sect, nz)
    drvar = np.var(dr, axis=1)                           # variances in pix^2
    meanrvar = float(np.mean(drvar))
    
    # Radius noise from difference of successive dr values
    ddr = dr - np.roll(dr, 1, axis=1)
    drnoise = np.var(ddr, axis=1)                        # noise variances, pix^2
    return meanrvar, drnoise


def angular_moments(coef, nsect, mmax, nz):
    '''
    Computes the angular power spectrum and covariance.
    
    Takes the variance and 1-frame-lag covariance of the angular
    coefficients, then folds the cosine (0..m) and sine (m+1..2m+1)
    terms together per mode m.
    
    Args:
        coef: (ncoef, nframes) per-frame coefficient array.
        nsect: Number of angular sectors (offset into coef).
        mmax: Max angular order.
        nz: Number of frames.
    
    Returns:
        (power, cov, meancoef): per-m power spectrum, per-m
        covariance, and the mean of each angular coefficient.
    '''
    acoef = coef[2 * nsect :, :]                         # (2m+2, nz)
    varcoef = np.var(acoef, axis=1)
    meancoef = np.mean(acoef, axis=1)
    tmp = (acoef * np.roll(acoef, 1, axis=1))[:, 1:]     # shift-1 product, drop first
    covar = np.sum(tmp, axis=1) / (nz - 1) - meancoef ** 2
    
    # Add cosine (0..m) and sine (m+1..2m+1) variances and covariances
    power = varcoef[: mmax + 1] + varcoef[mmax + 1 : 2 * mmax + 2]
    cov = covar[: mmax+ 1] + covar[mmax + 1 : 2 * mmax + 2]
    return power, cov, meancoef


def aberration_coef(coef, meancoef, nsect, mmax):
    '''
    Assembles the mean aberration coefficients.
    
    Args:
        coef: (ncoef, nframes) per-frame coefficient array.
        meancoef: Mean of each angular coefficient (from
            angular_moments).
        nsect: Number of angular sectors.
        mmax: Max angular order (locates the sine terms).
    
    Returns:
        A length nsect+6 array of mean aberration coefficients.
    '''
    mcoef = np.zeros(nsect + 6, dtype=np.float64)
    mcoef[:nsect] = np.mean(coef[:nsect, :], axis=1)
    mcoef[nsect : nsect + 3] = meancoef[1:4]
    mcoef[nsect + 3 : nsect + 6] = meancoef[mmax + 2 : mmax + 5]
    return mcoef


def save_image(power, cov, mmax):
    '''
    Saves the angular mode spectrum plot into the images folder.
    
    Args:
        power: Per-m angular power spectrum.
        cov: Per-m covariance.
        mmax: Max angular order (sets the x-axis range).
    '''
    arg = np.arange(mmax + 1)
    
    plt.figure(figsize=(7, 5))
    plt.semilogy(arg, power, 'k-o', label='Power Spectrum')
    plt.semilogy(arg, cov, 'r--', label='Covariance')
    plt.xlabel('m')
    plt.ylabel('Power')
    plt.title('Angular Mode Spectrum')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('images', 'angular_spectrum.jpg'), dpi=300, format='jpg')
    #plt.show()
    plt.close()


if __name__ == '__main__':
    main()
