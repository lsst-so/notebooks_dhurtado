# Here lies the statmom function
#!/usr/bin/env python3
'''
Computes the statistical moments (statmom.pro) of the per-frame
angular coefficients produced by cubecoef: variances, covariances,
and mean aberration coefficients.


Auth: A. Tokovinin
Translated: D. Hurtado

'''
import argparse
import logging  # debugging not supported
import matplotlib.pyplot as plt
import numpy as np



def main():
    
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
    
    args = p.parse_args()
    
    impar = args.impar
    coef = args.coef
    mmax = args.mmax
    nsect = args.nsect
    display = args.display
    
    return statmom(impar, coef, mmax, nsect, display)


def statmom(impar, coef, mmax=20, nsect=8, display=True):
    
    # Calculation of statistical moments
    m = mmax
    
    ncoef, nz = coef.shape
    expected_ncoef = 2 * nsect + 2 * m + 2
    print(f'ncoef {ncoef} expect: {expected_ncoef}')
    
    if ncoef != expected_ncoef:
        raise ValueError(f'Parameters do not match (ncoef={expected_ncoef}, got {ncoef})')
    
    # Differential radius variance across opposite sector pairs (DIMM-like signals)
    half_sect = nsect // 2
    dr = coef[:half_sect, :] + coef[half_sect:nsect, :]  # (half_sect, nz)
    drvar = np.var(dr, axis=1)                           # variances in pix^2
    meanrvar = float(np.mean(drvar))
    
    # Radius noise from difference of successive dr values
    ddr = dr - np.roll(dr, 1, axis=1)
    drnoise = np.var(ddr, axis=1)                        # noise variances, pix^2
    
    # Variance and covariance of angular coefficients
    acoef = coef[2 * nsect :, :]                         # (2m+2, nz)
    varcoef = np.var(acoef, axis=1)
    meancoef = np.mean(acoef, axis=1)
    tmp = (acoef * np.roll(acoef, 1, axis=1))[:, 1:]     # shift-1 product, drop first
    covar = np.sum(tmp, axis=1) / (nz - 1) - meancoef ** 2
    
    # Add cosine (0..m) and sine (m+1..2m+1) variances and covariances
    power = varcoef[: m + 1] + varcoef[m + 1 : 2 * m + 2]
    cov = covar[: m + 1] + covar[m + 1 : 2 * m + 2]
    
    # Mean coefficients for aberrations (mean radii and m=1,2,3 cos/sine terms)
    mcoef = np.zeros(nsect + 6, dtype=np.float64)
    mcoef[:nsect] = np.mean(coef[:nsect, :], axis=1)
    mcoef[nsect : nsect + 3] = meancoef[1:4]
    mcoef[nsect + 3 : nsect + 6] = meancoef[m + 2 : m + 5]
    
    if display:
        arg = np.arange(m + 1)
    
        plt.figure('Angular Power Spectrum', figsize=(7, 5))
        plt.semilogy(arg, power, 'k-o', label='Power Spectrum')
        plt.semilogy(arg, cov, 'r--', label='Covariance')
        plt.xlabel('m')
        plt.ylabel('Power')
        plt.title('Angular Mode Spectrum')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.clf
    
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
                    },
        'profrest': {
            'mmax': mmax,
            'wavelen': [impar['wavelen']],
            'sp':  [1.0],                   # flat single-wavelength response
            'weightfile': 'weights.json',
                    },
            }

    # Assemble <data> for profrest.py profile restoration.
    data = {
        'image': {
            'impar': impar, 
            'noisepar': impar['noisepar']},
        'moments': moments,
            }

    print('Par file created with',
          'D:', par['telescope']['D'], 'eps:', par['telescope']['eps'],
          'pdist:', par['telescope']['pdist'],
          '\npixel (asperpix):', par['telescope']['pixel'],
          'ringradpix:', par['telescope']['ringradpix'], 'ron:', par['telescope']['ron'])

    return par, data

if __name__ == '__main__':
    main()
