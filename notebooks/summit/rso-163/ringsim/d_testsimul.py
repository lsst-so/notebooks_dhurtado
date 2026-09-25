# Full simulation
#!/usr/bin/env python3
'''
Top-level driver: reads sim1.par-style inputs, then runs the full
chain (simatm -> ringsim -> cubemom -> weight -> profrest) and
prints the summary report (testsimul.pro).


Auth: A. Tokovinin
Translated: D. Hurtado

'''
import argparse
import matplotlib.pyplot as plt
import numpy as np

from d_simatm import simatm
from d_ringsim import ringsim
from d_cubecoef import cubecoef
from d_statmom import statmom
from d_weights import computeweight



def main():

    p = argparse.ArgumentParser()

    # Telescope / detector (sim1.par)
    p.add_argument('--diam', dest='d', type=float, default=0.304,
                   help='Aperture diameter, meters, def=0.304')
    p.add_argument('--effl', dest='effl', type=float, default=1.20845,
                   help='Effective focal length, meters, def=1.20845')
    p.add_argument('--eps', dest='eps', type=float, default=0.7,
                   help='Central obstruction, def=0.7')
    p.add_argument('--pixel', dest='pixel', type=float, default=0.017664,
                   help='Grid sampling, m per grid pixel, def=0.017664')
    p.add_argument('--pixsize', dest='pixsize', type=float, default=6.9e-6,
                   help='Physical detector pixel size, meters, def=6.9e-6')
    p.add_argument('--wavelen', dest='wavelen', type=float, default=0.6e-6,
                   help='Wavelength of reference light, meters, def=0.6e-6')
    p.add_argument('--pdist', dest='pdist', type=float, default=1050,
                   help='Defocus expressed as conjugation height, m, def=1050')
    p.add_argument('--drad', dest='drad', type=float, default=1.5,
                   help='Mask width in lambda/(0.5*D*(1-eps)) units, def=1.5')
    p.add_argument('--mmax', dest='mmax', type=int, default=20,
                   help='Max order of angular signals, def=20')
    p.add_argument('--nsect', dest='nsect', type=int, default=8,
                   help='Number of sectors for radius calculation, def=8')
    p.add_argument('--interpol', dest='interpol', type=int, default=1,
                   help='Use sub-pixel shifts, def=1')
    p.add_argument('--ron', dest='ron', type=float, default=0,
                   help='Readout noise, electrons, def=0')

    # Atmosphere / observation
    p.add_argument('--seeing', dest='seeing', type=float, default=1,
                   help='Seeing in arcsec at 0.5 micron, def=1')
    p.add_argument('--zlow', dest='zlow', type=float, default=500,
                   help='Low layer altitude, meters, def=500')
    p.add_argument('--zhigh', dest='zhigh', type=float, default=10500,
                   help='High layer altitude, meters, def=10500')
    p.add_argument('--highfrac', dest='highfrac', type=float, default=0.1,
                   help='Fraction of turbulence in the high layer, def=0.1')
    p.add_argument('--starmag', dest='starmag', type=float, default=2,
                   help='Star magnitude, def=2')
    p.add_argument('--gain', dest='gain', type=float, default=0,
                   help='Camera gain setting, def=0')
    p.add_argument('--seed0', dest='seed0', type=int, default=None,
                   help='Fixed RNG seed for reproducible runs, def=None (random)')
    p.add_argument('--ngrid', dest='ngrid', type=int, default=512,
                   help='Half size of the atmosphere grid to simulate, def=512')

    args = p.parse_args()

    d = args.d
    effl = args.effl
    eps = args.eps
    pixel = args.pixel
    pixsize = args.pixsize
    wavelen = args.wavelen
    pdist = args.pdist
    drad = args.drad
    mmax = args.mmax
    nsect = args.nsect
    interpol = args.interpol
    ron = args.ron
    seeing = args.seeing
    zlow = args.zlow
    zhigh = args.zhigh
    highfrac = args.highfrac
    starmag = args.starmag
    gain = args.gain
    seed0 = args.seed0
    ngrid = args.ngrid

    # Derived from inputs
    pixscale = pixsize / effl * 206265
    r0 = 0.98 * wavelen / seeing * 206265.0

    display = True

    simatm(pixel, r0, wavelen=wavelen, ngrid=ngrid, zlow=zlow, zhigh=zhigh,
           fhigh=highfrac, seed0=seed0)

    cubefile = ringsim(d, effl, eps, pdist, pixsize, ron=ron, gain=gain,
                       starmag=starmag, display=display)

    impar, coef = cubecoef(cubefile, mmax=mmax, nsect=nsect, drad=drad,
                           interpol=interpol, display=display)

    par, data = statmom(impar, coef, mmax=mmax, nsect=nsect, display=display)

    weight = computeweight(par)

    moments = data['moments']
    noisepar = impar['noisepar']

    # Noise on the angular and radial coefficients
    flux = impar['flux']                                   # star flux, ADU per frame
    anoise = noisepar[0] / flux + noisepar[1] * (ron / flux) ** 2  # noise variance of a-coef
    flux1 = flux / nsect                                   # flux per sector
    rnoise1 = 2 * (noisepar[2] / flux1 + noisepar[3] * (ron / flux1) ** 2 / nsect)  # radius noise, pix^2

    # IDL-named weight arrays pulled out of the getweight5 'weight' dict
    z = np.array(weight['z'])              # nominal weight-distance grid [m] (len 16)
    wt0 = np.array(weight['wt0']).T        # transpose to (m, nz) to match IDL wt0[m, z]
    ucoef0 = np.array(weight['ucoef0'])
    mm = weight['umm']

    # Angular power spectrum and covariance from statmom
    powspec = np.array(moments['var'], float)
    covspec = np.array(moments['cov'], float)

    # Standard altitude layers (0, 0.25 km, 0.5 km, 1 km ... 16 km)
    nz = 8
    z0 = np.concatenate(([0], 1000 * (np.power(2, (np.arange(nz - 1) - 2)))))  # 2: IDL '2.' float base allows negative exponents

    # Noise-subtract the power and get the scintillation index (testsimul.pro lines 66-67)
    powspec = np.maximum(powspec - anoise, 0)
    totvar = float(np.sum(powspec))

    # Interpolate the weighting functions onto the z0 altitude grid
    m = wt0.shape[0]                       # number of m-terms (mmax+1)
    wt = np.zeros((m, nz), dtype=np.float64)
    for i in range(m):
        wt[i, :] = np.interp(z0, z, wt0[i, :])

    # Radius rms from statmom (used as radvar = rrms**2 - rnoise1 in the report cell)
    rrms = np.sqrt(moments['rvar'])

    # Plot angular power spectrum, covariance and noise floor
    if display:
        arg = np.arange(len(powspec))
        plt.figure('Angular power / covariance / noise', figsize=(7, 5))
        plt.semilogy(arg, np.maximum(np.array(moments['var'], float), 1e-12), 'k-o', label='power')
        plt.semilogy(arg, np.maximum(covspec, 1e-12), 'r--', label='covariance')
        plt.axhline(max(anoise, 1e-12), ls=':', color='b', label='noise')
        plt.ylim(10e-6)
        plt.xlim(0,20)
        plt.xticks(np.arange(0, 21, step=1))
        plt.xlabel('m')
        plt.ylabel('Power')
        plt.title('Angular power spectrum')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
