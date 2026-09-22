#!/usr/bin/env python3
'''
Creates the atmosphere for simulation
of ring images.


Auth: A. Tokovinin
Translated: D. Hurtado

'''
import argparse
import codecs
import json
import logging
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import numpy as np
import os
from scipy import optimize
import sys
from tqdm.notebook import tqdm

from astropy.io import fits
import datetime
from IPython.display import display, clear_output
from scipy.signal import detrend, find_peaks
from scipy.ndimage import zoom, map_coordinates, shift as ndshift

import zernike



def main():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('pixel', type=float,
                   help='TBD')
    p.add_argument('r0', type=float,
                   help='Fried parameter from seeing')
    p.add_argument('--wavelen', dest='wavelen', type=float, default=0.6e-6,
                   help='Wavelength of reference light, in meters, def=0.6e-6')
    p.add_argument('--ngrid', dest='ngrid', type=float, default=1024,
                   help='Half size of the atmosphere grid to simulate, def=1024')
    p.add_argument('--zlow', dest='zlow', type=float, default=500,
                   help='Low layer altitude, def=500')
    p.add_argument('--zhigh', dest='zhigh', type=float, default=10500,
                   help='High layer altitude, def=10500')
    p.add_argument('--fhigh', dest='fhigh', type=float, default=0.1,
                   help='Fraction of high layer (?), def=0.1')
    p.add_argument('--debug', dest='debug_str', type=str, default='DEBUG',
                   help='Debug level, logger accepted values, def=INFO')
    
    args = p.parse_args()
    
    pixel = args.pixel
    r0 = args.r0
    wavelen = args.wavelen
    ngrid = args.ngrid
    zlow = args.zlow
    zhigh = args.zhigh
    fhigh = args.fhigh
    debug_str = args.debug_str
    size = 2 * ngrid * pixel
    
    print('Simulating atmosphere')
    
    logger.debug(f'size: {size} \n ngrid: {ngrid}')
    
    tint0 = (r0 ** (-5 / 3)) / 0.423 * ((0.5 * wavelen / np.pi) ** 2) # Turbulence integral in meters^1/3
    see = (tint0 / 6.83e-13) ** (0.6)
    tinthigh = tint0 * highfrac            # High layer integral
    tintlow = tint0 * (1 - highfrac)       # Low layer integral
    if zlow >= zhigh:
        raise ValueError(f"Inconsistency: zlow ({zlow}) can't be greater or equal than zhigh ({zhigh})")
    
    r0high = (0.423 * (0.5 * wavelen / np.pi)**(-2) * tinthigh)**(-3 / 5)  # Fried param for high layer
    r0low = (0.423 * (0.5 * wavelen / np.pi)**(-2) * tintlow)**(-3 / 5)    # Fried param for low layer
    
    print(f'Grid size (meters): {size} \n Fried parameters (meters) [low, high]: [{r0low, r0high}]')
    print(f'Integrals (meters^1/3) [low, high]: [{tintlow, tinthigh}] \n Altitudes (meters) {zlow, zhigh}')
    print(f'Seeing (arcsec): {see}')
    
    # For phase simulations
    if highfrac != 0:
        facthigh = np.sqrt(0.023) * ((size/r0high) ** (5/6))
    else:
        facthigh = 0
    factlow = np.sqrt(0.023) * ((size/r0low) ** (5/6))
    
    # Create grid and radial distance from center in pixels
    y, x = np.ogrid[:ngrid * 2, :ngrid * 2]
    r = np.sqrt((x - ngrid)**2 + (y - ngrid)**2, dtype=np.float64)
    r[ngrid, ngrid] = 1e-3
    
    # Create Fresnel filters
    farg = np.pi * wavelen / (size ** 2) * r ** 2
    
    # Set seed for rng, if seed0 is provided, use it. Otherwise, let the OS provide a random one
    if 'seed0' in globals() and seed0 is not None:
        rng = np.random.default_rng(seed0)
        logger.debug(f'Simulation on fixed seed: {seed0}.')
    else:
        rng = np.random.default_rng()
        logger.debug('Simulation on random seed.')
    
    # Simulate turbulence in high layer
    if highfrac > 0:
        print('Simulating high layer')
        rng_complex = rng.normal(size=(ngrid*2, ngrid*2)) +1j * rng.normal(size=(ngrid*2, ngrid*2))
        
        tmp_fourier = facthigh * (r ** (-11 / 6)) * rng_complex
        tmp_fourier[ngrid, ngrid] = 0 + 0j
        tmp_shifted = np.fft.ifftshift(tmp_fourier)
        tmp_spatial = np.fft.ifft2(tmp_shifted) * ((ngrid*2) ** 2)
        tmp_phase = np.fft.fftshift(tmp_spatial).real
    
        # Simulate phase
        u1 = np.exp(1j * tmp_phase)
        
        # Propagate using Angular spectrum method
        if (zhigh - zlow) > 0:
            print('Propagating to low layer')
            dz = zhigh - zlow
            H_transfer = np.exp(-1j * farg * dz)
            u1_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
            tmp_prop = H_transfer * u1_fourier
            u1 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(tmp_prop)))
    else:
        u1 = np.ones((ngrid*2, ngrid*2), dtype=np.complex128)
    
    # Simulate Low layer
    print('Simulating low layer')
    rng_complex_low = rng.normal(size=(ngrid*2, ngrid*2)) + 1j * rng.normal(size=(ngrid*2, ngrid*2))
    
    tmp_fourier_low = factlow * (r ** (-11 / 6)) * rng_complex_low
    tmp_fourier_low[ngrid, ngrid] = 0 + 0j
    
    tmp_shifted_low = np.fft.ifftshift(tmp_fourier_low)
    tmp_spatial_low = np.fft.ifft2(tmp_shifted_low) * ((ngrid*2) ** 2)
    tmp_phase_low = np.fft.fftshift(tmp_spatial_low).real
    
    # Apply low layer phase distortion
    u1 *= np.exp(1j * tmp_phase_low)
    
    # Propagate to ground
    print('Propagating to ground')
    H_ground = np.exp(-1j * farg * zlow)
    u1_fourier_ground = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
    tmp_ground = H_ground * u1_fourier_ground
    u1 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(tmp_ground)))
    
    np.savez(
        'atm.npz',
        u1=u1,
        ngrid=ngrid,
        pixel=pixel,
        wavelen=wavelen,
        see=see,
        r0=r0,
        highfrac=highfrac,
        zhigh=zhigh,
        zlow=zlow
        )
    
    # Diagnostics
    scint = np.sum((np.abs(u1)**2 - 1)**2) / (2 * ngrid) ** 2
    rytov = 19.22 * (wavelen ** (-7 / 6)) * ((zlow ** (5 / 6)) * tintlow + (zhigh**(5 / 6)) * tinthigh)
    intensity = np.abs(u1) ** 2
    
    print(f'Rytov variance, scintillation: {rytov}, {scint}')
    
    plt.figure(figsize=(7, 7))
    plt.imshow(intensity, cmap='GnBu', origin='lower', norm=colors.LogNorm())
    plt.title('Simulated atmosphere')
    plt.tight_layout()
    plt.savefig('atmsim.jpg', dpi=300, format='jpg')
    #plt.show()

if __name__ == '__main__':
    main()


# ============================================================================
# TODO / review notes (from ruff + comparison with the testsimul notebook)
# 'import argparse' has been added above. The rest below still needs fixing:
#
# --- BLOCKERS (script NameErrors without these) ---
# * `highfrac` is undefined (used at lines ~74,75,87,110,163). The argparse
#   defines `--fhigh` -> dest='fhigh' and unpacks `fhigh = args.fhigh`, but the
#   body uses `highfrac`. Fix: rename dest to 'highfrac' (--highfrac) and
#   `highfrac = args.highfrac`, OR rename every body use to `fhigh`. The
#   notebook uses `highfrac`, so that name is preferred.
# * `logger` is undefined (lines ~70,104,107). The notebook defines
#   `logger = logging.getLogger('testsimul')` in a separate cell; this
#   standalone script never creates it. Fix: add
#   `logger = logging.getLogger('simatm'); logger.setLevel(debug_str)`
#   after unpacking args (this also uses debug_str, see below).
# * `seed0` is undefined (lines ~102,103,104). The `if 'seed0' in globals()`
#   guard was written for the notebook's global namespace; inside main() it is
#   a local that does not exist. Fix: add
#   `p.add_argument('--seed0', dest='seed0', type=int, default=None, ...)`
#   then simplify the guard to `if seed0 is not None:`.
#
# --- UNUSED ASSIGNMENTS (dead inputs) ---
# * `fhigh` assigned but never used (line ~64) -- same as the highfrac issue.
# * `debug_str` assigned but never used (line ~65) -- meant to set the logger
#   level: `logger.setLevel(debug_str)` (ties into the logger fix above).
#
# --- UNUSED IMPORTS (ruff F401, 20 total, auto-fixable) ---
#   Only numpy, matplotlib.pyplot, matplotlib.colors, and argparse are used.
#   Unused: codecs, json, math, sys, matplotlib(as mpl), matplotlib.animation,
#   os, scipy.optimize, tqdm, datetime, IPython.display (display, clear_output),
#   scipy.signal (detrend, find_peaks), scipy.ndimage (zoom, map_coordinates,
#   shift), zernike. Run `ruff check d_simatm.py --fix` to strip them (do this
#   AFTER confirming argparse is actually used, so it is not removed too).
#
# --- NOTEBOOK DISCREPANCY ---
# * `--ngrid` default is 1024 here vs 512 in the notebook, and type=float.
#   ngrid is used as an array index (r[ngrid, ngrid], slice bounds), so a float
#   raises IndexError at line ~96. Fix: change `--ngrid` to type=int.
# ============================================================================