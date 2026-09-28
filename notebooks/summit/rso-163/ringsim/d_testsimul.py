# Full simulation
#!/usr/bin/env python3
'''
Top-level driver: reads sim1.par-style inputs, then runs the full
chain (simatm -> ringsim -> cubemom -> weight -> profrest) and
prints the summary report (testsimul.pro).


Auth: A. Tokovinin
Translated: D. Hurtado

'''
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

    # --- Telescope / detector (sim1.par) ---
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

    # --- Atmosphere / observation (settings cell) ---
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
    p.add_argument('--debug', dest='debug_str', type=str, default='DEBUG',
                   help='Debug level, logger accepted values, def=INFO')

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
    debug_str = args.debug_str

    # Derived from inputs
    pixscale = pixsize / effl * 206265
    r0 = 0.98 * wavelen / seeing * 206265.0

if __name__ == '__main__':
    main()
