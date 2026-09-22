# Here lies statmom and cubecoef functions
#!/usr/bin/env python3
'''
Processes a cube of ring images (cubecoef.pro) and computes
the statistical moments (statmom.pro) of the angular coefficients.


Auth: A. Tokovinin
Translated: D. Hurtado

'''
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sci
from astropy.io import fits
from tqdm.notebook import tqdm
import argparse



def main():

    p = argparse.ArgumentParser()

    p.add_argument('cubefile', type=str,
                   help='FITS file with the cube of ring images, e.g. test.fits')
    p.add_argument('--mmax', dest='mmax', type=int, default=20,
                   help='Max order of angular signals, def=20')
    p.add_argument('--nsect', dest='nsect', type=int, default=8,
                   help='Number of sectors for radius calculation, def=8')
    p.add_argument('--drad', dest='drad', type=float, default=1.5,
                   help='Mask width in lambda/(0.5*D*(1-eps)) units, def=1.5')
    p.add_argument('--interpol', dest='interpol', type=int, default=1,
                   help='Use sub-pixel (Fourier) shifts for centering, def=1')
    p.add_argument('--wavelen', dest='wavelen', type=float, default=0.6e-6,
                   help='Wavelength of reference light, in meters, def=0.6e-6')
    p.add_argument('--diam', dest='d', type=float, default=0.304,
                   help='Aperture diameter, meters, def=0.304')
    p.add_argument('--eps', dest='eps', type=float, default=0.7,
                   help='Central obstruction, def=0.7')
    p.add_argument('--pixel', dest='pixel', type=float, default=0.017664,
                   help='Grid sampling, m per grid pixel, def=0.017664')
    p.add_argument('--nstart', dest='nstart', type=int, default=50,
                   help='Number of initial frames for the ring template, def=50')
    p.add_argument('--leak', dest='leak', type=float, default=1,
                   help='Leaky-integrator gain for the running centroid, def=1')
    p.add_argument('--debug', dest='debug_str', type=str, default='DEBUG',
                   help='Debug level, logger accepted values, def=INFO')

    args = p.parse_args()

    cubefile = args.cubefile
    mmax = args.mmax
    nsect = args.nsect
    drad = args.drad
    interpol = args.interpol
    wavelen = args.wavelen
    d = args.d
    eps = args.eps
    pixel = args.pixel
    nstart = args.nstart
    leak = args.leak
    debug_str = args.debug_str

if __name__ == '__main__':
    main()
