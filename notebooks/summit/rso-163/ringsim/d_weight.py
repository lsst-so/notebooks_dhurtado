# Here lies awegiths and getweights5 functions
#!/usr/bin/env python3
'''
Computes the angular/radial response weights (aweight.pro) and
assembles the weight file for the two star colors (getweight5.pro).


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

    p.add_argument('ringradpix', type=float,
                   help='Measured ring radius in CCD pixels (from the cube)')
    p.add_argument('--diam', dest='d', type=float, default=0.304,
                   help='Aperture diameter, meters, def=0.304')
    p.add_argument('--eps', dest='eps', type=float, default=0.7,
                   help='Central obstruction, def=0.7')
    p.add_argument('--pdist', dest='pdist', type=float, default=1050,
                   help='Defocus expressed as conjugation height, m, def=1050')
    p.add_argument('--pixscale', dest='pixscale', type=float, default=1.17773,
                   help='Detector plate scale, arcsec per pixel, def=1.17773')
    p.add_argument('--mmax', dest='mmax', type=int, default=20,
                   help='Max order of angular signals, def=20')
    p.add_argument('--wav', dest='wav', type=float, nargs='+', default=[600],
                   help='Wavelength grid, nm, def=[600]')
    p.add_argument('--sp', dest='sp', type=float, nargs='+', default=[1.0],
                   help='Spectral response over the wavelength grid, def=[1.0]')
    p.add_argument('--drho', dest='drho', type=float, default=1.5,
                   help='Ring mask half-width in lambda/D units, def=1.5')
    p.add_argument('--weightfile', dest='weightfile', type=str, default='weights.json',
                   help='Output file for the serialized weights, def=weights.json')
    p.add_argument('--debug', dest='debug_str', type=str, default='DEBUG',
                   help='Debug level, logger accepted values, def=INFO')

    args = p.parse_args()

    ringradpix = args.ringradpix
    d = args.d
    eps = args.eps
    pdist = args.pdist
    pixscale = args.pixscale
    mmax = args.mmax
    wav = np.array(args.wav)
    sp = np.array(args.sp)
    drho = args.drho
    weightfile = args.weightfile
    debug_str = args.debug_str

if __name__ == '__main__':
    main()
