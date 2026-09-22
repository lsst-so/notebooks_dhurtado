# Profile restoration
#!/usr/bin/env python3
'''
Restores the turbulence profile from the angular moments,
weights, and saturation matrix (profrest5.pro).


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

    p.add_argument('datafile', type=str,
                   help='JSON/dict with image moments and noise parameters')
    p.add_argument('weightfile', type=str,
                   help='Weight file produced by d_weight.py, e.g. weights.json')
    p.add_argument('--zmatfile', dest='zmatfile', type=str, default=os.path.join('andrei', 'zmat.json'),
                   help='Saturation (Z-matrix) file, def=andrei/zmat.json')
    p.add_argument('--zgrid', dest='zgrid', type=float, nargs='+',
                   default=[0, 250, 500, 1000, 2000, 4000, 8000, 16000],
                   help='Heights [m] of the restored profile layers')
    p.add_argument('--diam', dest='d', type=float, default=0.304,
                   help='Aperture diameter, meters, def=0.304')
    p.add_argument('--pixel', dest='pixel', type=float, default=1.17773,
                   help='Detector plate scale, arcsec per pixel, def=1.17773')
    p.add_argument('--ron', dest='ron', type=float, default=0,
                   help='Readout noise, electrons, def=0')
    p.add_argument('--mmax', dest='mmax', type=int, default=20,
                   help='Number of angular terms used in restoration, def=20')
    p.add_argument('--zen', dest='zen', type=float, default=0,
                   help='Zenith distance of the star, degrees, def=0')
    p.add_argument('--bv', dest='bv', type=float, default=0,
                   help='Star color B-V (0=blue), def=0')
    p.add_argument('--gain', dest='gain', type=float, default=0,
                   help='Detector gain setting, def=0')
    p.add_argument('--texp', dest='texp', type=float, default=1e-3,
                   help='Exposure time, seconds, def=1e-3')
    p.add_argument('--debug', dest='debug_str', type=str, default='DEBUG',
                   help='Debug level, logger accepted values, def=INFO')

    args = p.parse_args()
    
    datafile = args.datafile
    weightfile = args.weightfile
    zmatfile = args.zmatfile
    zgrid = args.zgrid
    d = args.d
    pixel = args.pixel
    ron = args.ron
    mmax = args.mmax
    zen = args.zen
    bv = args.bv
    gain = args.gain
    texp = args.texp
    debug_str = args.debug_str

if __name__ == '__main__':
    main()
