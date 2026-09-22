#!/usr/bin/env python3
'''
Simulates ring images and creates fits file
Includes ringsim.pro and writefits.pro


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
from tqdm.notebook import tqdm
import argparse



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
    
    #Is there a way to make this easier?
    pixel = args.pixel
    r0 = args.r0
    wavelen = args.wavelen
    ngrid = args.ngrid
    zlow = args.zlow
    zhigh = args.zhigh
    fhigh = args.fhigh
    debug_str = args.debug_str
    size = 2 * ngrid * pixel