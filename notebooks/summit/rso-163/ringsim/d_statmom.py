# Here lies the statmom function
#!/usr/bin/env python3
'''
Computes the statistical moments (statmom.pro) of the per-frame
angular coefficients produced by cubecoef: variances, covariances,
and mean aberration coefficients.


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

    p.add_argument('coeffile', type=str,
                   help='File with the per-frame coefficient array from cubecoef')
    p.add_argument('--mmax', dest='mmax', type=int, default=20,
                   help='Max order of angular signals, def=20')
    p.add_argument('--nsect', dest='nsect', type=int, default=8,
                   help='Number of sectors for radius calculation, def=8')
    p.add_argument('--display', dest='display', type=bool, default=True,
                   help='Decides if images are displayed, boolean, def=True')

    args = p.parse_args()

    coeffile = args.coeffile
    mmax = args.mmax
    nsect = args.nsect
    display = args.display

    return statmom(coeffile, mmax, nsect, display)


def statmom(coeffile, mmax=20, nsect=8, display=True):

    pass  # TODO: statmom.pro body -> build and return the moments dict


if __name__ == '__main__':
    main()
