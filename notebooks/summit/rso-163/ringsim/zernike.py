#Notes:
    #Added int() to wf calculations, as floats such as 1.0 don't work

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math

def zernike_rad(m, n, rho):

    if (np.mod(n-m, 2) == 1):
        return rho*0.0
    wf = rho*0.0
    for k in range((n-m)//2+1):
        wf += rho**(n-2.0*k) * (-1.0)**k * math.factorial(int(n-k)) / ( math.factorial(int(k)) * math.factorial( int((n+m)/2.0 - k) ) * math.factorial( int((n-m)/2.0 - k )) )
    return wf

def zernike(m, n, rho, phi, norm=True):
    nc = 1.0
    if (norm):
        nc = (2*(n+1)/(1+(m==0)))**0.5
    if (m > 0): return nc*zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0): return nc*zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return nc*zernike_rad(0, n, rho)

def noll_to_zern(j):
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n

    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))
    return (n, m)

def zernikel(j, rho, phi, norm=True):
    n, m = noll_to_zern(j)
    return zernike(m, n, rho, phi, norm)

def zernikeg(j, size=256, norm=True):
    n, m = noll_to_zern(j)
#
    grid = (np.indices((size, size), dtype=np.float) - 0.5*size) / (0.5*size)
    grid_rad = (grid[0]**2. + grid[1]**2.)**0.5
    grid_ang = np.arctan2(grid[0], grid[1])
    return zernike(m, n, grid_rad, grid_ang, norm)



