#!/usr/bin/env python
# coding: utf-8

'''
Zernike aberrations evaluation script

Author:     Andrei Tokovinin
Adapted:    Adam Dobbyn
Docstrings: Diego Hurtado
'''


import numpy as np
import math

def zernike_rad(m, n, rho):
    '''
    Evaluates the radial Zernike polynomial R_n^m(rho).
    
    Returns zeros when (n - m) is odd (the polynomial is undefined /
    identically zero there).
    
    Args:
        m: Azimuthal order (non-negative here), integer.
        n: Radial order, integer (n >= m).
        rho: Radial coordinate(s), normalized to 1 at the edge.
    
    Returns:
        The radial polynomial evaluated at rho (same shape as rho).
    '''
    if (np.mod(n-m, 2) == 1):
        return rho*0.0
    wf = rho*0.0
    for k in range((n-m)//2+1):
        wf += rho**(n-2.0*k) * (-1.0)**k * math.factorial(int(n-k)) / ( math.factorial(int(k)) * math.factorial( int((n+m)/2.0 - k) ) * math.factorial( int((n-m)/2.0 - k )) )
    return wf


def zernike(m, n, rho, phi, norm=True):
    '''
    Evaluates the full Zernike polynomial Z_n^m(rho, phi).
    
    Combines the radial polynomial with the cos(m*phi) (m > 0) or
    sin(|m|*phi) (m < 0) azimuthal term, optionally normalized so
    each mode has unit variance over the unit disk.
    
    Args:
        m: Azimuthal order, integer (sign selects cos/sin term).
        n: Radial order, integer.
        rho: Radial coordinate(s), normalized to 1 at the edge.
        phi: Azimuthal angle(s), radians.
        norm: Apply the unit-variance normalization factor (True).
    
    Returns:
        The Zernike mode evaluated at (rho, phi).
    '''
    nc = 1.0
    if (norm):
        nc = (2*(n+1)/(1+(m==0)))**0.5
    if (m > 0): return nc*zernike_rad(m, n, rho) * np.cos(m * phi)
    if (m < 0): return nc*zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return nc*zernike_rad(0, n, rho)


def noll_to_zern(j):
    '''
    Converts a Noll single index j to the (n, m) Zernike orders.
    
    Args:
        j: Noll index, integer starting at 1.
    
    Returns:
        (n, m): the radial order n and azimuthal order m.
    '''
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
    '''
    Evaluates the Noll-indexed Zernike mode j at (rho, phi).
    
    Args:
        j: Noll index, integer starting at 1.
        rho: Radial coordinate(s), normalized to 1 at the edge.
        phi: Azimuthal angle(s), radians.
        norm: Apply the unit-variance normalization factor (True).
    
    Returns:
        The Zernike mode j evaluated at (rho, phi).
    '''
    n, m = noll_to_zern(j)
    return zernike(m, n, rho, phi, norm)


def zernikeg(j, size=256, norm=True):
    '''
    Renders the Noll-indexed Zernike mode j onto a square grid.
    
    Builds a normalized (size, size) coordinate grid over the unit
    disk and evaluates the Zernike mode on it.
    
    Args:
        j: Noll index, integer starting at 1.
        size: Side length of the output grid, pixels (256).
        norm: Apply the unit-variance normalization factor (True).
    
    Returns:
        A (size, size) array of the Zernike mode over the grid.
    '''
    n, m = noll_to_zern(j)
    grid = (np.indices((size, size), dtype=float) - 0.5*size) / (0.5*size)
    grid_rad = (grid[0]**2. + grid[1]**2.)**0.5
    grid_ang = np.arctan2(grid[0], grid[1])
    return zernike(m, n, grid_rad, grid_ang, norm)