#!/usr/bin/env python
# coding: utf-8

# Calculator of RINGSS weights. AT, 2021-04-19
# based on aweight3.pro IDL code
# radial weight is stored as m=0 in lambda/D units
# Input parameters (all lengths in meters):
# z - propagation distances (numpy array)
# mmax - maximum anguar frequency
# d - aperture diameter
# eps - central obscuration
# pdist - virtual propagation distance (measure of defocus)
# wav - list of wavelengths (array)
# sp - list of spectral response (same number as wav, array)
# drho - relative ring width (1.5 is default)
# pixel - CCD pixel size in arcseconds (0 is default)
# zn - numbers of static Zernike aberrations (optional)
# zrad - their amplitudes in radian (if zn is specified)
# ----------------------------------------
# returns wt[nz,mmax+1] - weights in m^(-1/3), m=0 is for sector variance in (lam/D)
# and ufunc[nz,mmax+1] - U-functions for wind-speed calculation 

import numpy as np
import zernike
import sys

def aweight(z,mmax,d,eps,pdist,wav,sp,drho=1.5,pixel=0,zn=[],zrad=[]):
    # Computing paramters
    nsect = 8     # number of sectors for radial weight
    ngrid = 256   # half-size of computing grid [pix]
    ksize = 6     # grid size/telescope diameter ratio

    nz = z.shape[0]
    nwav = wav.shape[0]  # nwav is number of wavelengths
    spnorm = sp/np.sum(sp) # normalize the spectrum
    lam0 = np.sum(wav*spnorm) # average wavelength
    if nwav > 1:
        wavstep = wav[1] - wav[0] # step of wavelength grid, assumed uniform
    else:
        wavstep = 0.

    # Define main calculation parameters
    size = d*ksize  # domain size in pupil plane, [m]
    asperpix = 206265.*lam0/size  # fine-pixel in the image plane [arcsec]
    fstep = 1./size  # frequency step [1/m]
    xstep = size/(2*ngrid)  # pixel in the pupil plane
    ringradpix = 0.85*d*(1.+ eps)/(4.*pdist)*206265./asperpix # ring radius in fine pixels
    if ringradpix > 0.8*ngrid:
        print("Error: ring too wide, returning!")
        sys.exit(0)

    # Prepare the arrays
    i = np.indices((2*ngrid,2*ngrid))
    x = i[1] - ngrid
    y = i[0] - ngrid
    r = np.sqrt( np.square(x) + np.square(y))
    r[ngrid,ngrid] = 1e-3 # to avoid division by zero
    phi = np.arctan2(y,x)  # 2D array of phase # plt.imshow(phi, cmap='Greys')

    # Define the annular aperture in the pupil space
    radpix = ngrid*d/size  # aperture radius, pixels
    pupil = (r <= radpix) * (r >= eps*radpix) # boolean array, true inside pupil
    ninside = np.sum(pupil)  # pupil surface [pix]

    # Define conic wavefront at the pupil
    a4 = d**2/(lam0*pdist)* (np.pi/8*pow(3,-0.5))   # Zernike defocus corresponding to the propagation distance [rad]
    a11 = -0.1*a4       # spherical aberration coef. [rad]
    tmp = a11*pow(5,0.5)*(6.*np.power(r/radpix, 4)  - 6.*np.power(r/radpix,2))
    tmp = tmp + a4*2.*pow(3,0.5)*( np.power(r/radpix,2) - 0.5)  # wavefront shape [rad]
    # Optionally add Zernike aberrations
    nzern = len(zn)
    for j in range(0,nzern):
        tmp += zrad[j]*zernike.zernikel(zn[j],r/radpix,phi)

    
    uampl = pupil*(np.cos(tmp) + np.sin(tmp)*1j) # complex amplitude at the pupil

    # Compute nominal ring image at the focal plane
    imh = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(uampl)))
    imh = np.power(np.abs(imh),2)/( (2.*ngrid)**2) # np.sum(imh) = ninside to check normalization

# ###----- Compute the masks
    ringradpix2 = np.sum(imh*r)/ninside # true ring radius in fine pixels
    ringrad = ringradpix2*asperpix/pixel  # ring radius in CCD pixels
    # print ("Ring radius [pix]: ",ringrad)
    drhopix = drho*lam0/d/(1. -eps)*2.*206265./asperpix # ring half-width [pix]
    filtap = (r >= ringradpix2 - drhopix) * (r <= ringradpix2 + drhopix) # radial part of image mask
    nm = mmax+1  # number of angular coefficients
    wt = np.zeros((nz,nm))
    ufunc = np.zeros((nz,nm))

    # Prepare things used in the loop
    spturb = np.power(r, -11./3.)*pow(fstep, -5./3.)*(0.5*9.62/np.pi)*pow(lam0,-2) # turbulence phase spectrum for Jturb=1
    spturb[ngrid,ngrid] = 0
    spufunc = spturb*np.square(np.pi*fstep*r)  # for U-function calculation
    flux = np.sum(imh*filtap) # flux inside the ring mask
    utmp = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.conj(uampl)))) # auxiliary conjugated amplitude
    #utmp = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.conj(uampl.copy())))) # auxiliary conjugated amplitude

    # Radial mask for differential sector motion
    sectrad = np.pi/nsect # sector width [rad] = 22.5deg for nsect=8
    tmp = np.mod(phi + np.pi, np.pi)  # 180-folded phase
    sector = (tmp >= np.pi/2 - sectrad) * (tmp < np.pi/2 + sectrad) # 1 within opposite 45-deg sectors
    sector = np.transpose(sector) # rotate 90 degrees to match IDL, sectors along X
    ringmask = filtap*sector
    rflux = np.sum(imh*ringmask) # 1/4 of full flux
    rwt = r * ringmask
    tmp = np.sum(imh*rwt)/rflux  # mean radius, to make np.sum(imh*rwt)=0
    rwt = rwt -  tmp*ringmask # subtract to get zero signal without turbulence
    rwt = rwt/ksize  # radius in lam/D units instead of fine pixels

    # Prepare 2D Fresnel filters for propagation calculation
    frecos = np.zeros((nz,2*ngrid,2*ngrid))
    fresin = np.zeros((nz,2*ngrid,2*ngrid))
    for iz in range(0,nz):
        zdist = z[iz]
        damp1 = np.exp(- np.square(0.5*np.square(r*fstep)*wavstep*zdist)) #0.5 damping factor
        spcos = np.zeros((2*ngrid,2*ngrid))
        spsin = np.zeros((2*ngrid,2*ngrid))
        for j in range(0,nwav):
            w = wav[j]
            arg = np.pi*w*zdist*np.square(r*fstep)
            a = spnorm[j]*lam0/w
            spcos += a*np.cos(arg)
            spsin += a*np.sin(arg)
        frecos[iz,:,:] = spcos/spcos[ngrid,ngrid]*damp1
        fresin[iz,:,:] = spsin/spcos[ngrid,ngrid]*damp1

# ### ----- Compute the weights, loops in m and z
    for m in range(0,nm):  # loop over m   
        if m==0: # radial weight
            cwt = rwt
            swt = 0
            pixfact = 1.
        else:
            cwt = np.cos(phi*m)*filtap # cosine mask
            swt = np.sin(phi*m)*filtap # sine mask 
            if pixel > 0:   # pixel averaging factor
                arg = pixel/asperpix/(1.5*ringradpix2)*m
                pixfact = np.sin(arg)/arg
            else:
                pixfact = 1
        # Normalize by sector or total flux
        if m==0:
            normfact = 1./rflux # for radial weight
        else:
            normfact = 1./flux  # for angular weight
        # Response to the cosine mask    
        result1 = uampl*np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(utmp*cwt)))
        result1 *= np.square(2*ngrid)
        fphase1 = -result1.imag*normfact
        fampl1 = result1.real*normfact
        # Response to the sine mask
        if m>0:
            result2 = uampl*np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(utmp*swt)))
            result2 *= np.square(2*ngrid)
            fphase2 = -result2.imag*normfact
            fampl2 = result2.real*normfact
        else:
            fphase2 = fampl2 = 0
        # Propagation filter
        for iz in range(0,nz):   # loop ovr distance grid
            zdist = z[iz]
            ctmp = frecos[iz,:,:]
            stmp = fresin[iz,:,:]
            # Propagate cos/sin filters over distance z
            tmp1 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fphase1)))*ctmp 
            tmp1 += np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fampl1)))*stmp
            if m>0:                                                                                     
                tmp2 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fphase2)))*ctmp 
                tmp2 +=  np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fampl2)))*stmp
            else:
                tmp2 = 0
            pfilter = np.power(np.abs(tmp1),2) + np.power(np.abs(tmp2),2)     
            wt[iz,m] = np.sum(pfilter*spturb)*pixfact
            ufunc[iz,m] = np.sum(pfilter*spufunc)
###  End of the weight-calculation loop over m and z            
    return wt, ufunc, ringrad
