#!/usr/bin/env python
# coding: utf-8

# Calculator of RINGSS image. AT, 2021-06-01
# Input parameters (all lengths in meters):
# d - aperture diameter
# eps - central obscuration
# pdist - virtual propagation distance (measure of defocus)
# lam0 - wavelength
# pixel - CCD pixel size in arcseconds
# ----------------------------------------
# returns the image and the ring radius in pixels

import numpy as np
import zernike

def rebin(a, shape):  # recipe from stack overflow 
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def getimage(d,eps,pdist,lam,pixel,zn=[],zrad=[]):
    # Computing paramters
    ngrid = 256   # half-size of computing grid [pix]

    d1 = (lam/pixel)*206265.  # linear size matching pixel
    npixperpix = pow(2, int( np.log(1.5*d/d1)/np.log(2.) + 2 ))
    # print("Over-sampling: ",npixperpix)
    nccd = int(2*ngrid/npixperpix)  # Image size in coarse pixels

    asperpix = pixel/npixperpix  # fine-pixel in the image plane [arcsec]
    size = lam/asperpix*206265. # Computing grid [m]
    # print("Grid size [m[: ",size)
    
    xstep = size/(2*ngrid)  # pixel in the pupil plane
    ringradpix = 0.93*d*(1.+ eps)/(4.*pdist)*206265./asperpix # approx ring radius in fine pixels
    if ringradpix > 0.8*ngrid:
        print("Error: ring too wide, returning!")
        return 0, 0

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
    ninside = np.sum(pupil)  # number of pixels in the pupil

    # Define the conic wavefront at the pupil
    a4 = d**2/(lam*pdist)* (np.pi/8*pow(3,-0.5))   # Zernike defocus corresponding to the propagation distance [rad]
    a11 = -0.1*a4       # spherical aberration coef. [rad]
    tmp = a11*pow(5,0.5)*(6.*np.power(r/radpix, 4)  - 6.*np.power(r/radpix,2))
    tmp = tmp + a4*2.*pow(3,0.5)*( np.power(r/radpix,2) - 0.5)  # wavefront shape [rad]

    # Optionally add Zernike aberrations
    nz = len(zn)
    for j in range(0,nz):
        tmp += zrad[j]*zernike.zernikel(zn[j],r/radpix,phi)
    
    uampl = pupil*(np.cos(tmp) + np.sin(tmp)*1j) # complex amplitude at the pupil

    # Compute nominal ring image at the focal plane
    imh = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(uampl)))
    #imh = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(uampl))) #n0n-inverse FFT
    imh = np.power(np.abs(imh),2) # np.sum(imh) = ninside to check normalization
    #imh = np.power(np.abs(imh),2)/( (2.*ngrid)**2) # np.sum(imh) = ninside to check normalization
    #imh = imh/np.sum(imh) # normalize

    ringradpix2 = np.sum(imh*r)/np.sum(imh) # true ring radius in fine pixels
    rad = ringradpix2/npixperpix  # radius in CCD pixels
    shift = npixperpix // 2
    imh = np.roll(imh, (shift,shift),axis=[0,1]) # shift by 1/2 of CCD pixel before rebinning
    img = rebin(imh,(nccd,nccd))
    img = img/np.sum(img) # normalize on coarse pixels
    return img, rad
