#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json 
import codecs
import sys
import getimage

# read a json file, return a dictionary; returns None if fails
def read_json(filename):
    try:
        file = open(filename, "r")
        p = json.load(file)  
        file.close()
        return p
    except FileNotFoundError as err:
        print(err)
        return

# Find response to 7 aberrations (focus to trefoil)
# Returns list of 7 numbers
def aberresp(par):

    d = par["telescope"]["D"]
    eps = par["telescope"]["eps"]
    pdist = par["telescope"]["pdist"]
    pixel = par["telescope"]["pixel"]
    lam = 0.6e-6   #  600nm fixed

    img0, radpix = getimage.getimage(d,eps,pdist,lam,pixel) # image array and ring radius in pixels
    nx =img0.shape[0]
    print("Image size, ring radius [pix]: ",nx, radpix)
        
    m = 3 # mmax order of angular signals
    nsect = 8 # number of sectors for radius calculation

    i = np.indices((nx,nx))  # to create xx and yy arrays
    yy =i[0] - nx/2
    xx =i[1] - nx/2
    r = np.sqrt(np.square(xx) + np.square(yy)) # distance from center: plt.imshow(xx, cmap='Greys')
    phi = np.arctan2(yy,xx)  # 2D array of phase # plt.imshow(phi, cmap='Greys')


    ### Masks are defined here as in cube2.py
    rwt = np.zeros((nsect,nx,nx))  # for radius calculation  
    fwt = np.zeros((nsect,nx,nx))  # fluxes in the sectors
    sect = 2.*np.pi/nsect # sector width in radians
    for j in range(0,nsect):
       sector = (phi >= (sect*(j - nsect/2))) & (phi < (sect*(j+1 - nsect/2)))
       fwt[j] = sector
       rwt[j] = sector*r

    phisect = sect*(np.arange(nsect, dtype=float) - nsect/2 + 0.5) # sector angles
    xsect, ysect = np.cos(phisect), np.sin(phisect)
    
    rwidth = lam/d/(1. - eps)*2.*206265./pixel # diffraction-lim. ring width 
    drhopix = 1.5*rwidth # mask width, pixels
    ringmask = (r >= radpix - drhopix) * (r <= radpix + drhopix)

    ncoef = 2*nsect + 2*m
    maskmat = np.zeros((ncoef,nx*nx))

    for j in range(0,nsect):   # radial masks
        maskmat[j,:] = np.ndarray.flatten(rwt[j]*ringmask)  # image pixels arranged in 1D array
        maskmat[j+nsect,:] = np.ndarray.flatten(fwt[j]*ringmask)

    for j in range(0,m):  # cosine ans sine masks for m=1,2,3
        tmp = np.cos(phi*(j+1))*ringmask
        cwt = tmp - np.sum(tmp)/nx/nx # rmove residual piston
        tmp = np.sin(phi*(j+1))*ringmask
        swt = tmp - np.sum(tmp)/nx/nx # remove piston
        maskmat[2*nsect+j,:] = np.ndarray.flatten(cwt)
        maskmat[2*nsect+m+j,:] = np.ndarray.flatten(swt)

    # Compute sector radii for unaberrated image img0
    c = np.dot(maskmat,np.ndarray.flatten(img0))
    radii = c[0:nsect]/c[nsect:2*nsect]
    radpix2 = np.mean(radii)
    print("Nominal radius (sect, getimage): ", radpix2, radpix)

    zn = [4,5,6,7,8,9,10] # Zernike numbers
    zampl = 0.5 # 0.5 rad at lam0 wavelength
    nz = len(zn) # number of terms
    ncoef2 = nsect + 2*m # only useful coefficients
    coef = np.zeros((nz,ncoef2)) # coefficients for each aberration

    for i in range(0,nz): # Main loop over 7 aberrations
        # image array and ring radius in pixels for i-th aberration
        img, rad = getimage.getimage(d,eps,pdist,lam,pixel, [zn[i]],[zampl]) 
  
        c = np.dot(maskmat,np.ndarray.flatten(img)) # coefs before centering
        radii = c[0:nsect]/c[nsect:2*nsect]
        dx = np.sum(radii*xsect)/nsect*2.3
        dy = np.sum(radii*ysect)/nsect*2.3
        arg = 2*np.pi*(dx*xx + dy*yy)/nx  # FFT centering
        img = np.fft.ifft2(np.fft.fft2(img)*np.fft.fftshift(np.cos(arg) + np.sin(arg)*1j)).real
      
        c = np.dot(maskmat,np.ndarray.flatten(img)) # re-compute on centered ring
        radii = c[0:nsect]/c[nsect:2*nsect] - radpix2
        coef[i,0:nsect] = radii
        coef[i,nsect:nsect+2*m] = c[2*nsect:2*nsect+2*m]
    # end of main loop
    coef = coef/zampl # normalize to 1 radian aberration amplitude

    # Find response to aberrations from the coefficients
    smat = np.zeros((nz,ncoef2))
    smat[0, 0:nsect] = 1   # focus
    smat[1, 0:nsect] = np.sin(2.*phisect) #a5
    smat[2, 0:nsect] = np.cos(2.*phisect) #a6
    smat[3,nsect+3] = 1  # a7
    smat[4,nsect] = 1  # a8
    smat[5,nsect+5] = 1  # a9
    smat[6,nsect+2] = 1  # a10
    
    rmat = smat.copy()
    r = np.zeros((nz))
    for i in range(0,nz):
        r[i] = np.sum(smat[i,:] * coef[i,:])
        rmat[i,:] = smat[i,:]/r[i]         
    
    dict = {'aberresp':r.tolist(),'smat':smat.tolist(),'meanrad':radpix2}   
    return dict

### Find aberrations from 14 mean coefficients saved by cube2.pro
# Returns array of  7 Zernike amlitudes. Input: aberresp dictionary and 14 mean coefs
def getzampl(adict,mcoef):
    smat = adict['smat']    # 2D list
    abercoef = adict['aberresp'] # 7-el. list
    meanrad = adict['meanrad']

    nz = len(smat)
    s = np.array(smat)
    r = np.array(abercoef)
    coef = np.array(mcoef)
    coef[0:8] -= meanrad  # subtract expected ring radius
    
    zampl = np.zeros(nz) # Zernike amplitude in radian at 0.6 micron
    for i in range(0,nz):
        zampl[i] = np.sum(s[i,:]*coef)/r[i]

    return zampl 

# ### Main module. usage: > python <par> <data>
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aberration.py <par-file> ")
        sys.exit()

    parfile = sys.argv[1]
    #datafile = sys.argv[2]
    # print("Parameter file: " + parfile)
    
    # Ingest all information needed
    par = read_json(parfile)
    if par == None:
        sys.exit()

    if not ("aber" in par["profrest"]):
        print("Parameters[profrest] do not contain the aber tag, returning")
        sys.exit()
    aberfile =  par["profrest"]["aber"]    
    
    # Find the response    
    adict = aberresp(par)
    
    print("Response:",adict['aberresp'])   
 
    json.dump(adict, codecs.open(aberfile, 'w', encoding='utf-8'), separators=(',', ':'))
    print("Result is saved in "+aberfile)

   
   
    

