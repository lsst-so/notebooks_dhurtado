#!/usr/bin/env python
# coding: utf-8

#import matplotlib.pyplot as plt
import numpy as np
import json 
import codecs
import sys
import aweight

# Return blackbody spectrum with arbitrary normalization, photons/lambda
# wavelength in m, temperature in K
def blackbody(wav,temp):
    const = 0.014387618 # in [m.K]
    planck = np.power(wav[0]/wav,4)/ (np.exp(const/wav/temp) -1)
    return planck/np.max(planck)
# Find U-coefficients
def getucoef(ufunc,z,mm):
    nm = len(mm)
    nz = z.shape[0] -2 # number of layers, exclude 2 lowest
    amat = np.zeros((nm,nz)) # Matrix of the inear-equations system
    # wz = z[2:nz+2]*1e-3 + 0.5 # distance-dependent weight for response calc.
    wz = np.zeros(nz) + 1. # distance-independent weight
    for i in range(0,nm):
        amat[i,:] = ufunc[2:nz+2,mm[i]]*wz
# Least-squares system 
    aa = np.dot(amat,np.transpose(amat)) # 6x6 square matrix 
    bb = np.dot(amat,wz) # 6-element vector of right-hand terms
    ainv = np.linalg.pinv(aa,1e-4)  # SVD inversion with 1E-4 threshold
    ucoef = np.dot(ainv,bb)
    uresp = np.zeros(nz) # resulting response, must be close to one at all z>0
    for i in range(0,nm):
        uresp = uresp + ucoef[i]*ufunc[2:nz+2,mm[i]]
    #plt.plot(z,uresp)
    return ucoef, uresp

# read parameters, return the dictionary <par>
def getpar(parfile):  
    try:
        file = open(parfile, "r")
        par = json.load(file)   # par is a nested dictionary
    except FileNotFoundError as err:
        print(err)
        quit()
    file.close()    
    return par

def computeweight(par):  # actual weight calculation
    d = float(par["telescope"]["D"])
    eps = float(par["telescope"]["eps"])
    # pdist = par["telescope"]["pdist"]
    pixel = float(par["telescope"]["pixel"])
    ringradpix = float(par["telescope"]["ringradpix"])
    wav = np.array(par["profrest"]["wav"])*1e-9 # wavelength in m
    sp = np.array(par["profrest"]["sp"]) # spectral response
    sp0 = sp * blackbody(wav, 10213.) # B-V=0 spectrum
    sp1 = sp * blackbody(wav, 3938.) # B-V=1 spectrum
    sp0 = sp0/np.sum(sp0) # normalize
    sp1 = sp1/np.sum(sp1)
    lameff = (np.sum(wav*sp0), np.sum(wav*sp1)) # effective wavelength
    mmax = par["profrest"]["mmax"]
    if "aber" in par["profrest"]:
        adict = getpar(par["profrest"]["aber"])
        zn = [2,3,4,5,6,7,8,9,10]
        zrad = np.zeros(9)
        zrad[2:9] = np.array(adict["zampl"], float) # Focus to trefoil
        zrad[0] =  -5.*zrad[6]
        zrad[1] = -5.*zrad[5]
        s = "Zrad:  "
        for i in range(0,7):
            s += " {:.3f}".format(zrad[i])
        #print(s)
        #print(zrad)                    
    else:
         zn = zrad = []
         
    
# distance grid, log-spaced with sqrt(2) step, 0.25-32km
    nz = 16 
    z = np.zeros(nz)
    z[1:nz] = 1e3*2.**(0.5*np.arange(nz-1)-2)

# Find propagation distance from ring radius. Use analytic approx. first
    HR = 0.85*d*(1 + eps)/4./pixel*206265
    #print("Initial H*R [m.pix]: {:.2f} ".format(HR))
    pdist = HR / ringradpix
    wt0, ufunc0, ringrad = aweight.aweight(np.zeros(1),1,d,eps,pdist,wav,sp0,1.5,pixel,zn,zrad)
    HR1 = ringrad*pdist
    pdist = HR1 / ringradpix
    #print("Final H*R [m.pix] and pdist: {:.2f} {:.2f} ".format(HR1,pdist))

    # return    

# Weight for B-V=0
    #print("Computing weight for B-V=0...")
    wt0, ufunc0, ringrad = aweight.aweight(z,mmax,d,eps,pdist,wav,sp0,1.5,pixel,zn,zrad) # arrays of [nz,mmax+1] dimension
    hslope = pdist*ringrad
    #print("Ring radius and H*rad [m.pix]: {:.3f} {:.3f} ".format(ringrad, hslope))
# Weight for B-V=1
    #print("Computing weight for B-V=1...")
    wt1, ufunc1, ringrad1 = aweight.aweight(z,mmax,d,eps,pdist,wav,sp1,1.5,pixel,zn,zrad) # arrays of [nz,mmax+1] dimension
    
    mm =  [1,3,6,7,8,9] # selected frequencies for wind measurement
    ucoef0, resp0 = getucoef(ufunc0,z,mm)
    ucoef1, resp1 = getucoef(ufunc1,z,mm)
    #plt.plot(resp0)
    #plt.plot(resp1)
# Color dependence
    wtslope = wt1 - wt0
    ucoefslope = ucoef1 - ucoef0

# serialize and save the weights
    weight = {"z":z.tolist(),"wt0":wt0.tolist(),"wtslope":wtslope.tolist(),"ucoef0":ucoef0.tolist(),"ucoefslope":ucoefslope.tolist(),"umm":mm,"lameff":lameff,"ringrad":ringrad,"pdist":pdist} 

    json_output_file = par["profrest"]["weightfile"]
    try:
        json.dump(weight, codecs.open(json_output_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    except FileNotFoundError as err:
        print('{}:{}'.format(err, json_output_file))
    #print("Saved weights in "+json_output_file)
# ### End of getweight module


 # ### Main module. usage: > python <par> <data>
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python getweight5.py <par-file> ")
        sys.exit()

    parfile = sys.argv[1]
    #print("Parameter file: " + parfile)    
    par = getpar(parfile)
    computeweight(par)

