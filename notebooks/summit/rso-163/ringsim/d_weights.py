# Here lies awegiths and getweights5 functions
#!/usr/bin/env python3
'''
Computes the angular/radial response weights (aweight.pro) and
assembles the weight file for the two star colors (getweight5.pro).


Auth: A. Tokovinin
Translated: D. Hurtado

'''
import argparse
import codecs
import json
import logging
import numpy as np
import zernike



def main(par=None):

    # Import path: pass the par dict (from statmom) straight in.
    # CLI path: par is None, so read it from a --parfile JSON instead.
    if par is None:
        p = argparse.ArgumentParser()
        p.add_argument('--parfile', dest='parfile', type=str, default=None,
                       help='JSON par file with telescope/profrest sections (from statmom)')
        args = p.parse_args()
        par = getpar(args.parfile)

    weight = computeweight(par)
    print(f'\n Weights computed and saved to weights.json')
    print('  z-layers = {},  coeffs/layer = {}'.format(len(weight['z']), len(weight['wt0'][0])))
    print('  ring radius [pix] = {:.3f},  pdist [m] = {:.2f}'.format(weight['ringrad'], weight['pdist']))
    
    return weight
    
    
def aweight(z,mmax,d,eps,pdist,wav,sp,drho=1.5,pixel=0,zn=[],zrad=[],nsect=8):
    # Computing parameters hard coded
    ngrid = 512   # half-size of computing grid [pix], only for inner calculations
    ksize = 6     # grid size/telescope diameter ratio

    nz = z.shape[0]
    nwav = wav.shape[0]  # nwav is number of wavelengths
    spnorm = sp/np.sum(sp) # normalize the spectrum
    lam0 = np.sum(wav*spnorm) # average wavelength
    if nwav > 1:
        wavstep = wav[1] - wav[0] # step of wavelength grid, assumed uniform
    else:
        wavstep = 0

    # Define main calculation parameters
    size = d * ksize  # domain size in pupil plane, [m]
    asperpix = 206265 * lam0 / size  # fine-pixel in the image plane [arcsec]
    fstep = 1 / size  # frequency step [1/m]
    xstep = size / (2*ngrid)  # sampling of pupil grid, left for debug
    ringradpix = 0.85 * d * (1+eps) / (4*pdist) * 206265 / asperpix # ring radius in fine pixels
    
    # if ringradpix > 0.8 * ngrid:
    #     print('Error: ring too wide, returning!')
        

    # Prepare the arrays
    i = np.indices((2 * ngrid, 2 * ngrid))
    x = i[1] - ngrid
    y = i[0] - ngrid
    r = np.sqrt( np.square(x) + np.square(y))
    r[ngrid,ngrid] = 1e-3 # to avoid division by zero
    phi = np.arctan2(y,x)  # 2D array of phase # plt.imshow(phi, cmap='Greys')

    # Define the annular aperture in the pupil space
    radpix = ngrid * d / size  # aperture radius, pixels
    pupil = (r <= radpix) * (r >= eps*radpix) # boolean array, true inside pupil
    ninside = np.sum(pupil)  # pupil surface [pix]

    # Define conic wavefront at the pupil
    a4 = d**2 / (lam0 * pdist) * (np.pi / 8 * 3**(-0.5))   # Zernike defocus corresponding to the propagation distance [rad]
    a11 = -0.1 * a4       # spherical aberration coef. [rad]
    tmp = a11 * 5**(0.5) * (6 * (r/radpix)**4  - 6 * (r/radpix)**2)
    tmp = tmp + a4 * 2 * 3**0.5 * ((r/radpix)**2 - 0.5)  # wavefront shape [rad]
    
    # Optionally add Zernike aberrations
    nzern = len(zn)
    for j in range(0,nzern):
        tmp += zrad[j]*zernike.zernikel(zn[j],r/radpix,phi)


    uampl = pupil*(np.cos(tmp) + np.sin(tmp)*1j) # complex amplitude at the pupil

    # Compute nominal ring image at the focal plane
    imh = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(uampl)))
    imh = np.power(np.abs(imh),2)/( (2 * ngrid)**2) # np.sum(imh) = ninside to check normalization

    # Compute the masks
    ringradpix2 = np.sum(imh * r) / ninside # true ring radius in fine pixels
    ringrad = ringradpix2 * asperpix / pixel  # ring radius in CCD pixels
    print ('Ring radius [pix]: ',ringrad)
    drhopix = drho * lam0 / d / (1 - eps) * 2 * 206265 / asperpix # ring half-width [pix]
    filtap = (r >= ringradpix2 - drhopix) * (r <= ringradpix2 + drhopix) # radial part of image mask
    nm = mmax + 1  # number of angular coefficients
    wt = np.zeros((nz,nm))
    ufunc = np.zeros((nz,nm))

    # Prepare things used in the loop
    spturb = np.power(r, -11 / 3 ) * fstep**(-5 / 3) * ( 0.5 * 9.62 / np.pi ) * lam0**(-2) # turbulence phase spectrum for Jturb=1
    spturb[ngrid,ngrid] = 0
    spufunc = spturb * np.square(np.pi * fstep * r)  # for U-function calculation
    flux = np.sum(imh * filtap) # flux inside the ring mask
    utmp = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.conj(uampl)))) # auxiliary conjugated amplitude
    #utmp = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.conj(uampl.copy())))) # auxiliary conjugated amplitude

    # Radial mask for differential sector motion
    sectrad = np.pi / nsect # sector width [rad] = 22.5deg for nsect=8
    tmp = np.mod(phi + np.pi, np.pi)  # 180-folded phase
    sector = (tmp >= np.pi / 2 - sectrad) * (tmp < np.pi / 2 + sectrad) # 1 within opposite 45-deg sectors
    sector = np.transpose(sector) # rotate 90 degrees to match IDL, sectors along X
    ringmask = filtap * sector
    rflux = np.sum(imh * ringmask) # 1/4 of full flux
    rwt = r * ringmask
    tmp = np.sum(imh * rwt) / rflux  # mean radius, to make np.sum(imh*rwt)=0
    rwt = rwt - tmp * ringmask # subtract to get zero signal without turbulence
    rwt = rwt / ksize  # radius in lam/D units instead of fine pixels

    # Prepare 2D Fresnel filters for propagation calculation
    frecos = np.zeros((nz, 2 * ngrid, 2 * ngrid))
    fresin = np.zeros((nz, 2 * ngrid, 2 * ngrid))
    for iz in range(0,nz):
        zdist = z[iz]
        damp1 = np.exp(-np.square(0.5 * np.square(r * fstep) * wavstep * zdist)) #0.5 damping factor
        spcos = np.zeros((2 * ngrid, 2 * ngrid))
        spsin = np.zeros((2 * ngrid, 2 * ngrid))
        for j in range(0,nwav):
            w = wav[j]
            arg = np.pi * w * zdist * np.square(r * fstep)
            a = spnorm[j] * lam0 / w
            spcos += a * np.cos(arg)
            spsin += a * np.sin(arg)
        frecos[iz,:,:] = spcos / spcos[ngrid,ngrid] * damp1
        fresin[iz,:,:] = spsin / spcos[ngrid,ngrid] * damp1

    # Compute the weights, loops in m and z
    for m in range(0,nm):  # loop over m
        if m==0: # radial weight
            cwt = rwt
            swt = 0
            pixfact = 1.
        else:
            cwt = np.cos(phi * m) * filtap # cosine mask
            swt = np.sin(phi * m) * filtap # sine mask
            if pixel > 0:   # pixel averaging factor
                arg = pixel / asperpix / (1.5 * ringradpix2) * m
                pixfact = np.sin(arg) / arg
            else:
                pixfact = 1
        # Normalize by sector or total flux
        if m==0:
            normfact = 1 / rflux # for radial weight
        else:
            normfact = 1 / flux  # for angular weight
        # Response to the cosine mask
        result1 = uampl * np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(utmp * cwt)))
        result1 *= np.square(2 * ngrid)
        fphase1 = -result1.imag * normfact
        fampl1 = result1.real * normfact
        # Response to the sine mask
        if m>0:
            result2 = uampl * np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(utmp * swt)))
            result2 *= np.square(2 * ngrid)
            fphase2 = -result2.imag * normfact
            fampl2 = result2.real * normfact
        else:
            fphase2 = fampl2 = 0
        # Propagation filter
        for iz in range(0,nz):   # loop over distance grid
            zdist = z[iz]
            ctmp = frecos[iz,:,:]
            stmp = fresin[iz,:,:]
            # Propagate cos/sin filters over distance z
            tmp1 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fphase1))) * ctmp
            tmp1 += np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fampl1))) * stmp
            if m>0:
                tmp2 = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fphase2))) * ctmp
                tmp2 +=  np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fampl2))) * stmp
            else:
                tmp2 = 0
            pfilter = np.power(np.abs(tmp1),2) + np.power(np.abs(tmp2),2)
            wt[iz,m] = np.sum(pfilter * spturb) * pixfact
            ufunc[iz,m] = np.sum(pfilter * spufunc)
    ###  End of the weight-calculation loop over m and z
    return wt, ufunc, ringrad 


def blackbody(wav,temp):
    # Return blackbody spectrum with arbitrary normalization, 
    # photons/lambda, wavelength in m, temperature in K
    const = 0.014387618 # in [m.K]
    planck = np.power(wav[0] / wav,4) / (np.exp(const / wav / temp) -1)
    return planck / np.max(planck)


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


def getpar(parfile): # read parameters, return the dictionary <par>
    try:
        file = open(parfile, 'r')
        par = json.load(file)   # par is a nested dictionary
    except FileNotFoundError as err:
        print(err)
        quit()
    file.close()
    return par


def computeweight(par):  # actual weight calculation

    # This should be argument parser
    d = float(par['telescope']['D'])
    eps = float(par['telescope']['eps'])
    pdist = par['telescope']['pdist']
    pixscale = float(par['telescope']['pixel'])
    ringradpix = float(par['telescope']['ringradpix'])
    wav = np.array(par['profrest']['wavelen'])
    sp = np.array(par['profrest']['sp']) # spectral response
    mmax = par['profrest']['mmax']
    nsect = par['profrest']['nsect']
    
    if 'aber' in par['profrest']:
        adict = getpar(par['profrest']['aber'])
        zn = [2,3,4,5,6,7,8,9,10]
        zrad = np.zeros(9)
        zrad[2:9] = np.array(adict['zampl'], float) # Focus to trefoil
        zrad[0] =  -5 * zrad[6]
        zrad[1] = -5 * zrad[5]
        s = 'Zrad:  '
        for i in range(0,7):
            s += ' {:.3f}'.format(zrad[i])
        print(s)
        #print(zrad)
    else:
         zn = zrad = []
    
    sp0 = sp * blackbody(wav, 10213) # B-V=0 spectrum
    sp1 = sp * blackbody(wav, 3938) # B-V=1 spectrum
    sp0 = sp0 / np.sum(sp0) # normalize
    sp1 = sp1 / np.sum(sp1)
    lameff = (np.sum(wav * sp0), np.sum(wav * sp1)) # effective wavelength
    
    # distance grid, log-spaced with sqrt(2) step, 0.25-32km
    nz = 16
    z = np.zeros(nz)
    z[1:nz] = 1e3 * 2**(0.5 * np.arange(nz-1) -2)

    # Find propagation distance from ring radius. Use analytic approx. first
    Rmean = d * (1 + eps) / 4 # Mean of annulus radius
    HR = 0.85 * Rmean / pixscale * 206265 # H * R (proxy for alpha in papers)
    print(f'Rmean = {d} * (1 + {eps}) / 4')
    print(f'Initial H*R [m.pix] and Mean radius: {HR}, {Rmean}')

    
    pdist1 = HR / ringradpix
    wt0, ufunc0, ringrad = aweight(np.zeros(1),1,d,eps,pdist,wav,sp0,1.5,pixscale,zn,zrad,nsect)
    
    HR1 = ringrad * pdist # Adjust    
    print(f'Expected Conjugation dist.: {pdist}')
    pdist = HR1 / ringradpix
    print(f'Calculated Conjugation dist.: {pdist1}')
    print(f'Adjusted H*R [m.pix] and pdist: {HR1}, {pdist}')

    # return

    # Weight for B-V=0
    print(f'\n')
    print('Computing weight for B-V=0...')
    wt0, ufunc0, ringrad = aweight(z,mmax,d,eps,pdist,wav,sp0,1.5,pixscale,zn,zrad,nsect) # arrays of [nz,mmax+1] dimension
    hslope = pdist * ringrad
    print(f'Ring radius and H*rad [m.pix]: {ringrad}, {hslope}')
    # Weight for B-V=1
    print('Computing weight for B-V=1...')
    wt1, ufunc1, ringrad1 = aweight(z,mmax,d,eps,pdist,wav,sp1,1.5,pixscale,zn,zrad,nsect) # arrays of [nz,mmax+1] dimension

    mm = [1,3,6,7,8,9] # selected frequencies for wind measurement
    ucoef0, resp0 = getucoef(ufunc0,z,mm)
    ucoef1, resp1 = getucoef(ufunc1,z,mm)
    #plt.plot(resp0)
    #plt.plot(resp1)
    
    # Color dependence
    wtslope = wt1 - wt0
    ucoefslope = ucoef1 - ucoef0

    # Serialize and save the weights
    weight = {
        'z':z.tolist(),'wt0':wt0.tolist(),
        'wtslope':wtslope.tolist(),'ucoef0':ucoef0.tolist(),
        'ucoefslope':ucoefslope.tolist(),'umm':mm,'lameff':lameff,
        'ringrad':ringrad,'pdist':pdist
    }

    json_output_file = par['profrest']['weightfile']
    try:
        json.dump(weight, codecs.open(json_output_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    except FileNotFoundError as err:
        print('{}:{}'.format(err, json_output_file))
    print('Saved weights in '+json_output_file)
    return weight


if __name__ == '__main__':
    main()