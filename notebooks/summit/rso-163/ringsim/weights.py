#!/usr/bin/env python3
'''
Computes the angular/radial response weights (aweight.pro) and
assembles the weight file for two star colors (getweight5.pro).

Author: A. Tokovinin
Compile and docstrings: D. Hurtado

'''
import argparse
import codecs
import json
import logging
import numpy as np
from zernike import zernikel


def main(par=None, verb=True):
    '''
    Compiles variables and runs the weight computation.
    
    If the par dict is passed in (import path), uses it directly;
    otherwise reads it from a --parfile JSON on the CLI.
    
    Args:
        par: Telescope/profrest parameter dict (from statmom); None
            to read from --parfile.
        verb: Print text to the CLI (True).
    
    Returns:
        The weight dict (also saved to the weightfile).
    '''
    if par is None:
        p = argparse.ArgumentParser()
        p.add_argument('--parfile', dest='parfile', type=str, default=None,
                       help='JSON par file with telescope/profrest sections (from statmom)')
        p.add_argument('--verbose', dest='verb', type=bool, default=True,
                       help='Print text into CLI, def=True')
        args = p.parse_args()
        par = getpar(args.parfile)
        verb = args.verb
    
    weight = computeweight(par, verb)
    if verb: print('\n Weights computed and saved to weights.json')
    if verb: print('  z-layers = {},  coeffs/layer = {}'.format(len(weight['z']), len(weight['wt0'][0])))
    if verb: print('  ring radius [pix] = {:.3f},  pdist [m] = {:.2f}'.format(weight['ringrad'], weight['pdist']))
    
    return weight
    
    
def aweight(z,mmax,d,eps,pdist,wav,sp,drho=1.5,pixel=0,zn=[],zrad=[],nsect=8,verb=True):
    '''
    Main script that evaluates the weights of each
    layer and respect a single star color.
    
    Builds pupil grid, defocused ring image, ring/sector
    masks and the Fresnel propagation filters, then loops over
    angular order m and layer z to compute the scintillation
    response weights and the U-functions used for wind.
    
    Args:
        z: Propagation-distance grid, meters.
        mmax: Max angular order (gives mmax+1 coefficients).
        d: Aperture diameter, meters.
        eps: Central obscuration fraction.
        pdist: Conjugation (defocus) distance, meters.
        wav: Array of wavelengths, meters.
        sp: Spectral response per wavelength.
        drho: Relative ring half-width (1.5).
        pixel: CCD plate scale, arcsec/pixel (0 = no pixel averaging).
        zn: Noll indices of static Zernike aberrations ([]).
        zrad: Amplitudes of those Zernikes, radians ([]).
        nsect: Number of angular sectors (8).
        verb: Print text to the CLI (True).
    
    Returns:
        (wt, ufunc, ringrad): the (nz, mmax+1) weight table, the
        matching U-function table, and the ring radius in CCD pixels.
    '''
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
    
    # Define main calculation parameters, grid and annular aperture
    size, asperpix, fstep, r, phi, pupil, ninside, radpix = weight_grid(d, eps, lam0, ngrid, ksize)
    
    # Define conic wavefront at the pupil (optionally with static Zernikes)
    uampl = conic_wavefront(r, phi, radpix, lam0, d, pdist, zn, zrad, pupil)
    
    # Compute nominal ring image at the focal plane
    imh = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(uampl)))
    imh = np.power(np.abs(imh),2)/( (2 * ngrid)**2) # np.sum(imh) = ninside to check normalization
    
    # Compute the masks
    (filtap, ringmask, rwt, flux, rflux, 
     ringrad, ringradpix2, spturb, spufunc, utmp) = ring_mask(imh, uampl, r, phi, ninside, asperpix, pixel, lam0,
                                                              d, eps, drho, fstep, nsect, ngrid, ksize, verb)
    
    nm = mmax + 1  # number of angular coefficients
    wt = np.zeros((nz,nm))
    ufunc = np.zeros((nz,nm))
    
    # Prepare 2D Fresnel filters for propagation calculation
    frecos, fresin = fresnel_filters(z, r, fstep, wav, spnorm, lam0, wavstep, nwav, nz, ngrid)
    
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


def computeweight(par, verb=True):  # actual weight calculation
    '''
    Complete weight processing based on expected weights
    and calculated weights from telescope parameters.
    
    Unpacks the parameters, builds the two star-color spectra and
    distance grid, calibrates the conjugation distance, runs aweight
    for B-V=0 and B-V=1, derives the wind coefficients and
    color slopes, then serializes everything to the weight file.
    
    Args:
        par: Telescope/profrest parameter dict (from statmom).
        verb: Print text to the CLI (True).
    
    Returns:
        The weight dict (z, wt0, wtslope, ucoef0, ucoefslope, umm,
        lameff, ringrad, pdist), also saved to the weightfile.
    '''
    # Unpack telescope/profrest parameters and optional aberrations
    d, eps, pdist, pixscale, ringradpix, wav, sp, mmax, nsect, zn, zrad = telescope_parameters(par, verb)
    
    # Color spectra and effective wavelengths
    sp0, sp1, lameff = color_spectrum(wav, sp)
    
    # distance grid, log-spaced with sqrt(2) step, 0.25-32km
    z = distance_grid()
    
    # Find propagation distance from ring radius
    pdist = correct_pdist(d, eps, pixscale, ringradpix, pdist, wav, sp0, zn, zrad, nsect, verb)
    
    # Weight for B-V=0
    if verb: print('\n')
    if verb: print('Computing weight for B-V=0...')
    wt0, ufunc0, ringrad = aweight(z,mmax,d,eps,pdist,wav,sp0,1.5,pixscale,zn,zrad,nsect,verb) # arrays of [nz,mmax+1] dimension
    hslope = pdist * ringrad
    if verb: print(f'Ring radius and H*rad [m.pix]: {ringrad}, {hslope}')
    # Weight for B-V=1
    if verb: print('Computing weight for B-V=1...')
    wt1, ufunc1, ringrad1 = aweight(z,mmax,d,eps,pdist,wav,sp1,1.5,pixscale,zn,zrad,nsect,verb) # arrays of [nz,mmax+1] dimension
    
    mm = [1,3,6,7,8,9] # selected frequencies for wind measurement
    ucoef0, resp0 = getucoef(ufunc0,z,mm)
    ucoef1, resp1 = getucoef(ufunc1,z,mm)
    
    # Color dependence
    wtslope = wt1 - wt0
    ucoefslope = ucoef1 - ucoef0
    
    # Serialize and save the weights
    weight = {
        'z':          z.tolist(),
        'wt0':        wt0.tolist(),
        'wtslope':    wtslope.tolist(),
        'ucoef0':     ucoef0.tolist(),
        'ucoefslope': ucoefslope.tolist(),
        'umm':        mm,
        'lameff':     lameff,
        'ringrad':    ringrad,
        'pdist':      pdist
    }
    
    json_output_file = par['profrest']['weightfile']
    try:
        json.dump(weight, codecs.open(json_output_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    except FileNotFoundError as err:
        if verb: print('{}:{}'.format(err, json_output_file))
    if verb: print('Saved weights in '+json_output_file)
    return weight


# Used functions ---------------------------------------------------------------------------

def weight_grid(d, eps, lam0, ngrid, ksize):
    '''
    Builds the pupil-plane grid and the boolean annular aperture.
    
    Args:
        d: Aperture diameter, meters.
        eps: Central obscuration fraction.
        lam0: Mean wavelength, meters.
        ngrid: Half-size of the computing grid, pixels.
        ksize: Grid-size / telescope-diameter ratio.
    
    Returns:
        (size, asperpix, fstep, r, phi, pupil, ninside, radpix): the
        domain size [m], fine plate scale [arcsec], frequency step,
        radial and azimuthal grids, pupil mask, pupil area [pix], and
        aperture radius [pix].
    '''
    # Define main calculation parameters
    size = d * ksize  # domain size in pupil plane, [m]
    asperpix = 206265 * lam0 / size  # fine-pixel in the image plane [arcsec]
    fstep = 1 / size  # frequency step [1/m]
    
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
    return size, asperpix, fstep, r, phi, pupil, ninside, radpix


def conic_wavefront(r, phi, radpix, lam0, d, pdist, zn, zrad, pupil):
    '''
    Builds the defocused pupil (conic wavefront) 
    complex amplitude for the given conjugation distance, 
    optionally adds static Zernike aberrations.
    
    Args:
        r: 2D radial-distance grid, pixels.
        phi: 2D azimuthal-angle grid, radians.
        radpix: Aperture radius, pixels.
        lam0: Mean wavelength, meters.
        d: Aperture diameter, meters.
        pdist: Conjugation (defocus) distance, meters.
        zn: Noll indices of static Zernike aberrations.
        zrad: Amplitudes of those Zernikes, radians.
        pupil: Boolean annular pupil mask.
    
    Returns:
        The complex amplitude at the pupil (uampl).
    '''
    # Define conic wavefront at the pupil
    a4 = d**2 / (lam0 * pdist) * (np.pi / 8 * 3**(-0.5))   # Zernike defocus corresponding to the propagation distance [rad]
    a11 = -0.1 * a4       # spherical aberration coef. [rad]
    tmp = a11 * 5**(0.5) * (6 * (r/radpix)**4  - 6 * (r/radpix)**2)
    tmp = tmp + a4 * 2 * 3**0.5 * ((r/radpix)**2 - 0.5)  # wavefront shape [rad]
    
    # Optionally add Zernike aberrations
    tmp = zernike(tmp, zn, zrad, r, radpix, phi)
    
    uampl = pupil*(np.cos(tmp) + np.sin(tmp)*1j) # complex amplitude at the pupil
    return uampl


def zernike(tmp, zn, zrad, r, radpix, phi):
    '''
    Adds static Zernike aberrations onto a wavefront.
    
    Args:
        tmp: Wavefront map to add the aberrations to, radians.
        zn: Noll indices of the Zernike modes to add.
        zrad: Amplitude of each mode, radians.
        r: 2D radial-distance grid, pixels.
        radpix: Aperture radius, pixels (normalizes r).
        phi: 2D azimuthal-angle grid, radians.
    
    Returns:
        The wavefront map with the aberrations added.
    '''
    nzern = len(zn)
    for j in range(0,nzern):
        tmp += zrad[j]*zernikel(zn[j],r/radpix,phi)
    return tmp


def ring_mask(imh, uampl, r, phi, ninside, asperpix, pixel, lam0, d, eps, drho,
              fstep, nsect, ngrid, ksize, verb=True):
    '''
    Builds the ring/sector masks and the turbulence spectra
    measuring the true ring radius, and prepares the
    turbulence and U-function spectra plus the conjugated amplitude
    used in the weight loop.
    
    Args:
        imh: Nominal ring image at the focal plane.
        uampl: Pupil complex amplitude (from conic_wavefront).
        r: 2D radial-distance grid, pixels.
        phi: 2D azimuthal-angle grid, radians.
        ninside: Pupil area, pixels.
        asperpix: Fine plate scale, arcsec/pixel.
        pixel: CCD plate scale, arcsec/pixel.
        lam0: Mean wavelength, meters.
        d: Aperture diameter, meters.
        eps: Central obscuration fraction.
        drho: Relative ring half-width.
        fstep: Frequency step, 1/m.
        nsect: Number of angular sectors.
        ngrid: Half-size of the computing grid, pixels.
        ksize: Grid-size / telescope-diameter ratio.
        verb: Print text to the CLI (True).
    
    Returns:
        (filtap, ringmask, rwt, flux, rflux, ringrad, ringradpix2,
        spturb, spufunc, utmp): the radial ring mask, sector ring
        mask, sector-motion radial weight, ring flux, sector flux,
        ring radius [CCD pix] and [fine pix], turbulence and
        U-function spectra, and the conjugated pupil amplitude.
    '''
    # Compute the masks
    ringradpix2 = np.sum(imh * r) / ninside # true ring radius in fine pixels
    ringrad = ringradpix2 * asperpix / pixel  # ring radius in CCD pixels
    if verb: print ('Ring radius [pix]: ',ringrad)
    drhopix = drho * lam0 / d / (1 - eps) * 2 * 206265 / asperpix # ring half-width [pix]
    filtap = (r >= ringradpix2 - drhopix) * (r <= ringradpix2 + drhopix) # radial part of image mask
    
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
    return filtap, ringmask, rwt, flux, rflux, ringrad, ringradpix2, spturb, spufunc, utmp


def fresnel_filters(z, r, fstep, wav, spnorm, lam0, wavstep, nwav, nz, ngrid):
    '''
    Builds the 2D Fresnel propagation for each layer distance z, 
    it sums the cosine/sine propagation kernels over the wavelength 
    band (with a bandwidth damping term) to give the cos and 
    sin Fresnel filters used in the weight loop.
    
    Args:
        z: Propagation-distance grid, meters.
        r: 2D radial-distance grid, pixels.
        fstep: Frequency step, 1/m.
        wav: Array of wavelengths, meters.
        spnorm: Normalized spectral response per wavelength.
        lam0: Mean wavelength, meters.
        wavstep: Wavelength-grid step, meters (0 if single wavelength).
        nwav: Number of wavelengths.
        nz: Number of layers.
        ngrid: Half-size of the computing grid, pixels.
    
    Returns:
        (frecos, fresin): the per-layer cosine and sine Fresnel
        filter stacks, each (nz, 2*ngrid, 2*ngrid).
    '''
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
    return frecos, fresin


def blackbody(wav,temp):
    '''
    Returns a blackbody spectrum with arbitrary normalization.
    
    Args:
        wav: Array of wavelengths, meters.
        temp: Blackbody temperature, kelvin.
    
    Returns:
        The normalized photon spectrum over wav (peak = 1).
    '''
    const = 0.014387618 # in [m.K]
    planck = np.power(wav[0] / wav,4) / (np.exp(const / wav / temp) -1)
    return planck / np.max(planck)


def getucoef(ufunc,z,mm):
    '''
    Finds the U-coefficients for the wind measurement.
    Solves a least-squares system (via SVD pseudo-inverse) so that
    the weighted combination of U-functions at the selected modes mm
    gives a flat, near-unity response across the layers above the
    two lowest.
    
    Args:
        ufunc: (nz, nm) U-function table (from aweight).
        z: Distance grid, meters.
        mm: Selected angular modes used for the wind estimate.
    
    Returns:
        (ucoef, uresp): the U-coefficients and the resulting response
        curve (should be close to 1 at all z > 0).
    '''
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


def getpar(parfile):
    '''
    Reads a JSON parameter file and returns the par dictionary.
    
    Args:
        parfile: Path to the JSON par file.

    Returns:
        The parsed nested par dict. Exits the program if the file is
        not found.
    '''
    try:
        file = open(parfile, 'r')
        par = json.load(file)   # par is a nested dictionary
    except FileNotFoundError as err:
        print(err)
        quit()
    file.close()
    return par


def telescope_parameters(par, verb=True):
    '''
    Unpacks the telescope/profrest parameters from the par dict.
    
    Args:
        par: Telescope/profrest parameter dict.
        verb: Print text to the CLI (True).
    
    Returns:
        (d, eps, pdist, pixscale, ringradpix, wav, sp, mmax, nsect,
        zn, zrad): the unpacked parameters, with zn/zrad empty when
        no aberration file is given.
    '''
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
        if verb: print(s)
        #print(zrad)
    else:
         zn = zrad = []
    return d, eps, pdist, pixscale, ringradpix, wav, sp, mmax, nsect, zn, zrad


def color_spectrum(wav, sp):
    '''
    Builds the two star-color spectra and effective wavelengths.
    
    Args:
        wav: Array of wavelengths, meters.
        sp: Instrument spectral response per wavelength.
    
    Returns:
        (sp0, sp1, lameff): the normalized B-V=0 and B-V=1 spectra
        and the (lam0_bv0, lam0_bv1) effective wavelengths.
    '''
    sp0 = sp * blackbody(wav, 10213) # B-V=0 spectrum
    sp1 = sp * blackbody(wav, 3938) # B-V=1 spectrum
    sp0 = sp0 / np.sum(sp0) # normalize
    sp1 = sp1 / np.sum(sp1)
    lameff = (np.sum(wav * sp0), np.sum(wav * sp1)) # effective wavelength
    return sp0, sp1, lameff


def distance_grid():
    '''
    Builds the layer-distance grid for the weights.
    
    Returns:
        The length-16 distance grid, meters.
    '''
    # distance grid, log-spaced with sqrt(2) step, 0.25-32km
    nz = 16
    z = np.zeros(nz)
    z[1:nz] = 1e3 * 2**(0.5 * np.arange(nz-1) -2)
    return z


def correct_pdist(d, eps, pixscale, ringradpix, pdist, wav, sp0, zn, zrad, nsect, verb=True):
    '''
    Calibrates the conjugation distance from the ring radius.
    
    Starts from the analytic H*R (alpha in literature) 
    estimate, runs a single-layer aweight to get the ring 
    radius, then rescales pdist so the modeled and 
    measured ring radii agree.
    
    Args:
        d: Aperture diameter, meters.
        eps: Central obscuration fraction.
        pixscale: CCD plate scale, arcsec/pixel.
        ringradpix: Measured ring radius, pixels.
        pdist: Initial conjugation distance, meters.
        wav: Array of wavelengths, meters.
        sp0: Normalized B-V=0 spectrum.
        zn: Noll indices of static Zernike aberrations.
        zrad: Amplitudes of those Zernikes, radians.
        nsect: Number of angular sectors.
        verb: Print text to the CLI (True).
    
    Returns:
        The calibrated conjugation distance (H), meters.
    '''
    # Find propagation distance from ring radius. Use analytic approx. first
    Rmean = d * (1 + eps) / 4 # Mean of annulus radius
    HR = 0.85 * Rmean / pixscale * 206265 # H * R (proxy for alpha in papers)
    if verb: print(f'Rmean = {d} * (1 + {eps}) / 4')
    if verb: print(f'Initial H*R [m.pix] and Mean radius: {HR}, {Rmean}')
    
    pdist1 = HR / ringradpix
    wt0, ufunc0, ringrad = aweight(np.zeros(1),1,d,eps,pdist,wav,sp0,1.5,pixscale,zn,zrad,nsect,verb)
    
    HR1 = ringrad * pdist # Adjust
    if verb: print(f'Expected Conjugation dist.: {pdist}')
    final_pdist = HR1 / ringradpix
    if verb: print(f'Calculated Conjugation dist.: {pdist1}')
    if verb: print(f'Adjusted H*R [m.pix] and pdist: {HR1}, {final_pdist}')
    return final_pdist


if __name__ == '__main__':
    main()