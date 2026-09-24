#!/usr/bin/env python3
'''
Simulates ring images and creates fits file
Includes ringsim.pro and writefits.pro
Currently does not support debugging
Images should be saved and/or displayed


Auth: A. Tokovinin
Translated: D. Hurtado

'''
import argparse
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation  # noqa: F401  (needed so mpl.animation.PillowWriter resolves)
import numpy as np

from astropy.io import fits
import datetime
from scipy.ndimage import zoom, shift as ndshift



def main():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('d', type=float,
                   help='Diameter of telescope M1 in meters')
    p.add_argument('effl', type=float,
                   help='Effective focal length')
    p.add_argument('eps', type=float,
                   help='Fraction of central obscuration')
    p.add_argument('pdist', type=float,
                   help='Conjugation distance in meters')
    p.add_argument('pixsize', type=float,
                   help='Camera pixel size in meters')
    p.add_argument('--texp', dest='texp', type=float, default=1e-3, 
                   help='Seconds of exposure time, default=1e-3')
    p.add_argument('--tacc', dest='tacc', type=float, default=1, 
                   help='Seconds of acquisition, default=1')
    p.add_argument('--ron', dest='ron', type=float, default=0, 
                   help='Read out noise in electrons, default=0')
    p.add_argument('--gain', dest='gain', type=float, default=0, 
                   help='Camera decibels(?) gain setting, default=0')
    p.add_argument('--starmag', dest='starmag', type=float, default=1, 
                   help='Magnitude of the star, default=2')
    p.add_argument('--wind', dest='wind', type=float, default=10, 
                   help='Windspeed in m/s, default=10') # This apparently is not being well calculated
    p.add_argument('--jitter', dest='jitter', type=float, default=0, 
                   help='Amount of jitter, unit TBD, default=0')
    p.add_argument('--oversample', dest='oversamp', type=bool, default=True, 
                   help="Decides if there's oversampling, boolean, default=True")
    p.add_argument('--blur', dest='blur', type=bool, default=True, 
                   help="Decides if there's blurring, boolean, default=True")
    p.add_argument('--display', dest='display', type=bool, default=True,
                   help='Decides if images are displayed, boolean default=True')
    
    # Hard-coded Parameters, these change with args.parse
    args     = p.parse_args()
    d        = args.d                   # meters, mirror diameter
    effl     = args.effl             # meters, effective focal length
    eps      = args.eps               # fraction, central obscuration 
    pdist    = args.pdist           # meters, H (Conjugation distance)
    pixsize  = args.pixsize       # meters, CCD pixel size
    texp     = args.texp             # seconds, exposure time
    starmag  = args.starmag       # star magnitude
    ron      = args.ron               # electrons, read out noise 
    gain     = args.gain             # db, camera gain
    tacc     = args.tacc             # seconds, accumulation time
    texp     = args.texp             # seconds, exposition time
    wind     = args.wind             # m/s, wind speed
    jitter   = args.jitter
    oversamp = args.oversamp     # check to oversample
    blur     = args.blur             # check for blurring
    display  = args.display       # check for image display
    
    rng = np.random.default_rng(seed0)
        
    with np.load('atm.npz') as data:
        u1        = data['u1']
        ngrid     = data['ngrid']
        pixel     = data['pixel']
        wavelen   = data['wavelen']
        see       = data['see']
        r0        = data['r0']
        fhigh     = data['fhigh']
        zlow      = data['zlow']
        zhigh     = data['zhigh']
        seed0     = int(data['seed0'])
    
    # Apertures move over the screen mostly in x-direction, but slide
    # in y-direction by SLIDE meters par grid length
    size  = 2 * ngrid * pixel                          # grid size in arcseconds?
    slide = 0.205
    alpha = slide / size                              # tangent of slide angle
    
    pixscale =  (pixsize) / effl * 206265
    print(f'Calculated pixel scale', pixscale)
    
    print(f'Loaded {data}')
    print('starmag, ron, d, eps, pdist, wind, size, ngrid')
    print(starmag, ron, d, eps, pdist, wind, size, ngrid)
    
    
    if blur:
        nblur = math.floor(wind * texp / pixel + 0.5)
            # ^math.floor^ returns integer like int(np.floor())
        print(f'Averaging atmospheric screens, N={nblur}')
        if nblur > 1:
            tmp = np.copy(u1)
            for k in range(1, nblur):
                tmp += np.roll(u1, shift=k, axis=1)  # Shift in x-direction
            u1 = tmp / nblur
    
    if 'zlow' not in locals() or zlow is None:
        zlow = zhigh
    
    # Calculate seeing in arcseconds (206265 converts radians to arcseconds)
    seeing = 0.98 * wavelen / r0 * 206265
    print(f'Seeing (arcsec): {round(seeing,5)} arcsec \n Layers at [{zlow}, {zhigh}] meters with high fraction {fhigh}')
    print(f'Screen size (meters): {2 * ngrid * pixel} \n Pixel size (meters): {pixel}')
    
    # Total turbulence integral J (m^(1/3))
    tint = (r0 ** (-5 / 3)) / 0.423 * ((0.5 * wavelen / np.pi) ** 2)
    print(f'Input r0 (meters): {r0}, J: {tint}')
    
    niter = math.floor(tacc / texp)           # Steps
    jstep = math.floor(wind * texp / pixel + 0.5) # Integer pixel shift
    if jstep < 1:
        jstep = 1
    windef = jstep * pixel / texp
    
    
    print(f'Screen shift per exposure: {jstep} pixels')
    print(f'Effective wind speed: {windef} m/s')
    print(f'Total iterations to simulate: {niter}\n')
    
    # Define grid size and oversampling
    nap = math.floor(ngrid / 2)
    print(f'Aperture grid, pixels: {nap}')
    
    d1 = wavelen / pixscale * 206265 # Pupil match pixels
    print(f'd1: {d1}')
    d2 = wavelen / pixel * 206265    # Pupil match pixels
    print(f'd2: {d2}')
    
    if oversamp: 
        npixperpix = 2 ** (math.floor(math.log2(1.5 * d / d1)) + 1) # Oversample
    else:
        npixperpix = 1
    print(f'Pixel per pixel: {npixperpix}')
    
    nscr = 2 ** (math.floor(math.log2(1.5 * d / pixel)) + 1)            # Screen size in pixels
    print(f'Screen size in pixels: {nscr}')
    
    ksamp = max(int(nap // nscr), 1)
    print(f'Over-sampling factor: {ksamp}')
    
    
    # Pixel and ring sizes
    # `pixel` is the phase-screen / pupil-plane grid sampling in METERS per grid pixel
    # (its value sets the physical screen size, it is NOT a plate scale). The image-plane
    # plate scale in arcsec/pixel is a DERIVED quantity: after zooming the pupil sub-window
    # the pupil sampling is pixel/ksamp [m] over nap grid pixels, so the fine-image scale is
    # 206265*wavelen/(nap*pixel/ksamp) and the binned-CCD scale multiplies that by npixperpix.
    # Fine focal-plane pixel scale [arcsec], set purely by the pupil sampling.
    
    finepix = 206265 * wavelen / (nap * pixel / ksamp)
    asperpix = pixscale                 # detector plate scale = user input [arcsec/pix]
    ccdbin = asperpix / finepix         # fine pixels per CCD pixel
    nccd = int(round(nap / ccdbin))     # pixels, detector size
    rradiuspix = 0.85 * d * (1 + eps) / (4 * pdist) * 206265 / asperpix
    HR = rradiuspix * pdist
    
    print(f'Re-sampled pixel size [m]: {pixel / ksamp:.6e}')
    print(f'CCD size & pixel [arcsec]: {nccd}, {asperpix}')
    print(f'Nominal ring radius [pix, arcsec]: {rradiuspix:.3f}, {rradiuspix * asperpix:.3f}')
    print(f'HR [unit?]: {HR}')
    
    # Warn only if the fine grid is coarser than the detector (under-sampled sim).
    if ccdbin < 1.0:
        print(f'Warning: fine grid {finepix:.4f} arcsec/pix is coarser than the '
              f'detector {asperpix:.4f} arcsec/pix; raise oversampling for a finer sim.')
    
    # Circular image shifts
    if jitter > 0:
        omega = (3.3 / niter) * (2 * np.pi)  # angular frequency
    
    # Star (If statement would've crashed the script if starmag=0)
    BW = 0.26                      # effective bandwidth
    phot_con = 1e11                # constant representing photons/sec/m^2 for a Mag 0 star at the top of the atmosphere
    starph = phot_con * texp * BW * 10**(-0.4 * starmag) * np.pi * (d/2)**2 * (1 - eps)**2 
        
    print(f'Stellar photons per exposure and Star Magnitude: {round(starph,2),starmag}')
    print(f'Readout noise (e-): {ron}')
    print(f'Jitter = {jitter}')
    
    # Prepare the aperture mask
    apert = np.zeros((nap, nap))
    y, x = np.ogrid[:nap, :nap]
    r = np.hypot(x - nap / 2, y - nap / 2) # Radial distance in pixels
    radpix = d * 0.5 / pixel * ksamp
    inside = (r <= radpix) & (r >= radpix * eps)
    apert[inside] = 1
    # apert[0 : nap // 2, nap // 2 - 10 : nap // 2 + 10] = 0 # Optional sector mask test
    
    if display:
        print(f'radpix: ({d} * 0.5) / {pixel}) * {ksamp} = {radpix} \n n inside: {np.sum(apert)}')
        plt.figure(figsize=(6, 6))
        plt.imshow(apert, cmap='gray', origin='lower')
        plt.title(f'Debug Aperture Mask (nap={nap}, radpix={round(radpix,2)})')
        plt.colorbar(ticks = (0,1))
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.show()
        plt.clf
    
    # Add defocus and spherical, a4 negative for intrafocal
    a4 = (d ** 2 / (wavelen * pdist)) * (np.pi / (8 * np.sqrt(3)))
    a11 = -0.1 * a4  # Matching spherical aberration
    rho = r / radpix
    
    tmp = a11 * np.sqrt(5) * (6.0 * (rho**4) - 6 * (rho**2))
    tmp += a4 * 2.0 * np.sqrt(3) * ((rho**2) - 0.5)
    
    print(f'Nominal a4, a11 [rad]: {a4:}, {a11}')
    
    if display:
        plt.figure(figsize=(6, 6))
        plt.imshow(tmp, cmap='gray', origin='lower', norm=colors.LogNorm())
        plt.title(f'Debug Zernike applied (nap={nap}, radpix={round(radpix/ksamp, 2)})')
        plt.colorbar()
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.show()
        plt.clf
    
    # Undistorted image
    fresnel = np.exp(1j * tmp) * apert
    
    # Mirror the correct shift order for consistency
    pupil_shifted = np.fft.ifftshift(fresnel)
    imh0_complex = np.fft.ifft2(pupil_shifted) * (nap**2)
    focus_centered = np.fft.fftshift(imh0_complex)
    imh0 = np.abs(focus_centered) ** 2
    
    if display:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.abs(fresnel), cmap='gray', origin='lower')
        plt.title(f'Debug at fresnel')
        plt.colorbar()
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.show()
        plt.clf
    
        plt.figure(figsize=(6, 6))
        plt.imshow(np.abs(pupil_shifted), cmap='gray', origin='lower')
        plt.title(f'Debug at pupil_shifted')
        plt.colorbar()
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.show()
        plt.clf
    
        plt.figure(figsize=(6, 6))
        plt.imshow(np.abs(imh0_complex), cmap='gray', origin='lower')
        plt.title(f'Debug at imh0_complex')
        plt.colorbar()
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.show()
        plt.clf
    
        plt.figure(figsize=(6, 6))
        plt.imshow(np.abs(focus_centered), cmap='gray', origin='lower')
        plt.title(f'Debug at focus_centered')
        plt.colorbar()
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.show()
        plt.clf
        
        plt.figure(figsize=(6, 6))
        plt.imshow(np.abs(imh0), cmap='gray', origin='lower')
        plt.title(f'Debug at imh0')
        plt.colorbar()
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.show()
        plt.clf
    
    # Intensity normalization
    normconst = np.sum(imh0)
    
    # Calculate expected radius using clean centered coordinates
    rring = np.sum(imh0 * r) / normconst                             
    rradpix2 = np.sum(imh0*r) / np.sum(imh0) # true ring radius in fine pixels
    rad = rradpix2 / ccdbin  # radius in CCD pixels (CHANGED: ccdbin, not npixperpix)
    
    print(f'True ring radius [pix]: {rring}')
    print(f'True ring radius [arcsec]: {(rring * asperpix)}')
    print(f'rring / ksamp {rring / ksamp}')
    
    # Cube for loop
    cube = np.zeros((niter, nccd, nccd), dtype=np.float64)
    ix = 0
    iy_float = 0
    
    # Progress display step
    ndispl = max(niter // 20, 1)
    print(f'Computing {niter} iterations...')
    
    # Gif creator
    writer = mpl.animation.PillowWriter(fps=5)
    tmp_fig = plt.figure()
    writer.setup(tmp_fig, 'ringsim.gif', dpi=100)
    plt.close(tmp_fig)
    
    # Main Loop
    for i in range(niter):
        ix += jstep
        iy_float += jstep * alpha
        iy = math.floor(iy_float)
        
        # Roll if we surpass the center of the array
        if ix > ngrid:
            u1 = np.roll(u1, -ngrid, axis=1)
            ix -= ngrid
        if iy > ngrid:
            u1 = np.roll(u1, -ngrid, axis=0)
            iy_float -= ngrid
            iy = math.floor(iy_float)
        
        uampl = u1[iy : iy+nscr, ix : ix+nscr]
        
        if ksamp > 1:
            zoom_frac = nap / nscr
            uampl_re = zoom(uampl.real, zoom_frac, order=1)
            uampl_im = zoom(uampl.imag, zoom_frac, order=1)
            uampl = uampl_re + 1j * uampl_im
        
        fresnel_uampl = fresnel * uampl
        
        pupil_shifted = np.fft.ifftshift(fresnel_uampl)
        imh1_complex = np.fft.ifft2(pupil_shifted) * (nap**2)
        focus_centered = np.fft.fftshift(imh0_complex if 'imh0_complex' in locals() else imh1_complex)
        imh1 = np.abs(np.fft.fftshift(imh1_complex))**2
        
        # Block-sum (rebin) from fine grid (nap x nap) down to CCD resolution (nccd x nccd)
        # CHANGED: resample fine image (nap) to detector grid (nccd) at exactly asperpix.
        # Flux-conserving: *(nap/nccd)**2 restores the total the old block-sum gave.
        impix = zoom(imh1, nccd / nap, order=1) * (nap / nccd) ** 2 / normconst
        
        # Optional Jitter / Image Shifts
        if jitter > 0:
            xc = jitter * np.cos(i * omega)
            yc = jitter * np.sin(i * omega)
            # Note: ndshift takes (y_shift, x_shift)
            impix = ndshift(impix, shift=(yc, xc), order=1, mode='nearest')
        
        # --- Photon and Readout Noise (Inside Loop) ---
        if 'starmag' in locals() and starmag != 0.0:
            impix *= starph
            # Ensure non-negative input for Poisson noise generator
            impix_clean = np.maximum(0.0, impix)
            impix = rng.poisson(impix_clean).astype(np.float64) + rng.normal(loc=0.0, scale=ron, size=(nccd, nccd))
        
        # Pupil image for diagnostics
        imh2 = np.abs(uampl * apert)**2
        
        # Save frame to 3D image cube
        cube[i, :, :] = impix
        
        if i % ndispl == 0:
            print(f'Frame {i}/{niter}')
            
            #clear_output(wait=True)  # Clears previous frame before rendering the new one
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
            
            # Left: Detector Plane
            h1, w1 = impix.shape
            im1 = ax1.imshow(impix, cmap='gray', origin='lower',
                            extent=[-w1/2, w1/2, -h1/2, h1/2])
            ax1.set_title(f'Detector Plane (impix) - Frame {i}')
            ax1.set_xlabel('X [pixels]')
            ax1.set_ylabel('Y [pixels]')
            #fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # Right: Pupil Plane
            h2, w2 = imh2.shape
            im2 = ax2.imshow(imh2, cmap='gray', origin='lower',
                            extent=[-w2/2, w2/2, -h2/2, h2/2])
            ax2.set_title('Pupil Plane (imh2)')
            ax2.set_xlabel('X [pixels]')
            ax2.set_ylabel('Y [pixels]')
            #fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Gif input
            writer.fig = fig  # Attach current figure instance to writer
            writer.grab_frame()  # Capture frame into GIF buffer
            plt.show()
            plt.clf
    
    print('Simulation done!')
    writer.finish()  # Compile output.gif
    print('Gif saved')
    
    imav = np.mean(cube, axis=0)
    
    if display:
        plt.figure(figsize=(6, 6))
        plt.imshow(imav, cmap='gray', origin='lower')
        plt.title(f'Debug image average')
        #plt.colorbar()
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        plt.show()
        plt.clf
    
        print(f'nap {nap} \n nccd {nccd} \n npixperpix {npixperpix} \n radpix {radpix}')
    
    hdu = fits.PrimaryHDU(cube.astype('float64'))
    
    header = hdu.header
    header['DATE-OBS'] = (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'File creation date / Observation Date')
    header['DIAM'] = (float(d), 'Telescope Diameter in meters')
    header['EFFL'] = (float(effl), 'Telescope focal length in meters')
    header['OBSC'] = (float(eps), 'Central obscuration (eps)')
    header['CONJ_H'] = (float(pdist), 'Conjugation height in meters')
    header['WAVELEN'] = (float(wavelen), 'Wavelength in meters')
    header['EXPOSURE'] = (float(texp*1e6), 'Exposure time in seconds')
    header['MAG'] = (float(starmag), 'Stellar magnitude')
    header['SEEING'] = (float(seeing), 'Input seeing in arcseconds')
    header['CAM'] = (int(nccd), 'Camera dimensions')
    header['PIXGRID'] = (int(pixel), 'Input grid pixel screen size')
    header['PIXSCALE'] = (float(asperpix), 'Arcseconds per pixel')
    # Keys expected by cube2.py
    # Note: cube2.py reads EXPOSURE as microseconds (multiplies by 1e-6)
    header['GAIN'] = (float(gain), 'Detector gain (cube2.py reads as float)')
    header['STAR'] = ('', 'Star name (optional in cube2.py)')
    
    # mmax = args.mmax
    # nsect = args.nsect
    # drad = args.drad
    # interpol = args.interpol
    
    filename = 'test.fits'
    hdu.writeto(filename, overwrite=True)
    
    print(f'Successfully saved {niter} frames to {filename}')

if __name__ == '__main__':
    main()