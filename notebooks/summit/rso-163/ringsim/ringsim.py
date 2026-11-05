#!/usr/bin/env python3
'''
Simulates ring images by propagating starlight through the
atmosphere previously created, then outputs into
a three-axis fits file.

Part of the code translated from IDL to Python

Usage:
    python ringsim.py --help          # list all arguments and defaults
    python ringsim.py 0.304 1.20845 0.7 1050 6.9e-6
    import ringsim
    ringsim.ringsim(0.304, 1.20845, 0.7, 1050, 6.9e-6)
    help(ringsim.ringsim)             # docstring for any function

Auth: A. Tokovinin
Translated: D. Hurtado

'''
import argparse
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import numpy as np
import os
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
    p.add_argument('--verbose', dest='verb', type=bool, default=True,
                   help='Print text into CLI, def=True')

    # Hard-coded Parameters, these change with args.parse
    args     = p.parse_args()
    d        = args.d             # meters, mirror diameter
    effl     = args.effl          # meters, effective focal length
    eps      = args.eps           # fraction, central obscuration
    pdist    = args.pdist         # meters, H (Conjugation distance)
    pixsize  = args.pixsize       # meters, CCD pixel size
    texp     = args.texp          # seconds, exposure time
    starmag  = args.starmag       # star magnitude
    ron      = args.ron           # electrons, read out noise
    gain     = args.gain          # db, camera gain
    tacc     = args.tacc          # seconds, accumulation time
    texp     = args.texp          # seconds, exposition time
    wind     = args.wind          # m/s, wind speed
    jitter   = args.jitter
    oversamp = args.oversamp      # check to oversample
    blur     = args.blur          # check for blurring
    display  = args.display       # check for image display
    verb     = args.verb          # check for CLI text output

    return ringsim(d, effl, eps, pdist, pixsize, texp, tacc, ron, gain,
                   starmag, wind, jitter, oversamp, blur, display, verb)


def ringsim(d, effl, eps, pdist, pixsize, texp=1e-3, tacc=1, ron=0, gain=0,
            starmag=1, wind=10, jitter=0, oversamp=True, blur=True,
            display=True, verb=True):

    '''Simulate a ring-image cube and write it to a FITS file.

    Work-horse of the script: loads the atmosphere field from
    simatm.py, propagates the starlight through it frame by frame to
    form the defocused ring images, adds detector noise and saves the
    resulting cube to a FITS file (returned as a path).

    Args:
        d: Diameter of telescope M1, meters.
        effl: Effective focal length, meters.
        eps: Fraction of central obscuration.
        pdist: Conjugation distance, meters.
        pixsize: Camera pixel size, meters.
        texp: Exposure time, seconds (1e-3).
        tacc: Acquisition time, seconds (1).
        ron: Read-out noise, electrons (0).
        gain: Camera gain setting (0).
        starmag: Star magnitude (1).
        wind: Wind speed, m/s (10).
        jitter: Amount of jitter, unit TBD (0).
        oversamp: Whether to oversample (True).
        blur: Whether to blur by averaging screens (True).
        display: Save displayed images to the images folder (True).
        verb: Print text to the CLI (True).

    Returns:
        Path to the written FITS cube ('test.fits').
    '''
    with np.load('atm.npz') as data:
        screen        = data['screen']
        ngrid     = data['ngrid']
        pixel     = data['pixel']
        wavelen   = data['wavelen']
        see       = data['see']
        r0        = data['r0']
        fhigh     = data['fhigh']
        zlow      = data['zlow']
        zhigh     = data['zhigh']
        seed0     = int(data['seed0'])

    rng = np.random.default_rng(seed0)

    # Apertures move over the screen mostly in x-direction, but slide
    # in y-direction by SLIDE meters par grid length
    if blur:
        nblur = math.floor(wind * texp / pixel + 0.5)
            # ^math.floor^ returns integer like int(np.floor())
        if verb: print(f'Averaging atmospheric screens, N={nblur}')
        screen = average_screens(screen, nblur)

    if zlow is None:
        zlow = zhigh

    geometry = compute_geometry(ngrid, pixel, pixsize, effl, wavelen, r0, wind,
                           texp, tacc, d, eps, pdist, oversamp, verb)
    size       = geometry['size']
    alpha      = geometry['alpha']
    seeing     = geometry['seeing']
    niter      = geometry['niter']
    jstep      = geometry['jstep']
    nap        = geometry['nap']
    npixperpix = geometry['npixperpix']
    nscr       = geometry['nscr']
    ksamp      = geometry['ksamp']
    asperpix   = geometry['asperpix']
    ccdbin     = geometry['ccdbin']
    nccd       = geometry['nccd']
    
    if verb: print(f'Loaded {data}')
    
    if verb: print('starmag, ron, d, eps, pdist, wind, size, ngrid')
    if verb: print(starmag, ron, d, eps, pdist, wind, size, ngrid)
    
    if verb: print(f'Seeing (arcsec): {round(seeing,5)} arcsec \n Layers at [{zlow}, {zhigh}] meters with high fraction {fhigh}')
    if verb: print(f'Screen size (meters): {2 * ngrid * pixel} \n Pixel size (meters): {pixel}')
    
    # Circular image shifts
    if jitter > 0:
        omega = (3.3 / niter) * (2 * np.pi)  # angular frequency
    
    starph = photon_budget(starmag, texp, d, eps)
    if verb: print(f'Stellar photons per exposure and Star Magnitude: {round(starph,2),starmag}')
    if verb: print(f'Readout noise (e-): {ron}')
    if verb: print(f'Jitter = {jitter}')
    
    # Prepare the aperture mask
    apert, r, radpix = make_aperture(nap, d, pixel, ksamp, eps)
    if display:
        print(f'radpix: ({d} * 0.5) / {pixel}) * {ksamp} = {radpix} \n n inside: {np.sum(apert)}')
        save_image(apert, f'Debug Aperture Mask (nap={nap}, radpix={round(radpix,2)})',
                         'aperture_mask.jpg')
    
    # Add defocus and spherical, a4 negative for intrafocal
    tmp, a4, a11 = defocus_phase(d, wavelen, pdist, r, radpix)
    if verb: print(f'Nominal a4, a11 [rad]: {a4:}, {a11}')
    if display:
        save_image(tmp, f'Debug Zernike applied (nap={nap}, radpix={round(radpix/ksamp, 2)})',
                         'zernike.jpg', log=True)
    
    # Undistorted image
    fresnel = np.exp(1j * tmp) * apert
    focus_centered = pupil_to_psf(fresnel, nap)
    imh0 = np.abs(focus_centered) ** 2
    
    if display:
        save_image(np.abs(fresnel), 'Debug at fresnel', 'fresnel.jpg')
        save_image(np.abs(focus_centered), 'Debug at focus_centered', 'focus_centered.jpg')
        save_image(np.abs(imh0), 'Debug at imh0', 'imh0.jpg')
    
    # Intensity normalization
    normconst = np.sum(imh0)
    
    # Calculate expected radius using clean centered coordinates
    rring = np.sum(imh0 * r) / normconst
    rradpix2 = np.sum(imh0*r) / np.sum(imh0) # true ring radius in fine pixels
    rad = rradpix2 / ccdbin  # radius in CCD pixels
    
    if verb: print(f'True ring radius [pix]: {rring}')
    if verb: print(f'True ring radius [arcsec]: {(rring * asperpix)}')
    if verb: print(f'rring / ksamp {rring / ksamp}')
    
    # Cube for loop
    cube = np.zeros((niter, nccd, nccd), dtype=np.float64)
    ix = 0
    iy_float = 0
    
    # Progress display step
    ndispl = max(niter // 20, 1)
    if verb: print(f'Computing {niter} iterations...')
    
    # Gif creator
    if display:
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
            screen = np.roll(screen, -ngrid, axis=1)
            ix -= ngrid
        
        if iy > ngrid:
            screen = np.roll(screen, -ngrid, axis=0)
            iy_float -= ngrid
            iy = math.floor(iy_float)

        uampl = screen[iy : iy+nscr, ix : ix+nscr]
        
        if ksamp > 1:
            zoom_frac = nap / nscr
            uampl_re = zoom(uampl.real, zoom_frac, order=1)
            uampl_im = zoom(uampl.imag, zoom_frac, order=1)
            uampl = uampl_re + 1j * uampl_im
        
        fresnel_uampl = fresnel * uampl
        imh1 = np.abs(pupil_to_psf(fresnel_uampl, nap))**2
        impix = zoom(imh1, nccd / nap, order=1) * (nap / nccd) ** 2 / normconst
        
        # Optional Jitter / Image Shifts
        if jitter > 0:
            xc = jitter * np.cos(i * omega)
            yc = jitter * np.sin(i * omega)
            # Note: ndshift takes (y_shift, x_shift)
            impix = ndshift(impix, shift=(yc, xc), order=1, mode='nearest')
        
        # --- Photon and Readout Noise (Inside Loop) ---
        if starmag != 0.0:
            impix = apply_detector_noise(impix, starph, ron, rng, nccd)
        
        # Pupil image for diagnostics
        imh2 = np.abs(uampl * apert)**2
        
        # Save frame to 3D image cube
        cube[i, :, :] = impix
        
        if i % ndispl == 0 and display:
            if verb: print(f'Frame {i}/{niter}')
            
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
            #plt.show()
            plt.close(fig)  # close per-frame figure so they don't accumulate
            
    if verb: print('Simulation done!')
    if verb and display:
        writer.finish()
        print('Gif saved')
        
    imav = np.mean(cube, axis=0)
    
    if display:
        save_image(imav, 'Debug image average', 'imaverage.jpg', colorbar=False)
        print(f'nap {nap} \n nccd {nccd} \n npixperpix {npixperpix} \n radpix {radpix}')
        
    filename = write_fits_cube(cube, d, effl, eps, pdist, wavelen, texp, ron,
                               gain, starmag, seeing, nccd, pixel, asperpix)
    
    if verb: print(f'Successfully saved {niter} frames to {filename}')
    
    return filename

# Used functions ---------------------------------------------------------------------------

def average_screens(screen, nblur):
    '''
    Averages blur and shifted copies of the screen to mimic the
    motion blur of the atmosphere drifting during each exposure
    '''
    if nblur > 1:
        tmp = np.copy(screen)
        for k in range(1, nblur):
            tmp += np.roll(screen, shift=k, axis=1)
        screen = tmp / nblur
    return screen


def compute_geometry(ngrid, pixel, pixsize, effl, wavelen, r0, wind, texp,
                     tacc, d, eps, pdist, oversamp, verb=True):
    
    '''
    Calculates the geometry of the ring and camera, and defines
    parameters for how the ring will move in the main loop

    Args:
        ngrid: Half-size of the atmosphere grid, pixels.
        pixel: Grid sampling, meters per grid pixel.
        pixsize: Physical detector pixel size, meters.
        effl: Effective focal length, meters.
        wavelen: Reference wavelength, meters.
        r0: Fried parameter, meters.
        wind: Wind speed, m/s.
        texp: Exposure time, seconds.
        tacc: Acquisition time, seconds.
        d: Aperture diameter, meters.
        eps: Fraction of central obscuration.
        pdist: Conjugation distance, meters.
        oversamp: Whether to oversample.
        verb: Print text to the CLI (True).

    Returns:
        A dict of derived geometry (size, alpha, pixscale, seeing,
        tint, niter, jstep, windef, nap, d1, d2, npixperpix, nscr,
        ksamp, finepix, asperpix, ccdbin, nccd, rradiuspix, HR).
    '''

    size  = 2 * ngrid * pixel
    slide = 0.205
    alpha = slide / size
    
    pixscale = pixsize / effl * 206265
    if verb: print(f'Calculated pixel scale', pixscale)
    
    seeing = 0.98 * wavelen / r0 * 206265
    tint = (r0 ** (-5 / 3)) / 0.423 * ((0.5 * wavelen / np.pi) ** 2)
    if verb: print(f'Input r0 (meters): {r0}, J: {tint}')
    
    niter = math.floor(tacc / texp)
    jstep = math.floor(wind * texp / pixel + 0.5)
    if jstep < 1:
        jstep = 1
    windef = jstep * pixel / texp
    if verb: print(f'Screen shift per exposure: {jstep} pixels')
    if verb: print(f'Effective wind speed: {windef} m/s')
    if verb: print(f'Total iterations to simulate: {niter}\n')
    
    nap = math.floor(ngrid / 2)
    if verb: print(f'Aperture grid, pixels: {nap}')
    
    d1 = wavelen / pixscale * 206265
    if verb: print(f'd1: {d1}')
    
    d2 = wavelen / pixel * 206265
    if verb: print(f'd2: {d2}')
    
    if oversamp: npixperpix = 2 ** (math.floor(math.log2(1.5 * d / d1)) + 1)
    else: npixperpix = 1
    if verb: print(f'Pixel per pixel: {npixperpix}')
    
    nscr = 2 ** (math.floor(math.log2(1.5 * d / pixel)) + 1)
    if verb: print(f'Screen size in pixels: {nscr}')
    
    ksamp = max(int(nap // nscr), 1)
    if verb: print(f'Over-sampling factor: {ksamp}')
    
    finepix = 206265 * wavelen / (nap * pixel / ksamp)
    asperpix = pixscale
    ccdbin = asperpix / finepix
    nccd = int(round(nap / ccdbin))
    rradiuspix = 0.85 * d * (1 + eps) / (4 * pdist) * 206265 / asperpix
    HR = rradiuspix * pdist
    
    if verb: print(f'Re-sampled pixel size [m]: {pixel / ksamp:.6e}')
    if verb: print(f'CCD size & pixel [arcsec]: {nccd}, {asperpix}')
    if verb: print(f'Nominal ring radius [pix, arcsec]: {rradiuspix:.3f}, {rradiuspix * asperpix:.3f}')
    if verb: print(f'HR [unit?]: {HR}')
    
    if ccdbin < 1.0:
        if verb: print(f'Warning: fine grid {finepix:.4f} arcsec/pix is coarser than the '
              f'detector {asperpix:.4f} arcsec/pix; raise oversampling for a finer sim.')
    
    return {'size': size, 'alpha': alpha, 'pixscale': pixscale, 'seeing': seeing,
            'tint': tint, 'niter': niter, 'jstep': jstep, 'windef': windef,
            'nap': nap, 'd1': d1, 'd2': d2, 'npixperpix': npixperpix, 'nscr': nscr,
            'ksamp': ksamp, 'finepix': finepix, 'asperpix': asperpix,
            'ccdbin': ccdbin, 'nccd': nccd, 'rradiuspix': rradiuspix, 'HR': HR}


def photon_budget(starmag, texp, d, eps, BW=0.26, phot_con=1e11):
    '''
    Calculate the stellar photons collected per exposure.

    Uses phot_con photons/s/m^2 for a magnitude-0 star at the top of
    the atmosphere, scaled by the star magnitude, exposure time,
    effective bandwidth and the annular collecting area. 

    Where does it come from???

    Args:
        starmag: Star magnitude.
        texp: Exposure time, seconds.
        d: Aperture diameter, meters.
        eps: Fraction of central obscuration.
        BW: Effective bandwidth fraction (0.26).
        phot_con: Photons/s/m^2 for a mag-0 star (1e11).

    Returns:
        Stellar photons collected in one exposure.
    '''
    return phot_con * texp * BW * 10**(-0.4 * starmag) * np.pi * (d/2)**2 * (1 - eps)**2


def make_aperture(nap, d, pixel, ksamp, eps):
    '''
    Creates the annular pupil aperture mask
    '''
    apert = np.zeros((nap, nap))
    y, x = np.ogrid[:nap, :nap]
    r = np.hypot(x - nap / 2, y - nap / 2)
    radpix = d * 0.5 / pixel * ksamp
    inside = (r <= radpix) & (r >= radpix * eps)
    apert[inside] = 1
    return apert, r, radpix


def defocus_phase(d, wavelen, pdist, r, radpix):
    '''Applies main Zernike aberrations [a4, a11]'''
    a4 = (d ** 2 / (wavelen * pdist)) * (np.pi / (8 * np.sqrt(3)))
    a11 = -0.1 * a4
    rho = r / radpix
    tmp = a11 * np.sqrt(5) * (6.0 * (rho**4) - 6 * (rho**2))
    tmp += a4 * 2.0 * np.sqrt(3) * ((rho**2) - 0.5)
    return tmp, a4, a11


def pupil_to_psf(pupil, nap):
    '''
    Propagate a pupil-plane complex amplitude to the focal plane.

    Fraunhofer (far-field) propagation: inverse-FFT the pupil field
    and re-center it, returning the centered focal-plane complex
    amplitude. The caller squares the magnitude to get the intensity
    (PSF / ring image).

    Args:
        pupil: Complex amplitude at the pupil.
        nap: Aperture grid size, pixels (FFT normalization factor).

    Returns:
        Centered focal-plane complex amplitude.
    '''
    shifted = np.fft.ifftshift(pupil)
    amp = np.fft.ifft2(shifted) * (nap ** 2)
    return np.fft.fftshift(amp)


def apply_detector_noise(impix, starph, ron, rng, nccd):
    '''Applies artificial noise into image'''
    impix = impix * starph
    impix_clean = np.maximum(0.0, impix)
    return rng.poisson(impix_clean).astype(np.float64) + rng.normal(loc=0.0, scale=ron, size=(nccd, nccd))


def save_image(arr, title, fname, cmap='gray', log=False, colorbar=True):
    '''Save function for several images in script'''
    
    plt.figure(figsize=(6, 6))
    if log:
        plt.imshow(arr, cmap=cmap, origin='lower', norm=colors.LogNorm())
    else:
        plt.imshow(arr, cmap=cmap, origin='lower')
    plt.title(title)
    if colorbar:
        plt.colorbar()
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.savefig(os.path.join('images', fname), dpi=300, format='jpg')
    plt.close()


def write_fits_cube(cube, d, effl, eps, pdist, wavelen, texp, ron, gain,
                    starmag, seeing, nccd, pixel, asperpix, filename='test.fits'):
    '''
    Saves the cube into a fits image for later analysis

    Args:
        cube: 3D image cube (nframes, ny, nx) to write.
        d: Aperture diameter, meters (header DIAM).
        effl: Effective focal length, meters (header EFFL).
        eps: Fraction of central obscuration (header OBSC).
        pdist: Conjugation height, meters (header CONJ_H).
        wavelen: Reference wavelength, meters (header WAVELEN).
        texp: Exposure time, seconds (header EXPOSURE, us).
        ron: Read-out noise, electrons (header RON).
        gain: Detector gain setting (header GAIN).
        starmag: Stellar magnitude (header MAG).
        seeing: Input seeing, arcsec (header SEEING).
        nccd: Camera dimension, pixels (header CAM).
        pixel: Grid pixel size, meters (header PIXGRID).
        asperpix: Plate scale, arcsec/pixel (header PIXSCALE).
        filename: Output FITS path ('test.fits').

    Returns:
        The written FITS file path.
    '''
    hdu = fits.PrimaryHDU(cube.astype('float64'))
    
    header = hdu.header
    header['DATE-OBS'] = (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'File creation date / Observation Date')
    header['DIAM'] = (float(d), 'Telescope Diameter in meters')
    header['EFFL'] = (float(effl), 'Telescope focal length in meters')
    header['OBSC'] = (float(eps), 'Central obscuration (eps)')
    header['CONJ_H'] = (float(pdist), 'Conjugation height in meters')
    header['WAVELEN'] = (float(wavelen), 'Wavelength in meters')
    header['EXPOSURE'] = (float(texp*1e6), 'Exposure time in seconds')
    header['RON'] = (float(ron), 'Read out noise')
    header['GAIN'] = (float(gain), 'Detector gain (cube2.py reads as float)')
    header['MAG'] = (float(starmag), 'Stellar magnitude')
    header['SEEING'] = (float(seeing), 'Input seeing in arcseconds')
    header['CAM'] = (int(nccd), 'Camera dimensions')
    header['PIXGRID'] = (float(pixel), 'Input grid pixel size [m]')
    header['PIXSCALE'] = (float(asperpix), 'Arcseconds per pixel')
    header['STAR'] = ('', 'Star name (optional in cube2.py)')
    
    hdu.writeto(filename, overwrite=True)
    return filename


if __name__ == '__main__':
    main()