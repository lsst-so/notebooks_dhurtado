#!/usr/bin/env python3
'''
Processes a cube of ring images (cubecoef.pro in IDL):
centers each frame, projects onto the sector/angular masks,
then returns the ring parameters and per-frame coefficients.

Usage:
    python cubecoef.py --help         # list all arguments and defaults
    python cubecoef.py --cubefile test.fits --mmax 20 --nsect 8
    import cubecoef; impar, coef = cubecoef.cubecoef('test.fits')
    help(cubecoef.cubecoef)           # docstring for any function

Auth: A. Tokovinin
Translated: D. Hurtado

'''

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import numpy as np
import os
from astropy.io import fits


def main():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--cubefile', dest='cubefile', type=str, default='test.fits',
                   help='FITS file with the cube of ring images, e.g. test.fits')
    p.add_argument('--mmax', dest='mmax', type=int, default=20,
                   help='Max order of angular signals, def=20')
    p.add_argument('--nsect', dest='nsect', type=int, default=8,
                   help='Number of sectors for radius calculation, def=8')
    p.add_argument('--drad', dest='drad', type=float, default=1.5,
                   help='Mask width in lambda/(0.5*D*(1-eps)) units, def=1.5')
    p.add_argument('--interpol', dest='interpol', type=int, default=1,
                   help='Use sub-pixel (Fourier) shifts for centering, def=1')
    p.add_argument('--nstart', dest='nstart', type=int, default=50,
                   help='Number of initial frames for the ring template, def=50')
    p.add_argument('--leak', dest='leak', type=float, default=1,
                   help='Leaky-integrator gain for the running centroid, def=1')
    p.add_argument('--flat', dest='flat', type=bool, default=True,
                   help='Divide frames by a flat field, boolean, def=True')
    p.add_argument('--display', dest='display', type=bool, default=True,
                   help='Decides if images are displayed, boolean, def=True')
    p.add_argument('--verbose', dest='verb', type=bool, default=True,
                   help='Print text into CLI, def=True')
    
    args     = p.parse_args()
    cubefile = args.cubefile
    mmax     = args.mmax
    nsect    = args.nsect
    drad     = args.drad
    interpol = args.interpol
    nstart   = args.nstart
    leak     = args.leak
    flat     = args.flat
    display  = args.display
    verb     = args.verb
    
    return cubecoef(cubefile, mmax, nsect, drad, interpol,
                    nstart, leak, flat, display, verb)


def cubecoef(cubefile='test.fits', mmax=20, nsect=8, drad=1.5,
             interpol=1, nstart=50, leak=1, flat=True, display=True, verb=True):
    '''
    Takes the cube FITS file, then processes the image to find
    ring size, shape, intensity, width and centroid.
    
    Performs a mask-shape physical comparison, then
    repeats the process for each frame in the FITS.
    Evaluates the coma as a diagnostic of the ring shape
    (reported in impar, not used to correct the measurements).
    
    Args:
        cubefile: FITS file with the cube of ring images ('test.fits').
        mmax: Max order of angular signals (20).
        nsect: Number of sectors for the radius calc (8).
        drad: Mask width in lambda/(0.5*D*(1-eps)) units (1.5).
        interpol: Use sub-pixel (Fourier) shifts for centering (1).
        nstart: Number of initial frames for the ring template (50).
        leak: Leaky-integrator gain for the running centroid (1).
        flat: Divide frames by a flat field (True).
        display: Save the diagnostic images/gif (True).
        verb: Print text to the CLI (True).
    
    Returns:
        (impar, coef): impar is a dict of ring parameters; coef is
        the (ncoef, nframes) array of per-frame coefficients.
    '''
    cube, param = read_cubefile(cubefile)
    wavelen  = param['wavelen']
    d        = param['d']
    effl     = param['effl']
    pdist    = param['pdist']
    pixel    = param['pixel']
    eps      = param['eps']
    asperpix = param['asperpix']
    ron      = param['ron']
    texp     = param['texp']
    
    nz, nx, ny = cube.shape
    if verb: print(f'Sizes {nx, ny, nz}')
    if nx != ny:
        raise ValueError(f'Non square frames! ({nx} =/= {ny})')
    if nz < nstart:
        raise ValueError(f'Fewer frames than minimum ({nz} < {nstart})')
    
    # Vector radius and coordinates
    x_coords = np.arange(nx) - nx / 2
    y_coords = np.arange(ny) - ny / 2
    x, y = np.meshgrid(x_coords, y_coords)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    phi[ny // 2, : nx // 2 + 1] = -np.pi
    
    # Sector filters
    rwt, fwt, xsect, ysect = sector_masks(nx, nsect, r, phi)
    
    # Initial ring parameters
    xc, yc, radpix, rwidth, background, imgcent = ring_params(cube, nstart, x, y, r, rwt, fwt, xsect, 
                                                              ysect, nx, wavelen, d, eps, pixel, verb)
    
    if radpix > nx / 2: # Check if cube is empty
        raise ValueError(f'Empty cube! Please check and restart simulation')
    
    # General matrix of masks
    maskmat, ncoef, ringmask = mask_matrix(nsect, mmax, nx, ny, r, phi, rwt, fwt, radpix, rwidth, drad, verb)
    
    # Optional control plot before main loop
    if display:
        imax = np.max(imgcent[:, nx // 2])
        plt.figure()
        plt.plot(x[0, :], np.maximum(imgcent[:, nx // 2], 0), 'b-', label='Profile')
        plt.plot(-x[0, :], imgcent[:, nx // 2], 'r--', label='Mirrored')
        
        plt.hlines(imax / 2, -radpix - rwidth, -radpix + rwidth, colors='g', linewidth=2, label='Width')
        plt.hlines(imax / 2, radpix - rwidth, radpix + rwidth, colors='g', linewidth=2)
        plt.legend()
        
        plt.title('Initial Ring Alignment Control')
        plt.savefig(os.path.join('images', 'ring_alignment.jpg'), dpi=300, format='jpg')
        #plt.show()
        plt.close()
        
        save_image(imgcent, 'Initial centered ring (imgcent)', 'initial_ring.jpg')
    
    # Main Loop over the Cube
    coef = np.zeros((ncoef, nz), dtype=np.float64)
    xcent = np.zeros(nz, dtype=np.float64)
    ycent = np.zeros(nz, dtype=np.float64)
    rad = np.zeros(nz, dtype=np.float64)
    
    x0, y0, rad0 = xc, yc, radpix
    if verb: print('Processing the cube')
    imav_sum = np.zeros((ny, nx), dtype=np.float64)
    
    # Gif creator
    if display:
        writer = mpl.animation.PillowWriter(fps=60)
        tmp_fig = plt.figure()
        writer.setup(tmp_fig, 'cube_centroid.gif', dpi=100)
        plt.close(tmp_fig)
    
    for i in range(nz):
        tmp_frame = cube[i, :, :].astype(np.float64) - background
        if flat is not None:
            tmp_frame /= flat
        
        tmp_shifted = center_frame(tmp_frame, x0, y0, x, y, nx, interpol)
        imav_sum += tmp_shifted
        
        # Project frame onto spatial mask
        c = project_coef(maskmat, tmp_shifted, nsect)
        coef[:, i] = c
        
        # Update radius and centroids
        radii_i = c[:nsect]
        x0, y0, rad0, dr = update_centroid(radii_i, nsect, xsect, ysect, x0, y0, rad0, leak)
        # Here it sees the last frame centroid
        xcent[i] = x0
        ycent[i] = y0
        rad[i] = dr
        
        if (i + 1) % 100 == 0 and display:
            plot_frame(writer, tmp_shifted, radii_i, xsect, ysect, nx, ny, i, nz)
    
    if display: writer.finish()  # Compile output.gif
    if verb and display: print('Gif saved')
    
    imav_final = imav_sum / nz # Image average
    
    # Post-processing
    flux = float(np.mean(coef[2 * nsect, :]))  # m=0 mode mean flux in ADU
    coef[2 * nsect :, :] /= flux
    
    fluxvar = float(np.std(coef[2 * nsect, :], ddof=1))
    rad_mean = float(np.mean(rad))
    xc_mean = float(np.mean(xcent))
    yc_mean = float(np.mean(ycent))
    xcvar = float(np.std(xcent, ddof=1))
    ycvar = float(np.std(ycent, ddof=1))
    
    # Evaluate coma
    coma, angle = check_coma(coef, nsect, mmax)
    
    # Ring contrast per sector
    contrast = np.zeros(nsect, dtype=np.float64)
    for j in range(nsect):
        tmp_sec = imav_final * fwt[:, :, j]
        contrast[j] = np.max(tmp_sec) / np.sum(tmp_sec)
    
    noisepar = noise_params(imav_final, ringmask, r, radpix)
    
    impar = {
        'backgr':   background,
        'flux':     flux,
        'fluxvar':  fluxvar,
        'rad':      rad_mean,
        'rwidth':   rwidth,
        'xc':       xc_mean,
        'yc':       yc_mean,
        'xcvar':    xcvar,
        'ycvar':    ycvar,
        'coma':     coma,
        'angle':    angle,
        'contrast': float(np.mean(contrast)),
        'noisepar': noisepar,
        'nsect':    nsect,
        'mmax':     mmax,
        'asperpix': asperpix,
        'd':        d,
        'effl':     effl,
        'eps':      eps,
        'pdist':    pdist,
        'ron':      ron,
        'wavelen':  wavelen,
        'pixel':    pixel,
        'texp':     texp
            }
    
    if verb: print(f'Cube processed! Parameters saved')
    
    if display:
        save_image(imav_final, 'Average aligned cube', 'cube_average.jpg')
        
        plt.figure()
        plt.plot(xcent, ycent, '+', label='Centroid (X, Y) [pix]',)
        #plt.plot(ycent * pixel, linestyle='--', label='Y-drift vs Frame')
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Centroid Position & Drift [pix]')
        #plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('images', 'centroid_drift.jpg'), dpi=300, format='jpg')
        #plt.show()
        plt.close()
    
    return impar, coef

# Used functions ---------------------------------------------------------------------------

def read_cubefile(cubefile):
    '''
    Takes the cube filename and reads the image cube and the
    header parameters from the FITS file.
    
    Args:
        cubefile: Path to the FITS cube to read.
    
    Returns:
        (cube, param): cube is the 3D image array; param is a dict
        of header values (wavelen, d, effl, pdist, pixel, eps,
        asperpix, ron).
    '''
    with fits.open(cubefile) as hdul:
        cube = hdul[0].data
        hdr  = hdul[0].header
    
    param = {
        'wavelen':  float(hdr['WAVELEN']),
        'd':        float(hdr['DIAM']),
        'effl':     float(hdr['EFFL']),
        'pdist':    float(hdr['CONJ_H']),
        'pixel':    float(hdr['PIXGRID']),
        'eps':      float(hdr['OBSC']),
        'asperpix': float(hdr['PIXSCALE']),
        'ron':      float(hdr['RON']),
        'texp':     float(hdr['EXPOSURE']) * 1e-6   # µs -> s
            }
    return cube, param


def sector_masks(nx, nsect, r, phi):
    '''
    Divides the field into nsect (usually 8) angular sectors and 
    builds each sector flux and radius weight maps, plus the vectors
    pointing to each sector center. These let the radius be measured
    independently in each sector (DIMM-like).
    
    Args:
        nx: Frame size, pixels (frames are square).
        nsect: Number of angular sectors.
        r: 2D radial-distance grid, pixels.
        phi: 2D azimuthal-angle grid, radians.
    
    Returns:
        (rwt, fwt, xsect, ysect): rwt is the per-sector radius weight
        (r inside the sector), fwt the per-sector flux mask (1 inside
        the sector), xsect/ysect the cos/sin of each sector center.
    '''
    rwt = np.zeros((nx, nx, nsect), dtype=np.float64) # Radius calc
    fwt = np.zeros((nx, nx, nsect), dtype=np.float64) # Flux in sectors
    sect = 2 * np.pi / nsect
    
    for j in range(nsect):
        low_bound = sect * (j - nsect / 2)
        high_bound = sect * (j + 1 - nsect / 2)
        sector = (phi >= low_bound) & (phi <= high_bound) # Bool mask
        fwt[:, :, j] = sector.astype(np.float64)
        rwt[:, :, j] = sector * r
    
    phisect = sect * (np.arange(nsect, dtype=np.float64) - nsect / 2 + 0.5)
    xsect = np.cos(phisect)
    ysect = np.sin(phisect)
    return rwt, fwt, xsect, ysect


def ring_params(cube, nstart, x, y, r, rwt, fwt, xsect, ysect, nx,
                wavelen, d, eps, pixel, verb=True):
    '''
    Estimates the initial ring parameters from the average of the
    first nstart frames: background, center (xc, yc), mean radius,
    and ring width then recenters with a Fourier shift.
    
    Args:
        cube: 3D image cube (nframes, ny, nx).
        nstart: Number of initial frames to average for the template.
        x: 2D x-coordinate grid, pixels (center at 0).
        y: 2D y-coordinate grid, pixels (center at 0).
        r: 2D radial-distance grid, pixels.
        rwt: Per-sector radius weight maps (from sector_masks).
        fwt: Per-sector flux masks (from sector_masks).
        xsect: cos of each sector center angle.
        ysect: sin of each sector center angle.
        nx: Frame size, pixels.
        wavelen: Wavelength, meters (for the min ring width).
        d: Aperture diameter, meters.
        eps: Central obscuration fraction.
        pixel: Detector plate scale, arcsec/pixel.
        verb: Print text to the CLI (True).
    
    Returns:
        (xc, yc, radpix, rwidth, background, imgcent): ring center,
        mean radius and width [pix], background level, and the
        centered average image.
    '''
    nsect = rwt.shape[2]
    imav = np.mean(cube[:nstart, :, :], axis=0)
    background = float(np.median(np.concatenate([imav[:, 0], imav[:, -1]])))
    tmp = imav - background
    
    itot = np.sum(tmp)
    xc = np.sum(tmp * x) / itot
    yc = np.sum(tmp * y) / itot
    # Crude integer recenter: xc shifts columns (axis 1), yc shifts rows (axis 0)
    imgcent = np.roll(tmp, (-int(np.round(yc)), -int(np.round(xc))), axis=(0, 1))
    
    radii = np.zeros(nsect, dtype=np.float64)
    for j in range(nsect):
        radii[j] = np.sum(imgcent * rwt[:, :, j]) / np.sum(imgcent * fwt[:, :, j])
    
    radpix = np.mean(radii) # Average radius
    dx1 = np.sum(radii * xsect) / nsect * 2.3 # 2.3 is empirical, this might change
    dy1 = np.sum(radii * ysect) / nsect * 2.3
    xc += dx1
    yc += dy1
    
    # Fourier phase shift centering
    imgcent = fourier_shifting(imgcent, dx1, dy1, x, y, nx)
    
    # Diffraction-limited minimum ring width
    rwidthmin = (wavelen / d / (1 - eps)) * 2 * 206265 / pixel
    tmp_thresh = np.maximum(imgcent - 0.1 * np.max(imgcent), 0) # Image threshold
    radvar = np.sum(tmp_thresh * (r - radpix)**2) / np.sum(tmp_thresh)
    rwidth = np.sqrt(radvar) * 2.35
    
    if verb: print(f'Ring rad, width, minwidth [pix]: {radpix}, {rwidth}, {rwidthmin}')
    
    # Recompute background outside 1.5 ring radius
    out = r > 1.5 * radpix
    background += float(np.median(imgcent[out]))
    
    return xc, yc, radpix, rwidth, background, imgcent


def mask_matrix(nsect, mmax, nx, ny, r, phi, rwt, fwt, radpix, rwidth, drad, verb=True):
    '''
    Builds the projection matrix of masks used to measure the ring
    coefficients in each frame: the per-sector radius/flux masks plus
    the piston-subtracted cos/sin angular masks (m = 0..mmax), all
    restricted to the ring annulus.
    
    Args:
        nsect: Number of angular sectors.
        mmax: Max angular order (gives mmax+1 cos and sin terms).
        nx: Frame width, pixels.
        ny: Frame height, pixels.
        r: 2D radial-distance grid, pixels.
        phi: 2D azimuthal-angle grid, radians.
        rwt: Per-sector radius weight maps (from sector_masks).
        fwt: Per-sector flux masks (from sector_masks).
        radpix: Ring radius, pixels.
        rwidth: Ring width, pixels.
        drad: Mask half-width in units of rwidth.
        verb: Print text to the CLI (True).
    
    Returns:
        (maskmat, ncoef, ringmask): the (ncoef, nx*ny) projection
        matrix, its row count, and the boolean ring-annulus mask.
    '''
    drhopix = drad * rwidth
    ringmask = (r >= (radpix - drhopix)) & (r <= (radpix + drhopix))
    nring = np.count_nonzero(ringmask)
    if verb: print(f'nring {nring}')
    
    ncoef = 2 * nsect + 2 * (mmax + 1)
    maskmat = np.zeros((ncoef, nx * ny), dtype=np.float64)
    
    # Insert radial sector masks into projection matrix
    for j in range(nsect):
        maskmat[j, :] = (rwt[:, :, j] * ringmask).ravel()
        maskmat[j + nsect, :] = (fwt[:, :, j] * ringmask).ravel()
    
    cwt = np.zeros((ny, nx, mmax + 1), dtype=np.float64)
    swt = np.zeros((ny, nx, mmax + 1), dtype=np.float64)
    
    for j in range(mmax + 1):
        tmp_c = np.cos(phi * j) * ringmask
        cwt[:, :, j] = tmp_c - (np.sum(tmp_c) / (nx * ny) if j > 0 else 0)
        tmp_s = np.sin(phi * j) * ringmask
        swt[:, :, j] = tmp_s - np.sum(tmp_s) / (nx * ny)
        
        maskmat[2 * nsect + j, :] = cwt[:, :, j].ravel()
        maskmat[2 * nsect + mmax + 1 + j, :] = swt[:, :, j].ravel()
    
    return maskmat, ncoef, ringmask


def fourier_shifting(img, dx, dy, x, y, nx):
    '''
    Shifts an image by (dx, dy) sub-pixels using a Fourier phase
    ramp: multiply the FFT by exp(i*phase) and inverse-transform.
    
    Args:
        img: 2D image to shift.
        dx: Shift along x, pixels.
        dy: Shift along y, pixels.
        x: 2D x-coordinate grid, pixels.
        y: 2D y-coordinate grid, pixels.
        nx: Frame size, pixels (normalizes the phase ramp).
    
    Returns:
        The real part of the sub-pixel-shifted image.
    '''
    arg = 2 * np.pi * (dx * x + dy * y) / nx
    shift_kernel = np.fft.ifftshift(np.exp(1j * arg))
    return np.fft.ifft2(np.fft.fft2(img) * shift_kernel).real


def center_frame(frame, x0, y0, x, y, nx, interpol):
    '''
    Centers each frame to maintain cohesion between loops.
    
    Uses a sub-pixel Fourier shift when interpol is set, otherwise a
    whole-pixel np.roll.
    
    Args:
        frame: 2D frame to recenter.
        x0: Current ring-center x estimate, pixels.
        y0: Current ring-center y estimate, pixels.
        x: 2D x-coordinate grid, pixels.
        y: 2D y-coordinate grid, pixels.
        nx: Frame size, pixels.
        interpol: Use sub-pixel Fourier shift if truthy, else roll.
    
    Returns:
        The recentered 2D frame.
    '''
    if interpol:
        return fourier_shifting(frame, x0, y0, x, y, nx)
    return np.roll(frame, (-int(np.round(y0)), -int(np.round(x0))), axis=(0, 1))


def project_coef(maskmat, frame, nsect):
    '''
    Projects one flattened frame onto the mask matrix to get its
    coefficient vector, then normalizes the per-sector radii by the
    per-sector flux (first nsect entries divided by the next nsect).
    
    Args:
        maskmat: (ncoef, nx*ny) projection matrix (from mask_matrix).
        frame: 2D frame to project.
        nsect: Number of angular sectors.
    
    Returns:
        The length-ncoef coefficient vector for this frame.
    '''
    c = maskmat @ frame.ravel()
    c[:nsect] = c[:nsect] / c[nsect : 2 * nsect]  # Normalize radii
    return c


def update_centroid(radii_i, nsect, xsect, ysect, x0, y0, rad0, leak):
    '''
    Recomputes and updates the centroid in each
    frame of the main loop.
    
    Converts this frame's per-sector radii into center offsets
    (dx, dy) and a radius change dr, then applies them to the
    running estimate through the leaky-integrator gain.
    
    Args:
        radii_i: This frame's per-sector radii (first nsect coefs).
        nsect: Number of angular sectors.
        xsect: cos of each sector center angle.
        ysect: sin of each sector center angle.
        x0: Current center x estimate, pixels.
        y0: Current center y estimate, pixels.
        rad0: Current running radius estimate, pixels.
        leak: Leaky-integrator gain (1 = full update).
    
    Returns:
        (x0, y0, rad0, dr): updated center, updated running radius,
        and this frame's radius change.
    '''
    dr = np.sum(radii_i) / nsect
    dx = np.sum(radii_i * xsect) / nsect * 2.3
    dy = np.sum(radii_i * ysect) / nsect * 2.3
    
    x0 += leak * dx
    y0 += leak * dy
    rad0 = rad0 * (1 - leak) + dr * leak
    return x0, y0, rad0, dr


def plot_frame(writer, tmp_shifted, radii_i, xsect, ysect, nx, ny, i, nz):
    '''
    Takes the input frame and designs the plot image
    Returns nothing, only saves the image into the gif writer.
    
    Args:
        writer: Matplotlib animation writer capturing the gif.
        tmp_shifted: The recentered frame to render.
        radii_i: This frame's per-sector radii (for the overlay).
        xsect: cos of each sector center angle.
        ysect: sin of each sector center angle.
        nx: Frame width, pixels.
        ny: Frame height, pixels.
        i: Current frame index (0-based).
        nz: Total number of frames.
    '''
    max_val = np.max(tmp_shifted)
    tmp1 = (tmp_shifted / max_val if max_val != 0 else np.zeros_like(tmp_shifted))
    
    # Overlay sector center coordinates and central pixel marker
    x_pts = np.clip(np.round(radii_i * xsect + nx / 2).astype(int), 0, nx - 1)
    y_pts = np.clip(np.round(radii_i * ysect + ny / 2).astype(int), 0, ny - 1)
    
    tmp_visual = tmp1.copy()
    tmp_visual[y_pts, x_pts] = -0.5
    tmp_visual[int(ny // 2), int(nx // 2)] = 1
    
    fig, ax = plt.subplots(num='Live Frame Monitor', figsize=(5, 5))
    ax.imshow(tmp_visual, interpolation=None, cmap='gray', origin='lower')
    ax.set_title(f'Frame {i + 1} / {nz}')
    ax.axis('off')
    
    # Gif input
    writer.fig = fig  # Attach current figure instance to writer
    writer.grab_frame()  # Capture frame into GIF buffer
    #plt.show()
    plt.close(fig)  # close per-frame figure so they don't accumulate


def check_coma(coef, nsect, mmax):
    '''
    Evaluates the shape of the ring to find coma.
    
    Combines the mean m=1 cosine and sine coefficients into a 
    coma magnitude and orientation angle.
    
    Args:
        coef: (ncoef, nframes) per-frame coefficient array.
        nsect: Number of angular sectors (offset into coef).
        m: Max angular order (mmax; locates the m=1 sine term).
    
    Returns:
        (coma, angle): coma magnitude and its angle in degrees.
    '''
    cm = float(np.mean(coef[2 * nsect + 1, :]))
    sm = float(np.mean(coef[2 * nsect + mmax + 2, :]))
    coma = float(np.hypot(cm, sm))
    angle = float(np.degrees(np.arctan2(sm, cm)))
    return coma, angle
    
def noise_params(imav_final, ringmask, r, radpix):
    '''
    Evaluates the noise of the frame in respect to
    pixel intensity variances.
    
    Computes the angular and radial photon- and readout-noise
    factors from the normalized average image over the ring mask.
    
    Args:
        imav_final: Average aligned image over the cube.
        ringmask: Boolean ring-annulus mask.
        r: 2D radial-distance grid, pixels.
        radpix: Ring radius, pixels.
    
    Returns:
        [noise1, noise2, noise1r, noise2r]: angular photon, angular
        readout, radial photon, and radial readout noise factors.
    '''
    tmp_norm = imav_final / np.sum(imav_final)
    t0 = np.sum(tmp_norm * ringmask)
    noise1 = np.sum((ringmask**2) * tmp_norm) / t0  # Angular photon noise factor
    noise2 = np.sum(ringmask**2) / t0  # Angular readout noise factor
    
    tmp2 = ringmask * (r - radpix)
    noise1r = np.sum(tmp_norm * (tmp2**2))  # Radial photon noise factor
    noise2r = np.sum(tmp2**2)  # Radial readout noise factor
    return [float(noise1), float(noise2), float(noise1r), float(noise2r)]


def save_image(arr, title, fname, cmap='gray', log=False, colorbar=True):
    '''
    Saves plot image into debug folder.
    
    Args:
        arr: 2D array to display.
        title: Plot title.
        fname: Output filename, written under the images folder.
        cmap: Matplotlib colormap ('gray').
        log: Use a logarithmic color scale (False).
        colorbar: Draws a colorbar (True).
    '''
    plt.figure(figsize=(6, 6))
    if log: plt.imshow(arr, cmap=cmap, origin='lower', norm=colors.LogNorm())
    else: plt.imshow(arr, cmap=cmap, origin='lower')
    plt.title(title)
    if colorbar: plt.colorbar()
    plt.xlabel('X [pixels]')
    plt.ylabel('Y [pixels]')
    plt.savefig(os.path.join('images', fname), dpi=300, format='jpg')
    plt.close()


if __name__ == '__main__':
    main()