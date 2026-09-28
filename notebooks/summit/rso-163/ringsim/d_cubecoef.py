# Here lies the cubecoef function
#!/usr/bin/env python3
'''
Processes a cube of ring images (cubecoef.pro): centers each frame,
projects onto the sector/angular masks, and returns the ring
parameters and per-frame coefficients.


Auth: A. Tokovinin
Translated: D. Hurtado

'''

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation # noqa: F401
import numpy as np

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

    return cubecoef(cubefile, mmax, nsect, drad, interpol,
                    nstart, leak, flat, display)


def cubecoef(cubefile='test.fits', mmax=20, nsect=8, drad=1.5,
             interpol=1, nstart=50, leak=1, flat=True, display=True):

    with fits.open(cubefile) as hdul:
        cube = hdul[0].data
        hdr  = hdul[0].header
    
    wavelen  = float(hdr['WAVELEN'])
    d        = float(hdr['DIAM'])
    effl     = float(hdr['EFFL'])
    pdist    = float(hdr['CONJ_H'])
    pixel    = float(hdr['PIXGRID'])
    eps      = float(hdr['OBSC'])
    asperpix = float(hdr['PIXSCALE'])
    ron      = float(hdr['RON'])
    
    # Out from parameters
    m = mmax 
    nsect = nsect
    interpol = interpol
    drad = drad
    
    # Hard-coded
    nstart = 50
    leak = 1
    flat = True
    
    nz, nx, ny = cube.shape
    print(f'Sizes {nx, ny, nz}')
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
    
    # Filters and output arrays
    rwt = np.zeros((nx, nx, nsect), dtype=np.float64) # Radius calc, check if double nx or nx, ny
    fwt = np.zeros((nx, nx, nsect), dtype=np.float64) # Flux in sectors
    sect = 2 * np.pi / nsect
    
    # Un masked sectors
    for j in range(nsect):
        low_bound = sect * (j - nsect / 2)
        high_bound = sect * (j + 1 - nsect / 2)
        sector = (phi >= low_bound) & (phi <= high_bound) # Bool mask
        fwt[:, :, j] = sector.astype(np.float64)
        rwt[:, :, j] = sector * r
    
    phisect = sect * (np.arange(nsect, dtype=np.float64) - nsect / 2 + 0.5)
    xsect = np.cos(phisect)
    ysect = np.sin(phisect)
    
    # Initial ring parameters
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
    arg = 2 * np.pi * (dx1 * x + dy1 * y) / nx
    shift_kernel = np.fft.ifftshift(np.exp(1j * arg))
    imgcent = np.fft.ifft2(np.fft.fft2(imgcent) * shift_kernel).real
    
    # Diffraction-limited minimum ring width
    rwidthmin = (wavelen / d / (1 - eps)) * 2 * 206265 / pixel
    tmp_thresh = np.maximum(imgcent - 0.1 * np.max(imgcent), 0) # Image threshold
    radvar = np.sum(tmp_thresh * (r - radpix)**2) / np.sum(tmp_thresh)
    rwidth = np.sqrt(radvar) * 2.35
    
    print(f'Ring rad, width, minwidth [pix]: {radpix}, {rwidth}, {rwidthmin}')
    
    if radpix > nx / 2: # Check if cube is empty
        #
        raise ValueError(f'Empty cube! Please check and restart simulation')
    
    # Recompute background outside 1.5 ring radius
    out = r > 1.5 * radpix
    background += float(np.median(imgcent[out]))
    
    # General matrix of masks
    drhopix = drad * rwidth
    ringmask = (r >= (radpix - drhopix)) & (r <= (radpix + drhopix))
    nring = np.count_nonzero(ringmask)
    
    print(f'nring {nring}')
    
    # Full-frame projection matrix: the piston-subtracted cos/sin masks are nonzero
    # outside the ring, so projecting against the full (flattened) frame keeps the
    # DC-cancellation. Radial/flux masks are zero outside the ring anyway.
    ncoef = 2 * nsect + 2 * (m + 1)
    maskmat = np.zeros((ncoef, nx * ny), dtype=np.float64)
    
    # Insert radial sector masks into projection matrix
    for j in range(nsect):
        maskmat[j, :] = (rwt[:, :, j] * ringmask).ravel()
        maskmat[j + nsect, :] = (fwt[:, :, j] * ringmask).ravel()
    
    cwt = np.zeros((ny, nx, m + 1), dtype=np.float64)
    swt = np.zeros((ny, nx, m + 1), dtype=np.float64)
    
    for j in range(m + 1):
        tmp_c = np.cos(phi * j) * ringmask
        cwt[:, :, j] = tmp_c - (np.sum(tmp_c) / (nx * ny) if j > 0 else 0)
        tmp_s = np.sin(phi * j) * ringmask
        swt[:, :, j] = tmp_s - np.sum(tmp_s) / (nx * ny)
    
        maskmat[2 * nsect + j, :] = cwt[:, :, j].ravel()
        maskmat[2 * nsect + m + 1 + j, :] = swt[:, :, j].ravel()
    
    # Optional control plot before main loop
    if display:
        imax = np.max(imgcent[:, nx // 2])
        plt.figure('Sector Definition Control')
        plt.plot(x[0, :], np.maximum(imgcent[:, nx // 2], 0), 'b-', label='Profile')
        plt.plot(-x[0, :], imgcent[:, nx // 2], 'r--', label='Mirrored')
        
        plt.hlines(imax / 2, -radpix - rwidth, -radpix + rwidth, colors='g', linewidth=2, label='Width')
        plt.hlines(imax / 2, radpix - rwidth, radpix + rwidth, colors='g', linewidth=2)
        plt.legend()
        
        plt.title('Initial Ring Alignment Control')
        plt.show()
        plt.clf
    
    # Main Loop over the Cube
    coef = np.zeros((ncoef, nz), dtype=np.float64)
    xcent = np.zeros(nz, dtype=np.float64)
    ycent = np.zeros(nz, dtype=np.float64)
    rad = np.zeros(nz, dtype=np.float64)
    
    x0, y0, rad0 = xc, yc, radpix
    print('Processing the cube')
    imav_sum = np.zeros((ny, nx), dtype=np.float64)
    
    # Gif creator
    writer = mpl.animation.PillowWriter(fps=60)
    tmp_fig = plt.figure()
    writer.setup(tmp_fig, 'cube_centroid.gif', dpi=100)
    plt.close(tmp_fig)
    
    for i in range(nz):
        
        tmp_frame = cube[i, :, :].astype(np.float64) - background
        if flat is not None:
            tmp_frame /= flat
    
        # Frame alignment / sub-pixel shift
        if interpol:
            arg = 2 * np.pi * (x0 * x + y0 * y) / nx
            shift_kernel = np.fft.ifftshift(np.exp(1j * arg))
            tmp_shifted = np.fft.ifft2(np.fft.fft2(tmp_frame) * shift_kernel).real
        else:
            tmp_shifted = np.roll(tmp_frame, (-int(np.round(y0)), -int(np.round(x0))), axis=(0, 1))
    
        imav_sum += tmp_shifted
    
        # Project frame onto spatial mask
        c = maskmat @ tmp_shifted.ravel()
        c[:nsect] = c[:nsect] / c[nsect : 2 * nsect]  # Normalize radii
        coef[:, i] = c
    
        # Update radius and centroids
        radii_i = c[:nsect]
        dr = np.sum(radii_i) / nsect
        dx = np.sum(radii_i * xsect) / nsect * 2.3
        dy = np.sum(radii_i * ysect) / nsect * 2.3
    
        x0 += leak * dx
        y0 += leak * dy
        rad0 = rad0 * (1 - leak) + dr * leak
    
        xcent[i] = x0
        ycent[i] = y0
        rad[i] = dr
        
        max_val = np.max(tmp_shifted)
        tmp1 = (tmp_shifted / max_val
                if max_val != 0
                else np.zeros_like(tmp_shifted))
        if (i + 1) % 100 == 0:
            # Overlay sector center coordinates and central pixel marker
            x_pts = np.clip(np.round(radii_i * xsect + nx / 2).astype(int), 0, nx - 1)
            y_pts = np.clip(np.round(radii_i * ysect + ny / 2).astype(int), 0, ny - 1)
            
            tmp_vis = tmp1.copy()
            tmp_vis[y_pts, x_pts] = -0.5
            tmp_vis[int(ny // 2), int(nx // 2)] = 1
            
            #clear_output(wait=True)  # Clears previous frame before rendering the new one
            
            fig = plt.figure('Live Frame Monitor', figsize=(5, 5))
            plt.imshow(tmp_vis, interpolation=None, cmap='gray', origin='lower')
            plt.title(f'Frame {i + 1} / {nz}')
            plt.axis('off')
            plt.pause(0.0001)
            
            # Gif input
            writer.fig = fig  # Attach current figure instance to writer
            writer.grab_frame()  # Capture frame into GIF buffer
            plt.show()
            plt.clf
    
    writer.finish()  # Compile output.gif
    print('Gif saved')
        
    imav_final = imav_sum / nz
    
    # Post-processing
    flux = float(np.mean(coef[2 * nsect, :]))  # m=0 mode mean flux in ADU
    coef[2 * nsect :, :] /= flux
    
    fluxvar = float(np.std(coef[2 * nsect, :], ddof=1))
    rad_mean = float(np.mean(rad))
    xc_mean = float(np.mean(xcent))
    yc_mean = float(np.mean(ycent))
    xcvar = float(np.std(xcent, ddof=1))
    ycvar = float(np.std(ycent, ddof=1))
    
    # Coma
    cm = float(np.mean(coef[2 * nsect + 1, :]))
    sm = float(np.mean(coef[2 * nsect + mmax + 2, :]))
    coma = float(np.hypot(cm, sm))
    angle = float(np.degrees(np.arctan2(sm, cm)))
    
    # Ring contrast per sector
    contrast = np.zeros(nsect, dtype=np.float64)
    for j in range(nsect):
        tmp_sec = imav_final * fwt[:, :, j]
        contrast[j] = np.max(tmp_sec) / np.sum(tmp_sec)
    
    # Noise calc
    tmp_norm = imav_final / np.sum(imav_final)
    t0 = np.sum(tmp_norm * ringmask)
    noise1 = np.sum((ringmask**2) * tmp_norm) / t0  # Angular photon noise factor
    noise2 = np.sum(ringmask**2) / t0  # Angular readout noise factor
    
    tmp2 = ringmask * (r - radpix)
    noise1r = np.sum(tmp_norm * (tmp2**2))  # Radial photon noise factor
    noise2r = np.sum(tmp2**2)  # Radial readout noise factor
    noisepar = [float(noise1), float(noise2), float(noise1r), float(noise2r)]
    
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
        'ngrid':    ngrid,
        'mmax':     mmax,
        'asperpix': asperpix,
        'd':        d,
        'effl':     effl,
        'eps':      eps,
        'pdist':    pdist,
        'ron':      ron,
        'wavelen':  wavelen,
        'pixel':    pixel
            }
    
    print(f'Cube processed! Parameters saved')
    
    if display:
        plt.figure('Centroid Track (pix)')
        plt.plot(xcent, ycent, '+', label='Centroid (X, Y) [pix]',)
        #plt.plot(ycent * pixel, linestyle='--', label='Y-drift vs Frame')
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Centroid Position & Drift [pix]')
        #plt.legend()
        plt.grid(True)
        plt.show()
        plt.clf
    
    return impar, coef


if __name__ == '__main__':
    main()