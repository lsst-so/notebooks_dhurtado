""" Based on cubecoef.pro (Tokovinin) """
# version 1.0

from astropy.io import fits
import numpy as np
# import matplotlib.pyplot as plt
import json
import codecs
import sys


def Moments(data_cube):
    # print("inside Moments()")
    nz, ny, nx = data_cube.shape  # 2000 64 64

    nstart = 50  # initial average of first 50 frames
    imav = np.average(data_cube[0:nstart], axis=0)  # average nstart frames

    i = np.indices((nx, nx))  # to create xx and yy vectors
    yy = i[0] - nx / 2
    xx = i[1] - nx / 2

    r = np.sqrt(np.square(xx) + np.square(yy))  # distance from center: plt.imshow(xx, cmap='Greys')
    phi = np.arctan2(yy, xx)  # 2D array of phase # plt.imshow(phi, cmap='Greys')

    # ## Prepare initial wide masks
    m = 20  # max order of angular signals
    nsect = 8  # number of sectors for radius calculation
    rwt = np.zeros((nsect, nx, nx))  # for radius calculation
    fwt = np.zeros((nsect, nx, nx))  # fluxes in the sectors
    sect = 2. * np.pi / nsect  # sector width in radians

    for j in range(0, nsect):
        sector = (phi >= (sect * (j - nsect / 2))) & (phi < (sect * (j + 1 - nsect / 2)))
        fwt[j] = sector
        rwt[j] = sector * r

    test = np.zeros((nx, nx))  # test correctness of radial masks
    for j in range(0, nsect):
        test += rwt[j] * (j + 1)

    phisect = sect * (np.arange(nsect, dtype=float) - nsect / 2 + 0.5)  # sector angles
    xsect, ysect = np.cos(phisect), np.sin(phisect)

    backgr = np.median([imav[:, 0], imav[:, nx - 1]])  # left and right columns,  scalar
    tmp = imav - backgr

    itot = np.sum(tmp)  # total flux and centroids
    xc = np.sum(tmp * xx) / itot
    yc = np.sum(tmp * yy) / itot

    imavcent = np.roll(tmp, (int(-xc), int(-yc)), (1, 0))  # crude shift by integer pixel number

    radii = np.zeros(nsect)  # preliminary radii of the sectors, in pixels
    for j in range(0, nsect):
        radii[j] = np.sum(imavcent * rwt[j]) / np.sum(imavcent * fwt[j])

    dx1 = np.sum(radii * xsect) / nsect * 2.3  # accurate x-shift
    dy1 = np.sum(radii * ysect) / nsect * 2.3  # accurate y-shift

    xc += dx1  # more accurate ring center
    yc += dy1

    # FFT sub-pixel shift by -dx1,-dy1
    arg = 2 * np.pi * (dx1 * xx + dy1 * yy) / nx
    imavcent = np.fft.ifft2(np.fft.fft2(imavcent) * np.fft.fftshift(np.cos(arg) + np.sin(arg) * 1j)).real

    # re-compute radii to check the centering, should be similar
    for j in range(0, nsect):
        radii[j] = np.sum(imavcent * rwt[j]) / np.sum(imavcent * fwt[j])
    radpix = np.sum(radii) / nsect

    # Threshold the image at 0.1*max to find the ring width
    tmp = (imavcent - 0.1 * np.max(imavcent))
    tmp *= (tmp > 0)  # plt.imshow(tmp, cmap='Greys')
    radvar = np.sum(tmp * (r - radpix) ** 2) / np.sum(tmp)
    rwidth = pow(radvar, 0.5) * 2.35
    # print("Ring width [pix]: ",rwidth)

    backgr += np.median(imavcent * (r > 1.5 * radpix))  # outside-ring pixels for refined background estimate

    drhopix = 1.5 * rwidth  # mask width, replace 1.5 with parameter value in the future
    ringmask = (r >= radpix - drhopix) * (r <= radpix + drhopix)

    # ### Now build the final matrix of masks
    ncoef = 2 * nsect + 2 * (m + 1)
    maskmat = np.zeros((ncoef, nx * nx))

    for j in range(0, nsect):  # radial masks
        maskmat[j, :] = np.ndarray.flatten(rwt[j] * ringmask)  # image pixels arranged in 1D array
        maskmat[j + nsect, :] = np.ndarray.flatten(fwt[j] * ringmask)

    for j in range(0, m + 1):  # cosine ans sine masks
        tmp = np.cos(phi * j) * ringmask
        if j > 0:
            cwt = tmp - np.sum(tmp) / nx / nx  # remove residual piston
        else:
            cwt = tmp
        tmp = np.sin(phi * j) * ringmask
        swt = tmp - np.sum(tmp) / nx / nx  # remove piston
        maskmat[2 * nsect + j, :] = np.ndarray.flatten(cwt)
        maskmat[2 * nsect + m + 1 + j, :] = np.ndarray.flatten(swt)

    # ### Main loop over the cube
    coef = np.zeros((nz, ncoef))  # prepare the arrays for cube processing
    xcent = np.zeros(nz)  # x-center in each frame [pix]
    ycent = np.zeros(nz)  # y-center [pix]
    rad = np.zeros(nz)
    imav = np.zeros((nx, nx))  # average image
    x0 = xc  # current ring center
    y0 = yc

    for i in range(0, nz):  # process full cube
        tmp = data_cube[i] - backgr
        arg = 2 * np.pi * (x0 * xx + y0 * yy) / nx  # FFT centering
        tmp = np.fft.ifft2(np.fft.fft2(tmp) * np.fft.fftshift(np.cos(arg) + np.sin(arg) * 1j)).real
        imav = imav + tmp
        c = np.dot(maskmat, np.ndarray.flatten(tmp))
        c[0:nsect] = c[0:nsect] / c[nsect:2 * nsect]
        coef[i, :] = c
        radii = c[0:nsect]
        dr = np.sum(radii) / nsect
        dx = np.sum(radii * xsect) / nsect * 2.3
        dy = np.sum(radii * ysect) / nsect * 2.3
        x0 += dx
        y0 += dy
        xcent[i] = x0
        ycent[i] = y0
        rad[i] = dr
        # end of main loop

    # ### Normalization and average parameters of the cube
    imav = imav / nz  # average ring image, save as FITS file
    # hdr = fits.Header()
    # fits.writeto('avimage.fits', imav, hdr) # needs overwrite flag

    flux = np.mean(coef[:, 2 * nsect])  # average flux in the ring [ADU]
    coef[:, 2 * nsect:ncoef] *= 1. / flux  # normalize by the mean flux
    fluxvar = np.std(coef[:, 2 * nsect:ncoef])

    xc = np.mean(xcent)  # mean ring position and its variance
    xcvar = np.std(xcent)
    yc = np.mean(ycent)
    ycvar = np.std(ycent)

    cm = np.mean(coef[:, 2 * nsect + 1])  # mean cosine and sine terms to evaluate coma
    sm = np.mean(coef[:, 2 * nsect + m + 2])
    coma = pow((cm ** 2 + sm ** 2), 0.5)
    angle = 180 / np.pi * np.arctan2(sm, cm)

    contrast = np.zeros(nsect)  # Analog of Strehl ratio in each sector
    for j in range(0, nsect):
        tmp = imav * fwt[j]
        contrast[j] = np.max(tmp) / np.sum(tmp)

    contrast = np.mean(contrast)
    meanrad = np.mean(rad)
    impar = [backgr, flux, fluxvar, meanrad, rwidth, xc, yc, xcvar, ycvar, coma, angle, contrast]  # list object

    tmp = imav / np.sum(imav)  # noise coef. of angular coefficients
    t0 = np.sum(tmp * ringmask)
    noise1 = np.sum(ringmask ** 2 * tmp) / t0
    noise2 = np.sum(ringmask ** 2) / t0

    tmp2 = ringmask * (r - radpix)  # noise coef. of radii
    noise1r = np.sum(tmp * tmp2 ** 2)
    noise2r = np.sum(tmp2 ** 2)
    noisepar = [noise1, noise2, noise1r, noise2r]

    # ###  Calculation of statistical moments
    # Compute differential radius variance
    dr = coef[:, 0:int(nsect / 2)] + coef[:, int(nsect / 2):nsect]  # 4 DIMM-like signals (2000,4)
    drvar = np.var(dr, axis=0)  # 4 variances in pix^2
    meanrvar = np.mean(drvar)

    # radius noise from difference of successive dr values in 4 pairs of opposite sectors
    ddr = dr - np.roll(dr, 1, 0)  # difference (2000,4)
    drnoise = np.var(ddr, axis=0)  # 4 noise variances, pix^2

    # Variance and covariance of angular coefficients
    acoef = coef[:, 2 * nsect:ncoef]  # (2000,42)
    varcoef = np.var(acoef, axis=0)  # 42 variances for m=20
    meancoef = np.mean(acoef, axis=0)
    tmp = acoef * np.roll(acoef, 1, 0)  # shift-1 product
    tmp = tmp[1:nz, :]  # discard first element
    covar = np.sum(tmp, axis=0) / (nz - 1) - meancoef ** 2

    # add cosine and sine variances and covariances
    power = varcoef[0:m + 1] + varcoef[m + 1:2 * m + 2]
    cov = covar[0:m + 1] + covar[m + 1:2 * m + 2]


    # Mean coefficients for aberrations
    mcoef = np.zeros(nsect+6) # mean radii and m=1,2,3 cos/sine terms
    mcoef[0:nsect] = np.mean(coef[:,0:nsect],axis=0)
    mcoef[nsect:nsect+3] = meancoef[1:4]
    mcoef[nsect+3:nsect+6] = meancoef[m+2:m+5]

    # Plot variance and covariance in log-scale
    # arg = range(m + 1)
    # plt.plot(arg, np.log10(power))
    # plt.plot(arg, np.log10(cov))

    # ### Test of json output

    # json_obj = json.dumps(cov.tolist())
    # print(json_obj)

    # 
    moments = {'var': power.tolist(), 'cov': cov.tolist(),'rnoise': drnoise[0],'rvar': meanrvar,'mcoef':mcoef.tolist()}    
    data = {'image': {'impar': impar, 'noisepar': noisepar}, 'moments': moments}

    return data, imav

## ----    Read and process the cube file
## Input: fits cube, directory where it lives, output directory; directories end with '/'
## Returns name of the data-dictionary file
def cubeproc(file, indir, outdir):

# Read the data and header
    try:
        hdul = fits.open(indir+file)  # hdul.info() to see what it contains
    except FileNotFoundError as err:
        print(err)
        sys.exit()
    data_cube = hdul[0].data
    hdr = hdul[0].header
    hdul.close()

    basename = file.replace('.fits','')  # remove .fits and optionally _cube from the filename
    if '_cube' in basename:
        basename = basename.replace('_cube','')

# extract prameters from the header into a dictionary
    nx  = int(hdr["NAXIS1"])
    nz  = int(hdr["NAXIS3"])
    texp = float(hdr["EXPOSURE"])*1e-6  # in s
    gain = float(hdr["GAIN"])
    DATE = hdr["DATE-OBS"]
    try: 
        star = hdr["STAR"]
    except:
        star=''
    cubepar = {'nx': nx,'nz':nz,'texp':texp,'gain':gain,'date':DATE,'star':star}


    datafilename = basename + '.json'
    json_output_file = outdir + datafilename

    # Crunch the cube
    Data_dict, imav = Moments(data_cube)

    # Save the average image
    try:
        fits.writeto(outdir+basename+'_av.fits', imav, hdr, overwrite=True)
    except OsError as err:
        print(err)    

    # Save the data dictionary and return its name
    Data_dict['cubepar'] = cubepar
    try:
        json.dump(Data_dict, codecs.open(json_output_file, 'w', encoding='utf-8'), separators=(',', ':'),
                  sort_keys=True,indent=4)
    except FileNotFoundError as err:
        print('{}:{}'.format(err, json_output_file))

    return datafilename
    
# ------------ end of definitions

# ### Main module. usage: > python <par> <data>
""" This condition is to control code execution when file is imported """
if __name__ == "__main__":

    print("Cube processing...")
    if len(sys.argv) < 4:
        print("Usage: python cube2.py fitsfile indir/ outdir/")
        sys.exit()

    fitsfile = sys.argv[1]    
    indir = sys.argv[2]
    outdir = sys.argv[3]
    print('fits file:', fitsfile)

    cubeproc(fitsfile,indir,outdir)
    print("Success!!")
