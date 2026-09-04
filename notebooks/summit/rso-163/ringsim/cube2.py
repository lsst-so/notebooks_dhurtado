#Notes:
    #Changed input from single fits file containing multiple frames to single folder containing multiple fits files, all with one frame each
    
    #Changed nz (frames per cube) and nx (image size, square) from being extracted out of one fits file with multiple frames to being extracted from information about all collected single-frame fits files in the code
    #Changed date format to match Tokovinin's (ISO 8601 standard), UTC+0 is our timezone chosen for convention (about -13 hours from our timezone, so allows us to keep all observations from the same nigth under the same date)
    #Changed FWHM multiplier for noise mask to parameter (orignally 1.5)



""" Based on cubecoef.pro (Tokovinin) """
# version 1.0

from astropy.io import fits
import numpy as np
# import matplotlib.pyplot as plt
import json
import codecs
import sys
import os
import glob
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt

#Optional settings
radial_mask_plots = False #Toggle display of plots showing the radial mask overlaid with the intensity
radial_dist = 25  # pixels | amount of pixels radially outwards shown on plot

#Set paths
parameters_path = 'par-ardmore.json' #Relative parameters file path

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

    #Get mask multiplier value
    parameters_file = open(parameters_path, 'r') #Open parameters file
    parameters = json.load(parameters_file) #Load parameters
    mask_multiplier = parameters.get("profrest", {}).get("maskm") #Extract mask multiplier

    drhopix = mask_multiplier * rwidth  # mask width, replace 1.5 with parameter value in the future
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

    moments = {'var': power.tolist(), 'cov': cov.tolist(),'rnoise': drnoise[0],'rvar': meanrvar,'mcoef':mcoef.tolist()}    
    data = {'image': {'impar': impar, 'noisepar': noisepar}, 'moments': moments}
    
    if radial_mask_plots == True: #Display radial mask and intensity plots if wanted
        #Get dimensions
        size = imav.shape[0] #Find size
        centre = size // 2 #Find centre
    
        #Get intensity profile (x-axis cross-section)
        cross_section = np.arange(-radial_dist + centre, radial_dist + 1 + centre) #Set cross-section
        y_values = np.full(len(cross_section), centre) #Set constant y-values
        
        intensity_profile = imav[y_values, cross_section] #Extract 1D intensity profile
        ringmask_profile = ringmask[y_values, cross_section] #Extract 1D ringmask profile
        
        #Adjustments for plotting
        intensity_profile = intensity_profile / intensity_profile.max() #Normalise intensity
        ringmask_profile = 1 - ringmask_profile #Invert mask for plot filling
        
        #Plot intnesity profile
        plt.plot(cross_section - centre, intensity_profile, label = 'Intensity Profile', color = 'steelblue') #Plot intensity profile
        plt.fill_between(cross_section - centre, ringmask_profile, step = 'mid', alpha = 0.3, color = 'red', label = 'Radial Mask') #Plot ringmask profile
        plt.xlabel('Radial Distance [pixels]') #Label x-axis
        plt.ylabel('Normalised Intensity') #Label y-axis
        plt.legend() #Add legend
        plt.tight_layout() #Minimise overlap
        plt.show()
    
    return data, imav

## ----    Read and process the cube file
## Input: fits cube, directory where it lives, output directory; directories end with '/'
## Returns name of the data-dictionary file
def cubeproc(indir, outdir, single_number):
    
# Read the data and header
    fits_files = glob.glob(os.path.join(indir, '*.fits')) #Finds fits files in specified folder

    def extract_file_number(filename):
        fits_basename = os.path.basename(filename) #Extracts just file name from path
        
        match = re.search(r'(\d+)(?=\.fits$)', fits_basename)  #Extracts the number at the end of the file name
        
        return int(match.group(1)) if match else -1 #Returns number if found, otherwise sends it to the start of the list
    
    fits_files.sort(key=extract_file_number) #Orders the fits files
    
    frames_data = []
    hdr = None
    
    for fits_file in fits_files:
        if fits_file[0]:
            try:
                hdul = fits.open(fits_file)  # hdul.info() to see what it contains
            except FileNotFoundError as err:
                print(err)
                sys.exit()
            frames_data.append(hdul[0].data)
            hdr = hdul[0].header
            hdul.close()
        else:
            try:
                hdul = fits.open(fits_file)  # hdul.info() to see what it contains
            except FileNotFoundError as err:
                print(err)
                sys.exit()
            frames_data.append(hdul[0].data)
            hdul.close()
    
    data = np.array(frames_data)
    min_side = max(data.shape[1], data.shape[2]) #Find minimum size of frames

    data_cube = data
    
    basename = os.path.basename(os.path.normpath(indir))  #Create main name to identify files

# extract prameters from the header into a dictionary
    nx  = min_side #Image size, square
    nz  = data.shape[0] #Frames per cube
    texp = float(hdr["EXPTIME"]) #Exposure time (in s)
    gain = float(hdr["GAIN"]) #Camera gain
    DATE = (datetime.strptime(hdr["DATE-OBS"][:26], "%Y-%m-%dT%H:%M:%S.%f") + timedelta(seconds = round(datetime.strptime(hdr["DATE-OBS"][:26], "%Y-%m-%dT%H:%M:%S.%f").microsecond / 1_000_000))).replace(microsecond = 0).strftime("%Y-%m-%dT%H:%M:%S") #Date of observation, reformated into ISO 8601 standard
    star = int(hdr["OBJECT"][hdr["OBJECT"].rfind("(")+1 : hdr["OBJECT"].rfind(")")].strip()) #HR number
    
    cubepar = {'nx': nx,'nz':nz,'texp':texp,'gain':gain,'date':DATE,'star':star}
    
    #Set output paths
    if single_number == '0':
        datafilename = basename + '.json'
        json_output_file = outdir + datafilename
        
        avg_output = outdir+basename+'_avg.fits'
    else:
        datafilename = basename + ' ((' + single_number + ')).json'
        json_output_file = outdir[:-1] + ' ((moments))/' + datafilename
        
        avg_output = outdir[:-1]+' ((previews))/'+basename+'_avg ((' + single_number + ')).fits'

    # Crunch the cube
    Data_dict, imav = Moments(data_cube)

    # Save the average image
    try:
        fits.writeto(avg_output, imav, hdr, overwrite=True)
    except OSError as err:
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

    #print("Cube processing...")
    if len(sys.argv) < 3:
        print("Usage: python cube2.py indir outdir")
        sys.exit()
  
    indir = sys.argv[1]
    outdir = sys.argv[2]+'/'
    #print('fits files folder:', indir)
    
    if len(sys.argv) > 3: #Check coming from autoprocessing2.py and adjust accordingly
        single_number = sys.argv[3]
    else:
        single_number = '0'
    
    cubeproc(indir, outdir, single_number)
    #print("Success!!")