# Cross-correlate image with itself
import numpy as np
import sys
from astropy.io import fits

# Returns coordinates of the ring in image img by template correlation
# rad and dr are radius and half-width of the template ring
def corcent(img,rad,dr):  
    nx = img.shape[0] # array dimension
    i = np.indices((nx, nx))  # to create xx and yy vectors
    # yy = i[0] - nx / 2 ;  xx = i[1] - nx / 2
    # distance from template center
    r = np.sqrt(np.square(i[0] - nx /2) + np.square(i[1] - nx / 2))  
    tmpl = np.zeros([nx,nx])
    tmpl[ (r > rad-dr) & (r < rad+dr)] = 1.

    ccf = np.fft.ifft2(np.fft.fft2(img) * np.conjugate(np.fft.fft2(tmpl)))    
    ccf = np.fft.fftshift(ccf.real) # CCF centered in the image center
    s = np.where(ccf == np.max(ccf)) # find maximum
    xcent = s[1][0]
    ycent = s[0][0]
    print("Ring center at X={}, Y={}".format(xcent, ycent))    
    return xcent, ycent

# Main module. usage: > python corel.py <fits-file>
# ---------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python corcent.py <fits-file>, rad, dr ")
        sys.exit()

    fitsfile = sys.argv[1]
    print("Image file: " + fitsfile)    
    try:
        hdul = fits.open(fitsfile)  # hdul.info() to see what it contains
    except FileNotFoundError as err:
        print(err)
        sys.exit()
    img = hdul[0].data    
    hdul.close()
    img = np.array(img,float)
    rad = float(sys.argv[2])
    dr = float(sys.argv[3])


    xcent, ycent = corcent(img,rad,dr)
    print("Success")
