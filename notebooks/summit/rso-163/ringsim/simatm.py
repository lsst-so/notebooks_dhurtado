#!/usr/bin/env python3
'''
Creates the atmosphere for simulation
of ring images.


Auth: A. Tokovinin
Translated: D. Hurtado

'''
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sci
from tqdm.notebook import tqdm
import argparse



def main():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('pixel', type=float,
                   help='TBD')
    p.add_argument('r0', type=float,
                   help='Fried parameter from seeing')
    p.add_argument('--lamb', dest='lamb', type=float, default=0.5,
                   help='Wavelength of reference light, in meters, def=0.5')
    p.add_argument('--ngrid', dest='ngrid', type=float, default=512,
                   help='Half size of the atmosphere grid to simulate, def=512')
    p.add_argument('--zlow', dest='zlow', type=float, default=500,
                   help='Low layer altitude, def=500')
    p.add_argument('--zhigh', dest='zhigh', type=float, default=10500,
                   help='High layer altitude, def=10500')
    p.add_argument('--fhigh', dest='fhigh', type=float, default=0.1,
                   help='Fraction of high layer (?), def=0.1')
    p.add_argument('--debug', dest='debug_str', type=str, default='DEBUG',
                   help='Debug level, logger accepted values, def=INFO')
    
    args = p.parse_args()
    
    #Is there a way to make this easier?
    pixel = args.pixel
    r0 = args.r0
    lamb = args.lamb
    ngrid = args.ngrid
    zlow = args.zlow
    zhigh = args.zhigh
    fhigh = args.fhigh
    debug_str = args.debug_str
    
    logger = logging.getLogger("Simulation")
    logger.setLevel(debug_str)
    
    print('Atmosphere simulation, parameters calculated given below.\n')
    if 'seed0' in globals() and seed0 is not None:       # If seed0 is provided, use it. Otherwise, let the OS provide a random one
        rng = np.random.default_rng(seed0)
        logger.debug(f'Simulation on fixed seed: {seed0}.')
    else:
        rng = np.random.default_rng()
        logger.debug('Simulation on random seed.')
    
    # Check for consistency
    if zlow > zhigh:
        raise ValueError(f"Inconsistency: zlow ({zlow}) can't be greater than zhigh ({zhigh})")
        #return
    
    size = 2 * ngrid * pixel                                 # 2 times the size of the chosen grid in arc-seconds or meters
    
    # Starting parameters
    lamb_ref = 0.5                                                # reference wavelength
    r0_ref = r0 * (lamb_ref / lamb) ** (1.2)                            # scaling r0 to 500nm
    tint0 = (r0_ref ** ( -5 / 3) / 0.423) * (0.5 * lamb_ref/ np.pi) ** 2            # correct total turbulence integral for given wavelength
    tint1 = tint0 * fhigh                                               # high-layer integral
    tint2 = tint0 * (1 - fhigh)                                          # low-layer integral
    r01 = (0.423 *((0.5 * lamb /np.pi ) ** (-2)) * tint1) ** (-3 / 5)              # high fried parameter
    r02 = (0.423 *((0.5 * lamb /np.pi ) ** (-2)) * tint2) ** (-3 / 5)              # low fried parameter
    
    see = (tint0 / 6.83e-13) ** (0.6) # simulated seeing @ given wavelength
    
    print(f'Grid size arcsec: {size} \n Fried parameters, meters (low, high): {r02, r01} \n',
                f'Turbulence integrals, m^1/3: {tint2,tint1} \n Altitudes, meters: {zlow, zhigh}\n',
                f'Starting seeing: {see} @ {lamb} microns\n')
    logger.debug(f'tint0: {tint0}, r0: {r0}')
    
    
    # Phase simulations
    fact1 = np.sqrt(0.023)*(size/r01)**(5/6)
    fact2 = np.sqrt(0.023)*(size/r02)**(5/6)
    
    if fhigh == 0:
        fact1 = 0
    
    
    # Coord grid for atmosphere
    N = 2 * ngrid
    x = np.linspace(-ngrid, ngrid - 1, N)
    y = np.linspace(-ngrid, ngrid - 1, N)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    r[ngrid, ngrid] = 1e-4 # Arbitrary near-zero value for non-zero division 
    
    
    # Fresnel Filters
    farg = (np.pi*lamb*r**2)/(size**2)
    
    # Simulate the high layer
    if fhigh > 0:
        logger.debug(f'fhigh = {fhigh}')
        print('Simulating high screen')
        
        noise_real = rng.standard_normal((2*ngrid, 2*ngrid))                # create the noise screens
        noise_imag = rng.standard_normal((2*ngrid, 2*ngrid))
        complex_noise = noise_real + 1j * noise_imag
    
        #with np.errstate(divide='ignore'):
        temp = fact1 * (r**(-11/6)) * complex_noise                         # kolmogorov Power Law (r^-11/6)
        temp[ngrid, ngrid] = 0.0 + 0.0j
    
        temp_spatial = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(temp)))# spatial domain (ifft2 is the 2D Inverse FFT)
        phase_screen = temp_spatial.real
    
        u1 = np.exp(1j * phase_screen)                                      # convert to Complex Amplitude
    
    
    # Propagate to low layer, with Angular Spectrum method
    if (zhigh-zlow) > 0:
        logger.debug(f'zhigh: {zhigh}, zlow:{zlow}, difference: {zhigh-zlow}')
        print('Propagating to low layer')
    
        u1_freq = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))    # transform the high layer to Freq Domain
        transfer_function = np.exp(-1j * farg * (zhigh-zlow))           # apply the Fresnel Transfer Function (farg earlier)
        temp = transfer_function * u1_freq
        u1 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(temp)))      # back to Spatial domain
    
    else:
        
        u1 = np.ones((2*ngrid, 2*ngrid), dtype=complex)                 # if no turbulence, the amplitude is flat (np.ones)
        print('No turbulence computed')
    
    # Simulate turbulence in low layer
    print('Simulating low layer')
    noise_low = rng.standard_normal((2*ngrid, 2*ngrid)) + 1j * rng.standard_normal((2*ngrid, 2*ngrid))
    
    #with np.errstate(divide='ignore'):
    temp_low = fact2 * (r**(-11/6)) * noise_low
    temp_low[ngrid, ngrid] = 0.0 + 0.0j
    
    phase_low = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(temp_low))).real # transform to Spatial domain
    u1 *= np.exp(1j * phase_low)                                               # acummulate layers
    
    # Propagate to ground
    print('Propagating to ground \n')
    u1_freq = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
    transfer_ground = np.exp(-1j * farg * zlow)                                # propagate over zlow
    tmp_ground = transfer_ground * u1_freq
    
    u1 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(tmp_ground)))           # Back to Spatial domain
    
    # Rytov number
    intensity = np.abs(u1)**2                                                  # Scintillation
    scind = np.mean((intensity - 1.0)**2)                                      # Index
    rytov = 19.22 * (lamb**(-7/6)) * (zlow**(5/6) * tint2 + zhigh**(5/6) * tint1)
    
    print(f"Rytov variance (Theoretical): {rytov}")
    print(f"Scintillation Index (Simulated): {scind}\n")
    
    np.savez('atm_simulation.npz', 
             u1=u1, ngrid=ngrid, pixel=pixel, lamb=lamb, 
             see=see, r0=r0, fhigh=fhigh, 
             zhigh=zhigh, zlow=zlow)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(intensity, cmap='gray')
    plt.title('Simulated atmosphere')
    plt.colorbar(label='Intensity')
    plt.savefig('atmsim.jpg', dpi=300, format='jpg')

if __name__ == '__main__':
    main()