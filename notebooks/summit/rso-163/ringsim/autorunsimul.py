'''
Auth: Diego H.

Autoruns Andrei's simulation
'''

import argparse
import testsimul
import numpy as np
import time
import pandas as pd


def main():
    
    df_seeing = pd.read_csv('testsimul_results.csv')
    see_range = df_seeing['input_seeing_arcsec']
    see_range = see_range[see_range <= 4]   # skip runs with seeing > 4"
    #see_range = np.arange(0.4, 2.6, 0.01)
    
    print(f'Running {len(see_range)} simulations')
    print(f'Est: {(len(see_range)*90)/60} minutes')
    
    i = 0
    times = []
    
    for see in see_range:
        see = float(see)
        start_time = time.time()
        seed0 = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        wind = np.random.randint(1, 21)
        
        testsimul.main(['--seeing', f'{see}', '--wind', f'{wind}' ,'--seed0', f'{seed0}', '--verbose', '', '--display', ''])  # verb=False, display=False
        secs = round((time.time() - start_time),1)
        if i%(len(see_range)/10) == 0: print(f'Sim {i+1}/{len(see_range)} with Seeing: {round(see, 2)}" and Wind: {wind} m/s in {secs} seconds')
        times.append(secs)
        i += 1
    
    print(f'Autorun finished in {np.sum(times)/60}\'!')
    print(f'Mean run {np.mean(times)}"')


if __name__ == '__main__':
    main()
    