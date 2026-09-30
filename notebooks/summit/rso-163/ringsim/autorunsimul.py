'''
Auth: Diego H.

Autoruns Andrei's simulation
'''

import argparse
import d_testsimul
import numpy as np
import time


def main():
    
    
    # p = argparse.ArgumentParser()
    # p.add_argument('times', type=int, help='How many times to run the simulation')
    # args = p.parse_args(), t = args.times
    
    see_range = np.arange(0.4, 2.6, 0.1)
    
    print(f'Running {len(see_range)} simulations')
    
    i = 0
    times = []
    
    for see in see_range:
        start_time = time.time()
        seed0 = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        d_testsimul.main(['--seeing', f'{see}', '--seed0', f'{seed0}', '--verbose', '', '--display', ''])  # verb=False, display=False
        secs = round((time.time() - start_time),1)
        print(f'Sim {i+1}/{len(see_range)} with Seeing: {round(see, 2)} in {secs} seconds')
        times.append(secs)
        i += 1
    
    print(f'Autorun finished in {np.sum(times)/60}\"! \n Mean run {np.mean(times)}"')


if __name__ == '__main__':
    main()
    