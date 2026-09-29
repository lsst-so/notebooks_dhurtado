# This script auto-runs and inputted X amount of times
# testsimul, captures the outputs and stores them in a .csv

'''
Auth: Diego H.

Autoruns Andrei's simulation
'''

import argparse
import d_testsimul


def main():
    
    p = argparse.ArgumentParser()
    
    # Telescope / detector (sim1.par)
    p.add_argument('times', type=int,
                   help='How many times to run the simulation')
    
    args = p.parse_args()
    
    t = args.times
    
    print(f'Running {t} simulations')
    
    for i in range(0,t):
        d_testsimul.main([])
        if i % 10 == 0:
            print(f'{i}/{t}')
    
    print(f'Simulation ran {t} times!')


if __name__ == '__main__':
    main()
    