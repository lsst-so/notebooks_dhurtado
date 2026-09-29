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
        if i % 5 == 0: print(f'{i}/{t}')
        d_testsimul.main(['--verbose', '', '--display', ''])  # verb=False, display=False
    
    print(f'Simulation ran {t} times!')


if __name__ == '__main__':
    main()
    