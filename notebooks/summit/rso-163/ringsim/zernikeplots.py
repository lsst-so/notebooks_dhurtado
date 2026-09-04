#!/usr/bin/env python3
"""
Read Zernike coefficient files (plain text) from a folder and plot
histograms of chosen Zernikes across all files.
Expected file format (any extension, or no extension):
    4   123.4
    5    45.6
    6    -12.3
    ...
---------------------------
    Necessary parameter
        --folder (path)
---------------------------
Auth: Diego H. / Suzakuu
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Match lines like: "Z4  123.4" or "Z11 -55.2"
LINE_RE = re.compile(r"^\s*(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_zernike_file(path):
    """Return {zernike_index: value} from one text file."""
    coeffs = {}
    with open(path, 'r') as f:
        for line in f:
            m = LINE_RE.match(line)
            if m:
                idx = int(m.group(1))
                val = float(m.group(2))
                coeffs[idx] = val
    return coeffs


def collect_zernikes(folder, pattern='*'):
    """Read all matching files in folder and return dict {Zi: [values...]}."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    # Skip images, PDFs, etc.
    skip_ext = {'.jpg', '.jpeg', '.png', '.pdf', '.fits', '.fit',
                '.tif', '.tiff', '.log'}
    files = [f for f in files
             if os.path.isfile(f)
             and os.path.splitext(f)[1].lower() not in skip_ext]

    print(f'Reading {len(files)} candidate files from {folder}')
    all_coeffs = {}
    n_ok = 0
    for f in files:
        try:
            coeffs = parse_zernike_file(f)
        except Exception as e:
            print(f'  [SKIP] {os.path.basename(f)}: {e}')
            continue
        if not coeffs:
            continue
        n_ok += 1
        for idx, val in coeffs.items():
            all_coeffs.setdefault(idx, []).append(val)

    print(f'  Parsed {n_ok} files with valid Zernike content.')
    return all_coeffs


def plot_histograms(all_coeffs, zernikes, output_folder,
                    bins=20, prefix='zernike_hist'):
    """Save one histogram per requested Zernike."""
    for z in zernikes:
        values = np.array(all_coeffs.get(z, []))
        if values.size == 0:
            print(f'  [WARN] No data for Z{z}, skipping.')
            continue

        mean = np.mean(values)
        std  = np.std(values)
        med  = np.median(values)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(values, bins=bins, color='steelblue', edgecolor='black',
                alpha=0.85)
        ax.axvline(mean, color='red',  linestyle='--',
                   label=f'mean = {mean:.2f}')
        ax.axvline(med,  color='green', linestyle=':',
                   label=f'median = {med:.2f}')
        ax.set_xlabel(f'Z{z} (nm)')
        ax.set_ylabel('Count')
        ax.set_title(f'Z{z} distribution  (N = {values.size},  σ = {std:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(output_folder, f'{prefix}_Z{z}.jpg')
        fig.savefig(out_path, dpi=150, format='jpg')
        plt.close(fig)
        print(f'  Saved: {out_path}   (N={values.size}, '
              f'mean={mean:.2f}, std={std:.2f})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True,
                   help='Folder with Zernike text files')
    p.add_argument('--zernikes', nargs='+', type=int,
                   default=[4, 5, 7, 11],
                   help='Zernike indices to plot (e.g. 4 5 7 11)')
    p.add_argument('--pattern', default='*',
                   help='Glob pattern for input files (default: *)')
    p.add_argument('--bins', type=int, default=20,
                   help='Number of histogram bins')
    p.add_argument('--prefix', default='zernike_hist',
                   help='Prefix for output JPG filenames')
    args = p.parse_args()

    all_coeffs = collect_zernikes(args.folder, args.pattern)

    if not all_coeffs:
        print('No Zernike data parsed — nothing to plot.')
        return

    print(f'\nMaking histograms for Z{args.zernikes} …')
    plot_histograms(all_coeffs, args.zernikes, args.folder,
                    bins=args.bins, prefix=args.prefix)
    print('\nDone.')


if __name__ == '__main__':
    main()