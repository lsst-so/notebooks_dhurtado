# RINGSS Simulation

A Python port of A. Tokovinin's IDL RINGSS code (`testsimul.pro` and the
reduction tools). It simulates a defocused annular star image seen through a
two-layer turbulent atmosphere, extracts the angular/radial statistics from the
image cube, and restores the vertical turbulence profile (Cn² per layer),
seeing, wind speed, coherence time and isoplanatic angle.

Original algorithms: A. Tokovinin. Python translation: D. Hurtado.

---

## 1. What this folder contains

**Pipeline stages** (each is an importable module with a worker function + a CLI):

| Module | Function | Description |
|---|---|---|
| `simatm.py` | `simatm(...)` | Two-layer complex-amplitude atmosphere screen. Fresnel-propagates phase screens to the ground and writes `atm.npz`. → `images/atmsim.jpg` |
| `ringsim.py` | `ringsim(...)` | Builds the ring-image data cube from `atm.npz` (aperture, defocus, photon + readout noise), writes it to a FITS file. → `ringsim.gif`, `images/imaverage.jpg` (+ debug jpgs) |
| `cubecoef.py` | `cubecoef(...)` | Centers each frame and projects it onto the sector / angular masks → `(impar, coef)`. → `cube_centroid.gif`, `images/centroid_drift.jpg` |
| `statmom.py` | `statmom(...)` | Statistical moments of the coefficients (angular power spectrum, covariance, radius variance) → `(par, data)`. → `images/angular_spectrum.jpg` |
| `weights.py` | `computeweight(...)` (`aweight` inside) | Forward model: weighting functions linking turbulence to the observables → `weights.json` |
| `profrest.py` | `profrest.main(...)` | Inverse step: restores the Cn² profile and the derived atmospheric parameters (NNLS fit). → `images/power_spectrum.jpg` |

**Drivers:**

- `testsimul.py` — Full chain for ONE simulation (`simatm → ringsim → cubecoef → statmom → computeweight → profrest`), prints the report and appends one row to `testsimul_results.csv`.
- `autorunsimul.py` — Runs `testsimul` many times over a seeing sweep (silent), each with a random seed and wind, accumulating rows in the CSV.

**Notebooks:**

- `testsimul.ipynb` — Interactive full pipeline, one cell per stage, images shown inline.
- `simanalysis.ipynb` — Reads `testsimul_results.csv` (pandas) and makes the analysis plots (seeing vs input, angular power spectrum, turbulence-profile waterfall, etc.).

**Support / config:**

- `zernike.py` — Zernike polynomial toolbox, used by `weights.py` and written by Andrei.
- `weights.json` — Pre-computed weighting functions (one per instrument config).
- `zmat.json` — Saturation (Z-matrix) correction, fixed.

**Generated outputs** (created when you run the pipeline):

- `atm.npz` — The simulated atmosphere screen (`simatm` output).
- `test.fits` — The ring-image cube (`ringsim` output).
- `testsimul_results.csv` — One row per simulation: inputs + all restored results + the angular power/covariance spectra. Appended to.
- `images/` — All diagnostic plots (`*.jpg`).
- `ringsim.gif`, `cube_centroid.gif` — Live-preview animations.

**Reference material:**

- `andrei/`, `adam/` — Upstream Python (on-sky reduction / field orchestration) from A. Tokovinin and A. Dobbyn. Not used by this pipeline; kept for reference.
- `testsimul.pro` — The original IDL driver this code is ported from.

---

## 2. Requirements

Python 3.8+ with: `numpy`, `scipy`, `astropy`, `matplotlib`, `pandas` (for `simanalysis`).
The notebooks also use IPython display. No build step is needed.

---

## 3. How to run

There are three equivalent ways to run the simulation. All of them execute the
same stage functions; they differ only in how you drive them.

### (A) Interactively, in the notebook — `testsimul.ipynb`

Open `testsimul.ipynb` and run the cells top to bottom. The first cells set the
instrument and atmosphere parameters; each following cell calls one pipeline
stage (`simatm`, `ringsim`, `cubecoef`, ...) and displays its image(s) inline.
This is the best way to inspect each stage visually and to change parameters by
editing the parameter cells. The last cell appends a row to
`testsimul_results.csv`.

### (B) One run from the command line — `testsimul.py`

Runs the whole chain once with the given (or default) parameters:

```bash
python testsimul.py
```

Override any parameter with a flag, e.g.:

```bash
python testsimul.py --seeing 1.3 --starmag 1.5 --wind 8
```

Turn off text / images for a fast, quiet run (note: these flags are truthy for
any non-empty string, so pass an empty string to disable):

```bash
python testsimul.py --verbose '' --display ''
```

Each run appends one row to `testsimul_results.csv`. When `--display` is on, the
diagnostic plots are (re)written to `images/` and the GIFs to the folder.

**Available flags** (default in parentheses):

*Instrument / detector*

| Flag | Meaning |
|---|---|
| `--diam` (0.304) | aperture diameter, m |
| `--effl` (1.20845) | effective focal length, m |
| `--eps` (0.7) | central obstruction |
| `--pixel` (0.017664) | grid sampling, m per grid pixel (sets screen size) |
| `--pixsize` (6.9e-6) | physical detector pixel size, m |
| `--wavelen` (0.6e-6) | reference wavelength, m |
| `--pdist` (1050) | defocus as conjugation height, m |
| `--drad` (1.5) | ring-mask width, λ/(0.5·D·(1−eps)) units |
| `--mmax` (20) | max order of angular signals |
| `--nsect` (8) | number of sectors for the radius calculation |
| `--interpol` (1) | use sub-pixel (Fourier) shifts |
| `--ron` (0) | readout noise, electrons |
| `--gain` (0) | camera gain setting |

*Atmosphere / observation*

| Flag | Meaning |
|---|---|
| `--seeing` (1) | seeing in arcsec at 0.5 micron |
| `--wind` (10) | wind speed, m/s |
| `--zlow` (500) | low-layer altitude, m |
| `--zhigh` (10500) | high-layer altitude, m |
| `--highfrac` (0.1) | fraction of turbulence in the high layer |
| `--starmag` (2) | star magnitude |
| `--seed0` (52403) | fixed RNG seed for reproducible runs |
| `--ngrid` (512) | half size of the atmosphere grid |

*Output control*

| Flag | Meaning |
|---|---|
| `--display` (True) | save/update the diagnostic images in `images/` |
| `--verbose` (True) | print progress text to the terminal |

### (C) Many runs / a seeing sweep — `autorunsimul.py`

Runs `testsimul` repeatedly over a built-in seeing sweep (0.4 → 2.5 arcsec in
0.1 steps), each with a fresh random seed and a random wind, silently
(verbose/display off), and reports the per-run and mean time:

```bash
python autorunsimul.py
```

To accumulate more statistics (repeats for the error bars in `simanalysis`),
launch it several times with a shell loop — **not** with commas:

```bash
for i in 1 2 3; do python autorunsimul.py; done
```

Every simulation appends its row to `testsimul_results.csv`, so repeated sweeps
build up multiple samples per seeing value.

---

## 4. Analysing the results — `simanalysis.ipynb`

After one or more runs, open `simanalysis.ipynb`. It loads `testsimul_results.csv`
with pandas (`df_results = pd.read_csv('testsimul_results.csv')`) and produces the
analysis plots: measured vs input seeing, the angular power spectrum (with error
bars over repeats), and the turbulence-profile "waterfall" (altitude Z vs sample,
colored by the layer turbulence integral J). Columns are accessed by name, e.g.
`df_results['see_arcsec']`, the per-mode spectra as `power_m_0 .. power_m_20`, and
the profile as `Z_km_0 .. Z_km_7` / `J_1e13_0 .. J_1e13_7`.

**Output of interest:**

- `testsimul_results.csv` — the tabulated results (one row per simulation)
- `images/*.jpg` — the per-stage diagnostic plots
- `ringsim.gif`, `cube_centroid.gif` — simulation / centering previews
