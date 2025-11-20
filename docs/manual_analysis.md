# Manual Analysis Module

The `brillouin_spectra_analyzer_manual` module lets you inspect a single spectrum to tune peak-finding parameters and then run the same settings over an entire 3D dataset. Use it right after parsing data with `parse_brillouin_set`.

## Typical Workflow

1) Parse spectra into a 3D array.  
2) Call `analyze_brillouin_spectrum_manual` on a single pixel to pick laser/Brillouin windows and fitting options.  
3) Reuse those parameters in `analyze_brillouin_spectra_manual` for a full slice or volume.  
4) Visualize and segment the results with the plotting and cell-selection utilities.

## Single spectrum: `analyze_brillouin_spectrum_manual`

Analyzes one spectrum from the cube (or a provided `spectrum_data`) and optionally plots the fit.

**Key arguments**
- Coordinates: `x_coord`, `y_coord`, `z_coord` select the pixel; omit to inspect a random spectrum. `spectrum_cut_range` trims the waveform before processing.
- Spectral geometry: `free_spectral_range` fits the quadratic VIPA calibration. Windows for peaks come from `laser_peak_ranges` (index windows or exact indices) and `brillouin_peak_ranges` (pairs of index windows or a tuple `(a_GHz, b_GHz)` around each laser).
- Fitting: set `fit_lorentzians>0` to fit Lorentzian peaks inside a ±window (GHz). Control bounds via `fwhm_bounds_GHz` and `center_slack_GHz`. `match_brilouin_parameters=True` locks all left/right Brillouin peaks to shared width/shift.
- Filtering and weighting: provide `filter_settings` (Savgol, wavelet, Fourier, PCA, or custom callables) to denoise before peak finding; enable `baseline_spectra` or `poisson_weighting` if you need them.
- Debugging and plots: `debug_plot` overlays search windows, `plot_PCA`/`pca_*` help to locate laser peaks, `make_plot` controls size, and `save_plot` writes the figure.

**Outputs**

Returns a dictionary containing:
- `df`: per-peak rows with type, shift, FWHM, center/amplitude, and fit RMSE.
- `spectrum`, `rescaled_x_axis`, and the detected `laser_peaks_indices` / `brillouin_peaks_indices`.
- `fit_params` `(a, b, c)` from the VIPA calibration plus lists of per-peak shifts, FWHMs, amplitudes, and centers.
- Flags showing which options were enabled (baseline, weighting, matching, etc.).

Example:

```python
from brillouin_analyzer_src import *
%load_ext autoreload
%autoreload 2

res = def analyze_brillouin_spectrum_manual(
    brillouin_spectra,
    x_coord=None,
    y_coord=None,
    z_coord=None,
    free_spectral_range=29.98,
    laser_peak_ranges=[626, 986, 1269, 1510],
    brillouin_peak_ranges=(6, 11),
    spectrum_cut_range=(350, 1600),
    make_plot=(12, 8),
    debug_plot=False,
    # Lorentzian half-window in GHz (0 -> disable fitting)
    fit_lorentzians=1.5,

    # Optional filter pipeline configuration
    filter_settings=[{'type': 'savgol', 'params': (15, 6)},
    {'type': 'wavelet', 'wavelet': 'db6', 'level': 5},
    {'type': 'fft', 'params': (20, 100)}],
    save_plot=False,

    # Optional constraints for Lorentzian fitting (GHz)
    fwhm_bounds_GHz=(0.8, 4.0),     # (min_fwhm, max_fwhm) or None
    center_slack_GHz=0.5,    # ± slack around nominal center; if None, window-bound
    spectrum_data=None,
    baseline_spectra=False,
    poisson_weighting=False,
    match_brilouin_parameters=True,
    ignore_brilouin_peaks=False,
    plot_PCA=False,
    pca_n_components=2,
    pca_params={
        'svd_solver': 'full',
        'whiten': True,
        'tol': 0.0,
        'iterated_power': 5,
        'random_state': None,
        'copy': True,
        },
    pca_peak_finding_params={
        'height': 0.02,  # Minimum height of peaks
        'distance': 100,  # Minimum horizontal distance between peaks
        'prominence': 0, # Minimum prominence of peaks
        'width': 2, # Minimum width of peaks
    }
)

```

## Batch mode: `analyze_brillouin_spectra_manual`

Runs the same analysis across a z-slice or the full volume.

**Highlights**
- `z_coord=None` processes all slices; set an int to restrict to one slice.
- Parallel execution via `max_workers` and `parallel_backend` (`auto`, `thread`, `process`). Use `max_workers=1` for reproducible sequential runs.
- `laser_refit=True` performs a laser-only pre-pass to lock Rayleigh peak positions before full fitting. `refit` keeps a global VIPA refit (ignored when `laser_refit=True`).
- Pass through the same peak windows, fitting, filtering, and matching options you validated on a single spectrum. `keep_waveforms=False` drops raw spectra from the results to save memory.

**Returns**

- A `BrillouinPeaksMap` (or list of maps if `z_coord=None`) keyed by `(x, y, z)` with the per-pixel result dictionaries. The lateral step is stored internally for plotting and cell selection.

Example:

```python
from brillouin_analyzer_src.brillouin_spectra_analyzer_manual import analyze_brillouin_spectra_manual

peaks_maps = analyze_brillouin_spectra_manual(
    data,
    z_coord=None,
    free_spectral_range=29.98,
    laser_peak_ranges=[626, 986, 1269, 1510],
    brillouin_peak_ranges=(6, 11),
    spectrum_cut_range=(350, 1600),
    fit_lorentzians=1.5,
    filter_settings=[{'type': 'savgol', 'params': (15, 6)},
    {'type': 'wavelet', 'wavelet': 'db6', 'level': 5},
    {'type': 'fft', 'params': (20, 100)}],
    refit=False,
    laser_refit=True,
    match_brilouin_parameters=True,
    max_workers=8,
    parallel_backend='auto'
    keep_waveforms=False,
    baseline_spectra=False,
    poisson_weighting=False,
    plot_PCA=False,
    pca_n_components=2,
    pca_params={
        'svd_solver': 'full',
        'whiten': True,
        'tol': 0.0,
        'iterated_power': 5,
        'random_state': None,
        'copy': True,
        },
    pca_peak_finding_params={
        'height': 0.02,  # Minimum height of peaks
        'distance': 100,  # Minimum horizontal distance between peaks
        'prominence': 0, # Minimum prominence of peaks
        'width': 2, # Minimum width of peaks
    }
)
```
