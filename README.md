# Brillouin Analyzer

Tools for parsing, fitting, plotting, and segmenting Brillouin/Raman datasets. The project includes parsers for raw acquisitions, manual spectrum analysis with Lorentzian fitting, spatial heatmaps, and cell detection plus per-region statistics.

## Features

- Parse Brillouin and Raman datasets from folders or ZIP archives with automatic lateral step detection.
- Tune and run manual peak analysis on single spectra, then batch it across full 3D volumes.
- Visualize spectra, render shift/FWHM heatmaps, and export matrices for further processing.
- Detect cells/regions on RGB maps (Stokes shift, FWHM, Anti-Stokes shift) and export per-region medians and CSVs.
- Built-in docs with MkDocs + Material for quick publishing.

## Installation

Create a virtual environment and install the package plus runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
pip install numpy pandas scipy matplotlib scikit-learn scikit-image opencv-python tqdm pywt
```

For documentation and local preview:

```bash
pip install mkdocs mkdocs-material
```

## Quick Start

```python
from brillouin_analyzer_src.brillouin_parser import parse_brillouin_set
from brillouin_analyzer_src.brillouin_spectra_analyzer_manual import (
    analyze_brillouin_spectrum_manual,
    analyze_brillouin_spectra_manual,
)
from brillouin_analyzer_src.brillouin_plotter import plot_brillouin_heatmap
from brillouin_analyzer_src.brillouin_cell_selector import detect_cells

# 1) Load data
data = parse_brillouin_set("path/to/dataset", file_label="Sample")

# 2) Tune peaks on one spectrum
single = analyze_brillouin_spectrum_manual(
    brillouin_spectra=data,
    x_coord=5, y_coord=5, z_coord=0,
    laser_peak_ranges=[(210, 280), (590, 650)],
    brillouin_peak_ranges=(6, 11),
    fit_lorentzians=8.0,
    debug_plot=True,
)

# 3) Run over the full volume
peaks_maps = analyze_brillouin_spectra_manual(
    data,
    laser_peak_ranges=[(210, 280), (590, 650)],
    brillouin_peak_ranges=(6, 11),
    fit_lorentzians=8.0,
    laser_refit=True,
    match_brilouin_parameters=True,
)

# 4) Visualize and segment
plot_brillouin_heatmap(peaks_map=peaks_maps[0], title="Median shift", data_type="Shift", peak_type="Brillouin")
cells = detect_cells(peaks_map=peaks_maps[0], title="Cells", shift_range="auto", fwhm_range="auto")
```

See the full API notes in the `docs/` directory.

## Documentation

- Live preview: `mkdocs serve`
- Build static site: `mkdocs build` (output in `site/`)
- Publish to GitHub Pages: `mkdocs gh-deploy` (requires repository remote configuration)

## Project Layout

- `brillouin_analyzer_src/brillouin_parser.py`: Data loading/parsing for Brillouin and Raman sets.
- `brillouin_analyzer_src/brillouin_spectra_analyzer_manual.py`: Single-spectrum and batch analysis with Lorentzian fitting.
- `brillouin_analyzer_src/brillouin_plotter.py`: Raw spectrum previews, heatmaps, box/violin plots, and histograms.
- `brillouin_analyzer_src/brillouin_cell_selector.py`: Cell/background detection and CSV export.
- `docs/`: MkDocs content.

## License

Licensed under the MIT License (see `LICENSE`).
