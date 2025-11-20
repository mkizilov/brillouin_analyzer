# Brillouin Analyzer Documentation

This project provides a lightweight toolkit for parsing, fitting, plotting, and segmenting Brillouin/Raman datasets. Use the docs below as a quick reference for the main modules.

## Outline

- **[Parsing](parsing.md)**: Load Brillouin and Raman datasets, including ZIP handling and lateral step detection.
- **[Manual Analysis](manual_analysis.md)**: Fit peaks in a single spectrum or an entire volume with reproducible window and filtering controls.
- **[Plotting](plotting.md)**: Preview raw spectra, render spatial heatmaps, and summarize cell-level statistics.
- **[Cell Selection](cell_selection.md)**: Build RGB maps of shifts/FWHM, detect contours, and export per-region metrics.

## Quick Start

1) Parse data with `parse_brillouin_set` (or `parse_raman_set` for Raman).  
2) Tune peak windows on one pixel using `analyze_brillouin_spectrum_manual`.  
3) Run `analyze_brillouin_spectra_manual` over the full slice/volume.  
4) Visualize with `plot_brillouin_heatmap` and segment regions with `detect_cells`.  
5) Summarize groups using `plot_cell_boxplot` or `plot_cell_histogram`.

## Project Structure

- `brillouin_parser.py`: Parsing helpers for Brillouin/Raman datasets.
- `brillouin_spectra_analyzer_manual.py`: Peak finding, Lorentzian fitting, and batch processing.
- `brillouin_plotter.py`: Spectrum previews, heatmaps, and statistical plots.
- `brillouin_cell_selector.py`: Cell/background detection and CSV export.
