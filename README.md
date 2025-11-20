# Brillouin Analyzer


Tools for parsing, fitting, plotting, and segmenting Brillouin datasets.

Affiliation: Advanced Spectroscopy Laboratory, Texas A&M University

## Installation

To run it you need to have anaconda + OpenCV installed.

## Typical workflow

Copy folder `brillouin_analyzer_src` to your project.


```python
# 1) Load data
data = parse_brillouin_set("path/to/dataset", file_label="Sample")

# 2) Tune peaks on one spectrum
single = analyze_brillouin_spectrum_manual(
    ...
)

# 3) Run over the full volume
peaks_maps = analyze_brillouin_spectra_manual(
    ...
)

# 4) Visualize and segment
plot_brillouin_heatmap(peaks_maps[i], ...)
cells = detect_cells(peaks_map=peaks_maps[0], ...)
```

See example in `example.ipynb`.

Documentation is WIP. You can find it [here](https://mkizilov.github.io/brillouin_analyzer/).


## Project Layout

- `brillouin_analyzer_src/brillouin_parser.py`: Data loading/parsing for Brillouin and Raman sets.
- `brillouin_analyzer_src/brillouin_spectra_analyzer_manual.py`: Single-spectrum and batch analysis with Lorentzian fitting.
- `brillouin_analyzer_src/brillouin_plotter.py`: Raw spectrum previews, heatmaps, box/violin plots, and histograms.
- `brillouin_analyzer_src/brillouin_cell_selector.py`: Cell/background detection and CSV export.
- `docs/`: MkDocs content.

## License

Licensed under the MIT License (see `LICENSE`).
