# Plotting Module

The `brillouin_plotter` utilities turn parsed/analysed data into figures for validation and publication. They accept the outputs of `parse_brillouin_set`, `analyze_brillouin_spectrum_manual`, and `analyze_brillouin_spectra_manual`, plus the cell masks produced by `detect_cells`.

## Spectra preview: `plot_raw_spectrum`

Plots a single raw spectrum and marks peaks found by `scipy.signal.find_peaks`.
- `spectrum_range` trims the waveform before plotting; negative values are clamped to zero automatically.
- `height`, `distance`, and `prominence` are passed to `find_peaks`.
- `mark_ranges` shades useful index ranges (for example, candidate laser windows).

```python
from brillouin_analyzer_src.brillouin_plotter import plot_raw_spectrum
plot_raw_spectrum(spectra, x_coord=3, y_coord=4, z_coord=0, mark_ranges=[(200, 260)])
```

## Spatial heatmaps: `plot_brillouin_heatmap`

Builds a heatmap from a peaks map (or a list of maps) returned by the manual analyzer.
- Pick the value to plot with `data_type` (`Shift` or `FWHM`) and `peak_type` (`Brillouin`, `Brillouin Left`, `Brillouin Right`, `Laser`, etc.). `match_type='contains'` broadens label matching.
- Aggregation: `aggregation` applies inside each pixel (median/mean/robust); `pixel_aggregation` combines multiple maps (mean/median).
- Cleaning: `interpolate_nan`, optional filters (`median`, `gaussian`, `bilateral`, `wiener`, `anisotropic_diffusion`, `total_variation`, `non_local_means`), and `colorbar_range` control visibility.
- Axes: set `scale='auto'` to reuse the lateral step stored in the peaks map or pass a numeric scale. `matrix_save_path` writes the plotted matrix to a tab-separated file.

```python
from brillouin_analyzer_src.brillouin_plotter import plot_brillouin_heatmap
plot_brillouin_heatmap(
    peaks_map=peaks_maps[0],
    title="Median Brillouin shift",
    data_type="Shift",
    peak_type="Brillouin",
    pixel_aggregation="median",
    colorbar_range="auto",
    scale="auto"
)
```

## Cell-level statistics

Use the outputs of `detect_cells` (cell shift/FWHM maps plus background) to summarize samples.

### Box/violin/error plots: `plot_cell_boxplot`
- Inputs: `samples_data_list` is a list where each element is the 4-tuple returned by `detect_cells` (shift maps, FWHM maps, background shift, background FWHM). Pass multiple samples to compare cohorts.
- Pick metric with `data_type` (`shift`/`fwhm`) and `shift_type` (`stokes`, `anti-stokes`, `all`).
- Presentation controls: `plot_type` (`box`, `violin`, `errorbar`, or `scatter`), `aggregate_per_cell` (`pixel`, `cell`, or `both`), `show_data_points`, and `plot_background`.
- Statistics: enable `plot_p_values` with `comparisons=[(0,1), ...]`, choose `test_type` (`mannwhitney`, `ttest`, or `both`), and format with `p_value_format` (`numeric` or `stars`).

```python
from brillouin_analyzer_src.brillouin_plotter import plot_cell_boxplot
plot_cell_boxplot(
    samples_data_list=[cells_treated, cells_control],
    labels=["Treated", "Control"],
    data_type="shift",
    shift_type="all",
    plot_type="box",
    plot_p_values=True,
    comparisons=[(0, 1)]
)
```

### Histograms: `plot_cell_histogram`

Creates a histogram of per-cell or background values, fits a Gaussian, and can export the underlying arrays.
- `data_list` accepts the same 4-tuples from `detect_cells`; set `cell_or_background` accordingly.
- Limit with `cell_numbers`, clip with `value_range`, and choose `data_type`/`shift_type`.
- Set `save_values=True` (optionally with `values_output_dir` and `values_filename_prefix`) to write both shift and FWHM arrays to disk.

```python
from brillouin_analyzer_src.brillouin_plotter import plot_cell_histogram
plot_cell_histogram(
    [cells_treated],
    title="Shift distribution",
    cell_or_background="cell",
    data_type="shift",
    shift_type="all",
    bin_number=20,
    save_values=True,
    values_output_dir="exports"
)
```
