# Cell Selection Module

The `brillouin_cell_selector.detect_cells` helper turns a peaks map into an RGB heatmap (Stokes shift = red, FWHM = green, Anti-Stokes shift = blue) and extracts per-cell masks. Use it after running the manual analyzer and before aggregating statistics with the plotting functions.

## What goes in
- `peaks_map`: a `BrillouinPeaksMap` or list of maps from `analyze_brillouin_spectra_manual`. Lists are aggregated per pixel with `pixel_aggregation` (`mean` or `median`).
- Spatial scaling: set `scaling_factor='auto'` to reuse the lateral step stored with the peaks map or pass a numeric micron-per-pixel value.

## Pre-processing controls
- Filtering: choose `filter_type` (`median`, `gaussian`, `bilateral`, `wiener`, `anisotropic_diffusion`, `total_variation`, `non_local_means`) with `filter_params`. Set `interpolate_nan=True` to fill gaps first.
- Normalization and ranges: `normalize` scales channels to [0, 1]; `shift_range`/`fwhm_range` clip values (use `'auto'` or `('auto', p_low, p_high)` for percentile clipping).
- Aggregation inside each peak: `aggregation` (`median`, `mean`, or `robust`) controls how per-pixel values are extracted.

## How cells are detected
- Thresholding: pick `threshold_method` (`otsu`, `adaptive`, `fixed`, or `manual`). For manual mode, pass `manual_regions` with polygons or circles. `threshold_value` is required when `fixed`.
- Post-processing: `pre_gaussian_blur` smooths the grayscale map before thresholding; `morph_op` applies open/close morphology; `contour_smoothing` simplifies contour polygons.
- Contour filters: `cell_area` limits pixel area, and `prominence` rejects dim regions. `mark_all=True` skips detection and treats the whole field as one cell.

## Outputs

```text
cell_shift_maps      # dict with keys 'stokes', 'anti-stokes', 'all' -> list of arrays (one per cell)
cell_fwhm_maps       # dict with keys 'stokes', 'anti-stokes', 'all' -> list of arrays (one per cell)
background_shift_maps  # same keys, arrays with cells masked out
background_fwhm_maps   # same keys, arrays with cells masked out
```

If `save_csv` is a path, a CSV is written for every detected region (plus background when available) with median shift and FWHM. Use `fig_path` and `save_fig=True` to store the plotted RGB map with contours and numbering; `annotate=True` adds per-region statistics to the figure.

## Example

```python
from brillouin_analyzer_src.brillouin_cell_selector import detect_cells

cell_shift_maps, cell_fwhm_maps, bg_shift, bg_fwhm = detect_cells(
    peaks_map=peaks_maps[0],
    title="Cell map",
    shift_range="auto",
    fwhm_range="auto",
    filter_type="gaussian",
    filter_params={"sigma": 1.2},
    threshold_method="otsu",
    cell_area=[50, 5000],
    annotate=True,
    save_fig=True,
    fig_path="exports/cell_map.png",
    save_csv="exports/cell_stats"
)
```
