import numpy as np  # For numerical operations and array handling
import pandas as pd  # For DataFrame operations
import matplotlib.pyplot as plt  # For plotting (heatmaps, box plots, histograms)
from scipy.interpolate import griddata  # For interpolating NaN values
from scipy.ndimage import median_filter, gaussian_filter  # For applying filters
from scipy.signal import wiener  # For Wiener filtering
from scipy.stats import norm, mannwhitneyu, ttest_ind  # For statistical tests and Gaussian fitting
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_tv_bregman, denoise_nl_means  # For denoising filters
from scipy.signal import find_peaks  # For peak detection
from .aux_func import convert_to_df_map, extract_peak_values
from .data_registry import extract_lateral_step_from_container
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib import patheffects

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
})


def plot_raw_spectrum(spectra_array, x_coord, y_coord, z_coord, spectrum_range=None, height=None, distance=None, prominence=None, mark_ranges=None, plot_size=(12, 8)):
    spectra_array = spectra_array.copy()
    spectrum = spectra_array[x_coord, y_coord, z_coord]
    
    # Replace negative values with zeros
    spectrum[spectrum < 0] = 0
    # Cut off spectrum if specified
    if spectrum_range:
        spectrum = spectrum[spectrum_range[0]:spectrum_range[1]]
    # Remove baseline
    spectrum = spectrum - np.min(spectrum)

    # Find peaks
    peaks, properties = find_peaks(spectrum, height=height, distance=distance, prominence=prominence)
    
    plt.figure(figsize=plot_size)
    plt.plot(spectrum, label='Spectrum', color='blue')
    # Plot peaks
    plt.plot(
        peaks,
        spectrum[peaks],
        "x",
        label="Peaks",
        color='green',
        markersize=10,
        markeredgewidth=4
    )
    max_value = np.max(spectrum)
    shift = 0.05 * max_value
    for peak in peaks:
        # Annotate peak prominence
        plt.text(peak, spectrum[peak], f"pr:{properties['prominences'][list(peaks).index(peak)]:.2f}", fontsize=8)
        # Annotate peak height
        plt.text(peak, spectrum[peak]+shift, f"ph:{properties['peak_heights'][list(peaks).index(peak)]:.2f}", fontsize=8)
        # Annotate peak X position
        plt.text(peak, spectrum[peak]+2*shift, f"pp:{peak}", fontsize=8)
    
    if mark_ranges:
        for range_ in mark_ranges:
            plt.axvspan(range_[0], range_[1], color='red', alpha=0.2)
    # Increase x ticks amount
    plt.xticks(np.arange(0, len(spectrum), 50))
    # Tilt x labels
    plt.xticks(rotation=45)
    plt.title(f"Raw Brillouin Spectrum at (x={x_coord}, y={y_coord}, z={z_coord})")
    plt.xlabel("GHz")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_brillouin_heatmap(
    peaks_map,
    title,
    data_type="Shift",
    peak_type='Brillouin',
    aggregation='median',
    filter_type=None,
    filter_params=None,
    interpolate_nan=True,
    colorbar_range=("auto", 60, 99),
    cmap='jet',
    interpolation=None,
    annotate=True,
    scalebar=False,
    match_type='exact',
    scale=1,
    pixel_aggregation='mean',
    save_path=False,
    matrix_save_path=None       # <-- new
):
    """
    Plots a 2D heatmap based on values extracted from a dictionary of DataFrames.
    If peaks_map is a list of peaks_maps, computes per-pixel aggregation across maps.

    Parameters:
    -----------
    peaks_map : dict or list of dicts
        Dictionary containing analysis results for each pixel, or list of such dictionaries.
    title : str
        Title for the plot.
    data_type : str, optional
        The type of data to plot ('Shift' or 'FWHM'). Default is 'Shift'.
    peak_type : str, optional
        The type of peak to extract (e.g., 'Brillouin Left', 'Brillouin Right', 'Brillouin'). Default is 'Brillouin'.
    aggregation : str, optional
        The aggregation method to use when extracting values within a pixel ('median', 'mean', or 'robust'). Default is 'median'.
    pixel_aggregation : str, optional
        The aggregation method to use across pixels when multiple peaks_maps are provided ('mean' or 'median'). Default is 'mean'.
    filter_type : str, optional
        Type of filter to apply to the data ('median', 'gaussian', etc.).
    filter_params : dict, optional
        Parameters for the filter function.
    interpolate_nan : bool, optional
        Whether to interpolate over NaN values.
    colorbar_range : None, tuple, str or tuple
        If None: matplotlib chooses.  
        If (vmin, vmax): explicit range.  
        If 'auto': use 2nd and 98th percentiles of the data.  
        If ('auto', p_low, p_high): use the p_low and p_high percentiles.
    cmap : str, optional
        Colormap to use. Default is 'jet'.
    interpolation : str, optional
        Interpolation method for imshow.
    annotate : bool, optional
        Whether to annotate the plot with statistical information.
    scalebar : bool, optional
        If True, hide axis unit labels and draw a high-contrast scale bar.
    scale : float, optional
        Scaling factor for the axes.
    match_type : str, optional
        How to match the peak_type ('exact' or 'contains').

    Returns:
    --------
    None
    """
    if isinstance(scale, str) and scale.lower() == 'auto':
        lateral_step = extract_lateral_step_from_container(peaks_map)
        if lateral_step is None:
            print('No lateral step found; using scale=1.')
            scale = 1.0
        else:
            scale = float(lateral_step)

    
        # First, check if peaks_map is a list
    if isinstance(peaks_map, list):
        peaks_maps_list = peaks_map
    else:
        peaks_maps_list = [peaks_map]
    
    # Collect data_maps from each peaks_map
    data_maps_list = []
    for single_peaks_map in peaks_maps_list:
        # Check if single_peaks_map is actually a dictionary (proper peaks_map)
        # or if it's another list (from analyze_brillouin_spectra_manual with z_coord=None)
        if isinstance(single_peaks_map, list):
            # This means we have a list of z-slices, take the first one
            if len(single_peaks_map) > 0:
                actual_peaks_map = single_peaks_map[0]  # Use first z-slice
            else:
                continue  # Skip empty lists
        else:
            actual_peaks_map = single_peaks_map
        
        # Now actual_peaks_map should be a dictionary with (x,y) keys
        if not isinstance(actual_peaks_map, dict):
            print(f"Warning: Expected dict but got {type(actual_peaks_map)}")
            continue
            
        # Convert peaks_map to df_map
        df_map = convert_to_df_map(actual_peaks_map)
    
        # Extract data using extract_peak_values
        data_map = extract_peak_values(
            df_map,
            peak_type=peak_type,
            data_type=data_type,
            aggregation=aggregation,
            match_type=match_type
        )
        data_maps_list.append(data_map)
    
    # Now, stack data_maps along a new axis
    data_stack = np.stack(data_maps_list, axis=0)  # Shape: (num_maps, x_dim, y_dim)
    
    # Compute per-pixel aggregation across maps
    if pixel_aggregation == 'mean':
        data_map_aggregated = np.nanmean(data_stack, axis=0)
    elif pixel_aggregation == 'median':
        data_map_aggregated = np.nanmedian(data_stack, axis=0)
    else:
        raise ValueError(f"Unknown pixel_aggregation method: {pixel_aggregation}")
    
    # optionally save the raw matrix before any interpolation / filtering
    if matrix_save_path:
        # convert to DataFrame and save as tab-delimited
        import pandas as _pd
        df_mat = _pd.DataFrame(data_map_aggregated)
        df_mat.to_csv(matrix_save_path, sep='\t', index=False, header=False)
    
    processed_data = np.copy(data_map_aggregated)
    data_for_stat = processed_data
    # Proceed with the rest of the function
    # Interpolate NaN values if required
    if interpolate_nan:
        mask = ~np.isnan(processed_data)
        if np.sum(mask) == 0:
            raise ValueError("All data points are NaN. Cannot perform interpolation.")
        x, y = np.indices(processed_data.shape)
        x_valid = x[mask]
        y_valid = y[mask]
        data_valid = processed_data[mask]
        x_nan = x[~mask]
        y_nan = y[~mask]
        if len(x_nan) > 0:
            interpolated_values = griddata(
                points=(x_valid, y_valid),
                values=data_valid,
                xi=(x_nan, y_nan),
                method='linear'
            )
            nan_after_interp = np.isnan(interpolated_values)
            if np.any(nan_after_interp):
                interpolated_values[nan_after_interp] = griddata(
                    points=(x_valid, y_valid),
                    values=data_valid,
                    xi=(x_nan[nan_after_interp], y_nan[nan_after_interp]),
                    method='nearest'
                )
            processed_data[~mask] = interpolated_values

    # Apply the specified filter
    if filter_type == 'median':
        processed_data = median_filter(processed_data, **filter_params)
    elif filter_type == 'gaussian':
        processed_data = gaussian_filter(processed_data, **filter_params)
    elif filter_type == 'bilateral':
        processed_data = denoise_bilateral(processed_data, **filter_params)
    elif filter_type == 'wiener':
        processed_data = wiener(processed_data, **filter_params)
    elif filter_type == 'anisotropic_diffusion':
        processed_data = denoise_tv_chambolle(processed_data, **filter_params)
    elif filter_type == 'total_variation':
        processed_data = denoise_tv_bregman(processed_data, **filter_params)
    elif filter_type == 'non_local_means':
        processed_data = denoise_nl_means(processed_data, **filter_params)
    elif filter_type is not None:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # --- determine vmin/vmax from colorbar_range -------------------
    vmin, vmax = None, None
    if colorbar_range is not None:
        if colorbar_range == 'auto':
            p_low, p_high = 2, 98
            vmin, vmax = np.nanpercentile(processed_data, [p_low, p_high])
        elif isinstance(colorbar_range, tuple) and len(colorbar_range) == 3 and colorbar_range[0] == 'auto':
            _, p_low, p_high = colorbar_range
            vmin, vmax = np.nanpercentile(processed_data, [p_low, p_high])
        elif (isinstance(colorbar_range, (list, tuple)) and len(colorbar_range) == 2):
            vmin, vmax = colorbar_range
        else:
            raise ValueError(
                "Invalid colorbar_range. "
                "Use None, (vmin,vmax), 'auto', or ('auto', p_low, p_high)."
            )
    # ----------------------------------------------------------------

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        processed_data,
        cmap=cmap,
        origin='lower',
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation
    )
    
    plt.title(title, fontsize=20)
    xlabel = "" if scalebar else "μm"
    ylabel = "" if scalebar else "μm"
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    # Ticklabels format
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    # Set the ticks based on the scaling factors
    x_ticks = np.arange(0, processed_data.shape[1], step=10) * scale
    y_ticks = np.arange(0, processed_data.shape[0], step=10) * scale

    def _format_tick_label(value):
        rounded = int(np.round(value))
        if np.isclose(value, rounded):
            return str(rounded)
        formatted = f"{value:.2f}".rstrip('0').rstrip('.')
        return formatted

    def _nice_length(length_um):
        """Round a raw length to a visually pleasing number."""
        if not np.isfinite(length_um) or length_um <= 0:
            return 1.0
        exponent = np.floor(np.log10(length_um))
        fraction = length_um / (10 ** exponent)
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
        return nice_fraction * (10 ** exponent)

    x_tick_labels = [_format_tick_label(val) for val in x_ticks]
    y_tick_labels = [_format_tick_label(val) for val in y_ticks]

    plt.xticks(
        ticks=np.arange(0, processed_data.shape[1], step=10),
        labels=x_tick_labels if not scalebar else [],
        rotation=45
    )
    plt.yticks(
        ticks=np.arange(0, processed_data.shape[0], step=10),
        labels=y_tick_labels if not scalebar else []
    )
    #     # Force scientific notation on both axes
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((-1, 1))
    # plt.gca().xaxis.set_major_formatter(formatter)
    # plt.gca().yaxis.set_major_formatter(formatter)

    # Set tick label fontsize
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=20)
    if scalebar:
        ax.tick_params(axis='both', which='both', length=0)

    # Add colorbar with label
    cbar = plt.colorbar(im)
    cbar_label = f"Brillouin {data_type} [GHz]"
    # cbar_label = "GHz"
    cbar.set_label(cbar_label, fontsize=20)  # Updated fontsize

    # Set colorbar tick label fontsize
    cbar.ax.tick_params(labelsize=20)

    # Optionally annotate statistical information
    if annotate:
        mean_val = np.nanmean(data_for_stat)
        median_val = np.nanmedian(data_for_stat)
        std_val = np.nanstd(data_for_stat)
        ax.text(
            0.95,
            0.95,
            f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}",
            verticalalignment='top',
            horizontalalignment='right',
            transform=ax.transAxes,
            color='white',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5, pad=5)
        )

    if scalebar:
        px_size_um = scale if isinstance(scale, (int, float)) and np.isfinite(scale) and scale > 0 else 1.0
        data_height, data_width = processed_data.shape
        bar_pixel_length = max(1, int(data_width * 0.25))
        raw_length_um = bar_pixel_length * px_size_um
        nice_length_um = _nice_length(raw_length_um)
        bar_pixel_length = max(1, int(round(nice_length_um / px_size_um)))
        bar_height_pixels = max(1, int(data_height * 0.02))
        margin_x = data_width * 0.05
        margin_y = data_height * 0.05
        x0 = data_width - margin_x - bar_pixel_length
        y0 = margin_y
        text_offset = bar_height_pixels * 1.5
        cap_height = max(1, bar_height_pixels * 1.6)

        x_start = x0
        x_end = x0 + bar_pixel_length
        main_line = ax.plot(
            [x_start, x_end],
            [y0, y0],
            color='white',
            linewidth=6,
            solid_capstyle='butt',
            zorder=10
        )[0]
        main_line.set_path_effects([
            patheffects.Stroke(linewidth=9, foreground='black'),
            patheffects.Normal()
        ])



        label = f"{_format_tick_label(nice_length_um)} μm"
        text = ax.text(
            x0 + bar_pixel_length / 2,
            y0 + cap_height / 2 + text_offset,
            label,
            ha='center',
            va='bottom',
            color='white',
            fontsize=24,
            fontweight='bold',
            zorder=11
        )
        text.set_path_effects([
            patheffects.Stroke(linewidth=3, foreground='black'),
            patheffects.Normal()
        ])
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()
    
    


def plot_cell_boxplot(
    samples_data_list,
    labels=None,
    data_type='shift',
    shift_type='all',
    title='',
    plot_background=False,
    show_data_points=False,
    annotate_medians=False,
    aggregate_per_cell='pixel',  # Accept 'pixel', 'cell', or 'both'
    aggregate_func=np.nanmean,
    plot_type='box',  # Updated to accept 'box', 'violin', or 'errorbar'
    plot_size=(8, 6),
    plot_p_values=False,
    comparisons=None,
    p_value_source='cell',  # 'pixel' or 'cell'
    test_type='both',  # 'mannwhitney', 'ttest', or 'both'
    p_value_format='numeric'  # 'numeric' or 'stars'
):
    """
    Plots box plots, violin plots, or error bar plots of Brillouin shift or FWHM values for multiple samples and performs statistical tests.

    Parameters:
    -----------
    samples_data_list : list
        List of samples, where each sample is a list of datasets.
    labels : list, optional
        List of labels corresponding to each sample.
    data_type : str, optional
        Type of data to plot ('shift' or 'fwhm'). Default is 'shift'.
    shift_type : str, optional
        Type of shift to use ('stokes', 'anti-stokes', or 'all'). Default is 'all'.
    title : str, optional
        Title of the plot.
    plot_background : bool, optional
        Whether to include background data in the plot.
    show_data_points : bool, optional
        Whether to overlay individual data points on the plot.
    annotate_medians : bool, optional
        Whether to annotate medians on the plot.
    aggregate_per_cell : str, optional
        How to aggregate data ('pixel', 'cell', or 'both'). Default is 'pixel'.
    aggregate_func : function, optional
        Function to aggregate per-cell data. Default is np.nanmean.
    plot_type : str, optional
        Type of plot ('box', 'violin', or 'errorbar'). Default is 'box'.
    plot_size : tuple, optional
        Size of the plot.
    plot_p_values : bool, optional
        Whether to perform statistical tests and annotate p-values.
    comparisons : list of tuples, optional
        List of tuples specifying pairs of indices to compare.
    p_value_source : str, optional
        Source of data for p-values ('pixel' or 'cell'). Default is 'cell'.
    test_type : str, optional
        Statistical test to use ('mannwhitney', 'ttest', or 'both'). Default is 'both'.
    p_value_format : str, optional
        Format for p-values in annotations ('numeric' or 'stars'). Default is 'numeric'.
    """

    # Function to convert p-values to stars
    def p_value_to_stars(p):
        if p <= 1e-4:
            return '****'
        elif p <= 1e-3:
            return '***'
        elif p <= 1e-2:
            return '**'
        elif p <= 0.05:
            return '*'
        else:
            return 'ns'  # Not significant

    # Helper function to extract data
    def extract_data(data_list, cell_or_background='cell'):
        all_data_pixels = []
        all_data_cells = []
        dataset_indices_pixels = []  # Track dataset source for pixel data
        dataset_indices_cells = []   # Track dataset source for cell data

        # Ensure data_list is a list
        if not isinstance(data_list, list):
            data_list = [data_list]
        
        # Flatten the dataset if needed
        data_list = [item for sublist in data_list for item in sublist]
        
        # Loop over datasets
        for dataset_idx, dataset in enumerate(data_list):
            # Skip None values or empty datasets
            if dataset is None:
                continue
                
            try:
                if cell_or_background == 'cell':
                    if data_type.lower() == 'shift':
                        # Enhanced checks: verify dataset structure before attempting access
                        if not isinstance(dataset, (dict, tuple)) or len(dataset) == 0:
                            continue
                        # Check if first element is a dictionary containing the shift_type key
                        if not isinstance(dataset[0], dict):
                            print(f"Warning: Dataset[0] is not a dictionary: {type(dataset[0])}")
                            continue
                            
                        if shift_type not in dataset[0]:
                            print(f"Warning: Key '{shift_type}' not found in dataset[0]")
                            continue
                            
                        data_dict = dataset[0][shift_type]  # cell_shift_maps[shift_type]
                    elif data_type.lower() == 'fwhm':
                        # Enhanced checks for FWHM data
                        if not isinstance(dataset, (list, tuple)) or len(dataset) <= 1:
                            print(f"Warning: Dataset {dataset_idx} is not a valid list or has insufficient length for FWHM data")
                            continue
                        
                        if not isinstance(dataset[1], dict):
                            print(f"Warning: Dataset[1] is not a dictionary: {type(dataset[1])}")
                            continue
                            
                        if shift_type not in dataset[1]:
                            print(f"Warning: Key '{shift_type}' not found in dataset[1]")
                            continue
                            
                        data_dict = dataset[1][shift_type]  # cell_fwhm_maps[shift_type]
                    else:
                        raise ValueError("Invalid data_type: must be 'shift' or 'fwhm'.")
                    
                    # Verify data_dict is a list or array before processing
                    if not isinstance(data_dict, (list, tuple, np.ndarray)):
                        print(f"Warning: data_dict is not a list or array: {type(data_dict)}")
                        continue
                        
                    for arr in data_dict:
                        if arr is not None and isinstance(arr, np.ndarray):
                            # Get finite values from array
                            valid_values = arr[np.isfinite(arr)].flatten()
                            if len(valid_values) > 0:  # Only process if we have valid data
                                # Non-aggregated data (per-pixel)
                                all_data_pixels.extend(valid_values)
                                # Track dataset source for each pixel data point
                                dataset_indices_pixels.extend([dataset_idx] * len(valid_values))
                                
                                # Aggregated data (per-cell)
                                cell_value = aggregate_func(valid_values)
                                all_data_cells.append(cell_value)
                                # Track dataset source for each cell data point
                                dataset_indices_cells.append(dataset_idx)
                elif cell_or_background == 'background':
                    if data_type.lower() == 'shift':
                        # Enhanced checks for background shift data
                        if not isinstance(dataset, (list, tuple)) or len(dataset) <= 2:
                            print(f"Warning: Dataset {dataset_idx} is not a valid list or has insufficient length for background shift data")
                            continue
                        
                        if not isinstance(dataset[2], dict):
                            print(f"Warning: Dataset[2] is not a dictionary: {type(dataset[2])}")
                            continue
                            
                        if shift_type not in dataset[2]:
                            print(f"Warning: Key '{shift_type}' not found in dataset[2]")
                            continue
                            
                        arr = dataset[2][shift_type]  # background_shift_maps[shift_type]
                    elif data_type.lower() == 'fwhm':
                        # Enhanced checks for background FWHM data
                        if not isinstance(dataset, (list, tuple)) or len(dataset) <= 3:
                            print(f"Warning: Dataset {dataset_idx} is not a valid list or has insufficient length for background FWHM data")
                            continue
                        
                        if not isinstance(dataset[3], dict):
                            print(f"Warning: Dataset[3] is not a dictionary: {type(dataset[3])}")
                            continue
                            
                        if shift_type not in dataset[3]:
                            print(f"Warning: Key '{shift_type}' not found in dataset[3]")
                            continue
                            
                        arr = dataset[3][shift_type]  # background_fwhm_maps[shift_type]
                    else:
                        raise ValueError("Invalid data_type: must be 'shift' or 'fwhm'.")
                    
                    if arr is not None and isinstance(arr, np.ndarray):
                        # Get finite values from array
                        valid_values = arr[np.isfinite(arr)].flatten()
                        if len(valid_values) > 0:  # Only process if we have valid data
                            # Non-aggregated data (per-pixel)
                            all_data_pixels.extend(valid_values)
                            # Track dataset source for each data point
                            dataset_indices_pixels.extend([dataset_idx] * len(valid_values))
                            # Aggregated data (single value for background)
                            background_value = aggregate_func(valid_values)
                            all_data_cells.append(background_value)
                            dataset_indices_cells.append(dataset_idx)
                else:
                    raise ValueError("Invalid cell_or_background: must be 'cell' or 'background'.")
            except Exception as e:
                # More informative error message
                print(f"Warning: Skipping a dataset due to error: {e}")
                continue

        return np.array(all_data_pixels), np.array(all_data_cells), np.array(dataset_indices_pixels), np.array(dataset_indices_cells)

    def build_pretty_palette(num_items, cmap_name='Set2'):
        """Return a smooth pastel palette with enough colors for every sample."""
        if num_items <= 0:
            return []
        cmap = plt.get_cmap(cmap_name)
        if num_items == 1:
            return [cmap(0.5)]
        positions = np.linspace(0.15, 0.85, num_items)
        return [cmap(pos) for pos in positions]
    
    

    def clean_numeric(values):
        """Flatten values and drop NaN/inf so violin styling stays robust."""
        if values is None:
            return np.array([])
        try:
            arr = np.asarray(values, dtype=float).flatten()
        except (TypeError, ValueError):
            arr = np.array([float(v) for v in values if v is not None], dtype=float)
        if arr.size == 0:
            return arr
        finite_mask = np.isfinite(arr)
        return arr[finite_mask]
    # Default labels if not provided
    if labels is None:
        labels = [f'Sample {i+1}' for i in range(len(samples_data_list))]

    if len(labels) != len(samples_data_list):
        raise ValueError("Length of labels must match the number of samples in samples_data_list.")

    # Collect data for plotting and statistical testing
    plot_data = []
    stats_data = []
    dataset_indices_by_sample = []  # Store dataset indices for each sample

    for idx, sample_data_list in enumerate(samples_data_list):
        data_pixels, data_cells, dataset_indices_pixels, dataset_indices_cells = extract_data(sample_data_list, cell_or_background='cell')
        
        # Store dataset indices for coloring points later
        dataset_indices_by_sample.append(dataset_indices_pixels)

        # For plotting
        if aggregate_per_cell == 'pixel':
            plot_data.append({'data': data_pixels, 'label': labels[idx], 'type': 'pixel', 'dataset_indices': dataset_indices_pixels})
        elif aggregate_per_cell == 'cell':
            plot_data.append({'data': data_cells, 'label': labels[idx], 'type': 'cell', 'dataset_indices': dataset_indices_cells})
        elif aggregate_per_cell == 'both':
            plot_data.append({'data': data_pixels, 'label': f"{labels[idx]} (Pixel)", 'type': 'pixel', 'dataset_indices': dataset_indices_pixels})
            plot_data.append({'data': data_cells, 'label': f"{labels[idx]} (Cell)", 'type': 'cell', 'dataset_indices': dataset_indices_cells})
        else:
            raise ValueError("Invalid value for aggregate_per_cell. Choose 'pixel', 'cell', or 'both'.")

        # For statistical testing
        stats_data.append({'pixel': data_pixels, 'cell': data_cells, 'label': labels[idx]})

    # If plot_background is True, extract and combine background data
    if plot_background:
        combined_bg_data_pixels = []
        combined_bg_data_cells = []
        for sample_data_list in samples_data_list:
            bg_data_pixels, bg_data_cells, _, _ = extract_data(sample_data_list, cell_or_background='background')
            combined_bg_data_pixels.extend(bg_data_pixels)
            combined_bg_data_cells.extend(bg_data_cells)

        combined_bg_data_pixels = np.array(combined_bg_data_pixels)

        if aggregate_per_cell == 'pixel':
            plot_data.append({'data': combined_bg_data_pixels, 'label': 'Background', 'type': 'pixel'})
        elif aggregate_per_cell == 'cell':
            plot_data.append({'data': combined_bg_data_cells, 'label': 'Background', 'type': 'cell'})
        elif aggregate_per_cell == 'both':
            plot_data.append({'data': combined_bg_data_pixels, 'label': 'Background (Pixel)', 'type': 'pixel'})
            plot_data.append({'data': combined_bg_data_cells, 'label': 'Background (Cell)', 'type': 'cell'})

        # For statistical testing
        stats_data.append({'pixel': combined_bg_data_pixels, 'cell': combined_bg_data_cells, 'label': 'Background'})

    # Prepare data and labels for plotting
    data_to_plot = [entry['data'] for entry in plot_data]
    labels_to_plot = [entry['label'] for entry in plot_data]
    data_types = [entry['type'] for entry in plot_data]

    num_groups = len(data_to_plot)
    palette_colors = build_pretty_palette(num_groups)

    # Create the plot
    fig, ax = plt.subplots(figsize=plot_size)
    background_color = "#ffffff"
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)

    x_positions = np.arange(1, num_groups + 1)

    # Choose between box plot, violin plot, or errorbar plot
    if plot_type == 'box':
        bp = ax.boxplot(
            data_to_plot,
            labels=labels_to_plot,
            patch_artist=True,
            showfliers=False,
            zorder=2,
            whis=1.5
        )

        colors_cycle = palette_colors or [plt.get_cmap('Set2')(0.5)]
        if len(colors_cycle) < max(1, num_groups):
            repeats = int(np.ceil(max(1, num_groups) / len(colors_cycle)))
            colors_cycle = (colors_cycle * repeats)[:max(1, num_groups)]
        for patch, color in zip(bp['boxes'], colors_cycle):
            patch.set_facecolor(color)
            patch.set_edgecolor('#4a4a4a')
            patch.set_linewidth(1.2)

        for median in bp['medians']:
            median.set_color('#2f2f2f')
            median.set_linewidth(2)

        for artist_name in ('whiskers', 'caps'):
            for artist in bp[artist_name]:
                artist.set_color('#4a4a4a')
                artist.set_linewidth(1.1)

        # Store positions for statistical annotation
        whisker_max_positions = []
        for i in range(len(data_to_plot)):
            whisker = bp['whiskers'][i*2 + 1]  # Upper whisker for box i
            ydata = whisker.get_ydata()
            whisker_max = max(ydata)
            whisker_max_positions.append(whisker_max)

        whisker_min_positions = []
        for i in range(len(data_to_plot)):
            whisker = bp['whiskers'][i*2]  # Lower whisker for box i
            ydata = whisker.get_ydata()
            whisker_min = min(ydata)
            whisker_min_positions.append(whisker_min)

    elif plot_type == 'violin':
        cleaned_data = [clean_numeric(data) for data in data_to_plot]
        vp = ax.violinplot(
            cleaned_data,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.65,
            bw_method='scott'
        )

        colors_cycle = palette_colors or [plt.get_cmap('Set2')(0.5)]
        for body, color in zip(vp['bodies'], colors_cycle):
            body.set_facecolor(color)
            body.set_edgecolor('#4a4a4a')
            body.set_alpha(0.9)
            body.set_linewidth(1)

        whisker_max_positions = []
        whisker_min_positions = []
        for pos, data in zip(x_positions, cleaned_data):
            if data.size == 0:
                whisker_max_positions.append(np.nan)
                whisker_min_positions.append(np.nan)
                continue

            q1, median_val, q3 = np.percentile(data, [25, 50, 75])
            whisker_max_positions.append(np.max(data))
            whisker_min_positions.append(np.min(data))

            ax.vlines(pos, q1, q3, color='#2f2f2f', linewidth=2.4, zorder=4)
            ax.scatter(
                pos,
                median_val,
                s=60,
                color='white',
                edgecolor='#2f2f2f',
                linewidth=1.2,
                zorder=5
            )

        handles_existing, _ = ax.get_legend_handles_labels()
        if not handles_existing:
            legend_handles = [
                Line2D([0], [0], color='#2f2f2f', linewidth=2.4, label='IQR'),
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    markerfacecolor='white',
                    markeredgecolor='#2f2f2f',
                    markersize=6,
                    linestyle='',
                    label='Median'
                )
            ]
            ax.legend(handles=legend_handles, frameon=False, loc='upper right', fontsize=10)

    elif plot_type == 'errorbar':
        # Calculate means and standard errors for each dataset
        means = [np.nanmean(clean_numeric(data)) for data in data_to_plot]
        medians = [np.nanmedian(clean_numeric(data)) for data in data_to_plot]
        std_errors = [np.nanstd(clean_numeric(data)) for data in data_to_plot]

        if aggregate_func == np.nanmean:  # MK Fix
            plotdot = means
        else:
            plotdot = medians

        colors_cycle = palette_colors or [plt.get_cmap('Set2')(0.5)]
        if len(colors_cycle) < max(1, num_groups):
            repeats = int(np.ceil(max(1, num_groups) / len(colors_cycle)))
            colors_cycle = (colors_cycle * repeats)[:max(1, num_groups)]

        for i, (x, y, yerr) in enumerate(zip(x_positions, plotdot, std_errors)):
            color = colors_cycle[i % len(colors_cycle)]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt='o',
                capsize=5,
                capthick=1.8,
                elinewidth=1.8,
                markersize=8,
                color=color,
                markerfacecolor='white',
                markeredgecolor=color,
                label=labels_to_plot[i]
            )

        # Store positions for statistical annotation (mean + std_error for max, mean - std_error for min)
        whisker_max_positions = [val + err for val, err in zip(plotdot, std_errors)]
        whisker_min_positions = [val - err for val, err in zip(plotdot, std_errors)]

    elif plot_type == 'scatter':
        cleaned_arrays = [clean_numeric(data) for data in data_to_plot if clean_numeric(data).size > 0]
        if cleaned_arrays:
            all_data = np.concatenate(cleaned_arrays)
            y_min, y_max = np.nanmin(all_data), np.nanmax(all_data)
        else:
            y_min, y_max = 0, 1
        y_range = y_max - y_min
        if y_range == 0:
            y_range = max(abs(y_max) * 0.1, 1e-3)
        
        for i, entry in enumerate(plot_data):
            data = clean_numeric(entry['data'])
            if data.size == 0:
                continue
            
            width = 0.8
            x = np.random.uniform(i + 1 - width / 2, i + 1 + width / 2, data.size)
            color = palette_colors[i % len(palette_colors)] if palette_colors else '#4C72B0'
            
            ax.scatter(
                x,
                data,
                alpha=0.25,
                color=color,
                edgecolor='none',
                s=22,
                zorder=3
            )
        
        padding = y_range * 0.05
        ax.set_ylim(y_min - padding, y_max + padding)
        
        # Store positions for statistical annotation
        whisker_max_positions = []
        whisker_min_positions = []
        for data in data_to_plot:
            arr = clean_numeric(data)
            whisker_max_positions.append(np.nanmax(arr) if arr.size else np.nan)
            whisker_min_positions.append(np.nanmin(arr) if arr.size else np.nan)

    # Modified code for showing data points with different colors based on dataset source
# Modified code for showing data points with different colors based on dataset source
    if show_data_points and plot_type != 'errorbar':  # No need for data points in errorbar plot
        # Define a colormap for datasets
        
        # Track which sample labels have been included in the legend
        sample_labels_in_legend = set()
        
        for i, entry in enumerate(plot_data):
            data = entry['data']
            
            # Skip if no data or no dataset indices
            if len(data) == 0 or 'dataset_indices' not in entry:
                continue
                
            dataset_indices = entry['dataset_indices']
            
            # Ensure data is a numpy array for boolean indexing
            data = np.array(data) if not isinstance(data, np.ndarray) else data
            
            # Jitter x-values to avoid overlapping
            x = np.random.normal(i + 1, 0.04, size=len(data))
            
            # Get unique dataset indices
            unique_indices = np.unique(dataset_indices)
            
            # Add the main label to legend only once per sample
            if entry['label'] not in sample_labels_in_legend:
                # First dataset of this sample - add to legend
                first_idx = unique_indices[0]
                mask = dataset_indices == first_idx
                
                # Add main scatter point with the sample label
                ax.scatter(
                    x[mask], 
                    data[mask], 
                    alpha=0.7, 
                    color='grey',  # grey
                    s=20, 
                    zorder=3,
                    label=entry['label']  # Use the main sample label
                )
                
                # Mark this sample as added to legend
                sample_labels_in_legend.add(entry['label'])
                
                # Plot remaining points of this dataset without label
                if len(mask) > 0:
                    remaining_mask = mask[1:] if len(mask) > 1 else []
                    if len(remaining_mask) > 0:
                        ax.scatter(
                            x[1:][remaining_mask],
                            data[1:][remaining_mask],
                            alpha=0.7,
                            color='grey',  # grey
                            s=20,
                            zorder=3
                        )
                
                # Plot other datasets without labels but with different colors
                for idx in unique_indices[1:]:
                    mask = dataset_indices == idx
                    ax.scatter(
                        x[mask], 
                        data[mask], 
                        alpha=0.7, 
                        color='grey',  # grey
                        s=20, 
                        zorder=3
                    )
            else:
                # This sample already has a legend entry - plot all datasets without labels
                for idx in unique_indices:
                    mask = dataset_indices == idx
                    ax.scatter(
                        x[mask], 
                        data[mask], 
                        alpha=0.7, 
                        color='grey',  # grey
                        s=20, 
                        zorder=3
                    )
        
        # # Add legend if we have scatter points
        # if sample_labels_in_legend:
        #     # Place legend outside the plot to avoid obscuring data
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        


    # Optionally annotate medians
    if annotate_medians:
        medians = [np.nanmedian(clean_numeric(data)) for data in data_to_plot]
        for i, median in enumerate(medians):
            if not np.isfinite(median):
                continue
            x_pos = i + 1 - 0.1
            ax.text(
                x_pos,
                median,
                f'{median:.3f}',
                ha='center',
                va='center',
                fontsize=10,
                color='#1f1f1f',
                fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8),
                zorder=6
            )

    # Update tick labels to keep everything readable
    if num_groups > 0:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels_to_plot, rotation=33, ha='right')

    dtype_label = (data_type or 'shift').lower()
    ylabel = "GHz" if dtype_label == 'shift' else "GHz"

    # Set title and labels
    ax.set_title(title or '')
    ax.set_ylabel(ylabel)

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10, colors='#4a4a4a')

    # Show grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#cfd8dc', alpha=0.7, zorder=0)

    plt.tight_layout()

    # Perform statistical tests between specified pairs
    if plot_p_values and comparisons is not None:

        p_values = {}
        print("Statistical Test Results:")

        # Create a mapping from sample labels to stats data indices
        stats_label_to_index = {entry['label']: idx for idx, entry in enumerate(stats_data)}

        for comp in comparisons:
            i, j = comp

            if i >= len(plot_data) or j >= len(plot_data):
                print(f"Comparison indices {i}, {j} are out of range.")
                continue

            # Get plotting data labels
            label1 = plot_data[i]['label']
            label2 = plot_data[j]['label']

            # For statistical testing, we need to get the data specified by p_value_source
            # Extract base labels (without '(Pixel)' or '(Cell)')
            base_label1 = label1.replace(' (Pixel)', '').replace(' (Cell)', '')
            base_label2 = label2.replace(' (Pixel)', '')

            # Get indices in stats_data
            idx1 = stats_label_to_index.get(base_label1)
            idx2 = stats_label_to_index.get(base_label2)

            if idx1 is None or idx2 is None:
                print(f"Labels '{base_label1}' and/or '{base_label2}' not found in stats data.")
                continue

            data1 = stats_data[idx1][p_value_source]
            data2 = stats_data[idx2][p_value_source]

            # Check if data is not empty
            if len(data1) == 0 or len(data2) == 0:
                print(f"Cannot perform tests between {label1} and {label2} due to insufficient data.")
                continue

            # Perform statistical tests
            test_results = {}
            if test_type in ['mannwhitney', 'both']:
                u_stat, p_u = mannwhitneyu(data1, data2, alternative='two-sided')
                test_results['Mann-Whitney U'] = (u_stat, p_u)
            else:
                p_u = None
            if test_type in ['ttest', 'both']:
                t_stat, p_t = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
                test_results['t-test'] = (t_stat, p_t)
            else:
                p_t = None

            p_values[(i, j)] = test_results

            # Print test results
            print(f"Between '{label1}' and '{label2}':")
            for test_name, (stat, p_val) in test_results.items():
                print(f"  {test_name}: statistic = {stat:.4f}, p-value = {p_val:.8f}")

        # Now calculate the y_range based on whisker positions
        whisker_max_array = np.asarray(whisker_max_positions, dtype=float)
        whisker_min_array = np.asarray(whisker_min_positions, dtype=float)
        finite_max = whisker_max_array[np.isfinite(whisker_max_array)]
        finite_min = whisker_min_array[np.isfinite(whisker_min_array)]

        if finite_max.size == 0 or finite_min.size == 0:
            current_ylim = ax.get_ylim()
            y_max = current_ylim[1]
            y_min = current_ylim[0]
        else:
            y_max = finite_max.max()
            y_min = finite_min.min()

        y_range = y_max - y_min
        if y_range == 0:
            y_range = max(abs(y_max) * 0.1, 1e-3)

        line_offset = y_range * 0.08  # Space between lines
        text_offset = y_range * 0.03  # Space between line and text

        heights = list(np.nan_to_num(whisker_max_positions, nan=y_max))

        for idx, comp in enumerate(comparisons):
            i, j = comp

            if (i, j) not in p_values:
                continue  # Skip if no p-value calculated

            x1, x2 = i + 1, j + 1  # Boxplot positions

            # Adjust y position based on current heights
            y = max(heights[i], heights[j]) + line_offset
            h = line_offset * 1

            test_results = p_values.get((i, j), None)
            if test_results is None:
                continue

            # Get p-values
            p_u = test_results.get('Mann-Whitney U', (None, None))[1]
            p_t = test_results.get('t-test', (None, None))[1]

            # Determine significance level and color
            min_p_value = None
            if p_u is not None and p_t is not None:
                min_p_value = min(p_u, p_t)
            elif p_u is not None:
                min_p_value = p_u
            elif p_t is not None:
                min_p_value = p_t

            if min_p_value is not None:
                color = '#c0392b' if min_p_value < 0.05 else '#3a3a3a'
            else:
                color = '#3a3a3a'

            # Format p-value text
            if p_value_format == 'numeric':
                p_text = []
                if p_u is not None:
                    p_text.append(f"$p_U$ = {p_u:.3f}")
                if p_t is not None:
                    p_text.append(f"$p_t$ = {p_t:.3f}")
            elif p_value_format == 'stars':
                p_text = []
                if p_u is not None:
                    stars = p_value_to_stars(p_u)
                    p_text.append(f"{stars}")
                if p_t is not None:
                    stars = p_value_to_stars(p_t)
                    p_text.append(f"{stars}")
            else:
                raise ValueError("Invalid p_value_format. Choose 'numeric' or 'stars'.")

            # Draw the stat bar
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=color)
            # Add the p-value text
            ax.text((x1 + x2) * 0.5, y + h + text_offset, "\n".join(p_text),
                    ha='center', va='bottom', color=color, fontsize=10, fontweight='semibold')

            # Add vertical dotted lines from boxes to p-value annotation
            base_i = whisker_max_positions[i] if np.isfinite(whisker_max_positions[i]) else y
            base_j = whisker_max_positions[j] if np.isfinite(whisker_max_positions[j]) else y
            ax.plot([x1, x1], [base_i, y], linestyle='dotted', color='gray', zorder=1)
            ax.plot([x2, x2], [base_j, y], linestyle='dotted', color='gray', zorder=1)

            # Update the heights to prevent overlap
            heights[i] = y + h + text_offset + text_offset
            heights[j] = y + h + text_offset + text_offset

        # Adjust plot y-limits to make room for p-value annotations
        ylim_upper = max(heights) + line_offset * 2
        ax.set_ylim(y_min - line_offset, ylim_upper)

    plt.show()
    
def plot_cell_histogram(
    data_list, 
    title, 
    cell_or_background='cell', 
    cell_numbers=None, 
    data_type='shift', 
    shift_type='all', 
    value_range=None,
    color=None,
    bin_number=15,
    save_values=False,          # bool OR (legacy) path string
    values_output_dir=None,
    values_filename_prefix=None,
    flatten_single_wrappers=True,  # NEW: auto‐unwrap [[dataset]] -> dataset
    debug=False,
    **kwargs
):
    """
    Histogram of shift / FWHM values aggregated over cells or background.

    Backward compatibility:
      - Old param name 'range' still works.
      - Passing save_values=<path string> now means: save both value files to that directory.
      - Accepts accidentally double / triple wrapped lists like [[hm_cells[0]]] if flatten_single_wrappers=True.

    Saves TWO files when save_values=True (always both arrays):
        <prefix>_shift_values.txt
        <prefix>_fwhm_values.txt
    """
    # ---- legacy 'range' kw ----
    if value_range is None and 'range' in kwargs:
        value_range = kwargs['range']

    import numpy as _np
    from scipy.stats import norm
    import os, inspect

    # ---- normalize input list ----
    if not isinstance(data_list, list):
        data_list = [data_list]

    # Flatten one level if user passed something like [[hm_cells[0]], [hm_cells[1]], ...]
    # Each element should ultimately be a 4‑tuple/list: (cell_shift_maps, cell_fwhm_maps, background_shift_maps, background_fwhm_maps)
    def _unwrap(ds):
        # Repeatedly unwrap single-element lists that just wrap the real 4‑tuple
        safety = 0
        while (
            flatten_single_wrappers and
            isinstance(ds, list) and
            len(ds) == 1 and
            isinstance(ds[0], (list, tuple)) and
            len(ds[0]) == 4 and
            isinstance(ds[0][0], dict)
        ):
            ds = ds[0]
            safety += 1
            if safety > 5:
                break
        return ds

    data_list = [_unwrap(d) for d in data_list]

    # ---- backward compatibility for save_values path string ----
    if isinstance(save_values, str):
        # Treat provided string as output directory
        values_output_dir = save_values
        save_values = True

    if color is None:
        color = 'b' if data_type.lower() == 'shift' else ('g' if data_type.lower() == 'fwhm' else 'b')

    collected_shift = []
    collected_fwhm  = []

    for idx, dataset in enumerate(data_list):
        if dataset is None:
            if debug: print(f"[hist] Dataset {idx}: None -> skipped")
            continue

        # If still wrapped like [tuple] (e.g. user passed triple brackets) attempt one more unwrap
        if isinstance(dataset, list) and len(dataset) == 1 and isinstance(dataset[0], (list, tuple)) and len(dataset[0]) == 4:
            dataset = dataset[0]

        # Validate structure
        if not isinstance(dataset, (list, tuple)) or len(dataset) < 4:
            if debug:
                print(f"[hist] Dataset {idx}: unexpected structure (type={type(dataset)}, len={len(dataset) if isinstance(dataset,(list,tuple)) else 'n/a'}) -> skipped")
            continue

        try:
            if cell_or_background == 'cell':
                cell_shift_maps = dataset[0]
                cell_fwhm_maps  = dataset[1]
                if (not isinstance(cell_shift_maps, dict)) or (not isinstance(cell_fwhm_maps, dict)):
                    if debug: print(f"[hist] Dataset {idx}: first two elements not dicts -> skipped")
                    continue
                if shift_type not in cell_shift_maps or shift_type not in cell_fwhm_maps:
                    if debug: print(f"[hist] Dataset {idx}: shift_type '{shift_type}' missing -> skipped")
                    continue
                shift_cells_list = cell_shift_maps[shift_type] or []
                fwhm_cells_list  = cell_fwhm_maps[shift_type] or []
                indices = range(len(shift_cells_list)) if cell_numbers is None else cell_numbers
                for ci in indices:
                    if ci < 0 or ci >= len(shift_cells_list):
                        continue
                    s_arr = shift_cells_list[ci]
                    f_arr = fwhm_cells_list[ci] if ci < len(fwhm_cells_list) else None
                    if s_arr is not None:
                        collected_shift.extend(_np.asarray(s_arr).ravel())
                    if f_arr is not None:
                        collected_fwhm.extend(_np.asarray(f_arr).ravel())

            elif cell_or_background == 'background':
                bg_shift_maps = dataset[2]
                bg_fwhm_maps  = dataset[3]
                if isinstance(bg_shift_maps, dict) and shift_type in bg_shift_maps:
                    s_arr = bg_shift_maps[shift_type]
                    if s_arr is not None:
                        collected_shift.extend(_np.asarray(s_arr).ravel())
                if isinstance(bg_fwhm_maps, dict) and shift_type in bg_fwhm_maps:
                    f_arr = bg_fwhm_maps[shift_type]
                    if f_arr is not None:
                        collected_fwhm.extend(_np.asarray(f_arr).ravel())
            else:
                raise ValueError("cell_or_background must be 'cell' or 'background'.")
        except Exception as e:
            print(f"Warning: skipping one dataset ({e})")
            continue

    shift_values = _np.asarray(collected_shift)
    fwhm_values  = _np.asarray(collected_fwhm)
    shift_values = shift_values[_np.isfinite(shift_values)]
    fwhm_values  = fwhm_values[_np.isfinite(fwhm_values)]

    # Decide which array to plot
    if data_type.lower() == 'shift':
        plot_array = shift_values
    elif data_type.lower() == 'fwhm':
        plot_array = fwhm_values
    else:
        raise ValueError("data_type must be 'shift' or 'fwhm'.")

    if debug:
        print(f"[hist] shift_values count={shift_values.size}, fwhm_values count={fwhm_values.size}, plotting '{data_type}'")

    if plot_array.size == 0:
        print("No data to plot for histogram.")
        if save_values:
            if values_output_dir is None:
                values_output_dir = os.getcwd()
            os.makedirs(values_output_dir, exist_ok=True)
            prefix = values_filename_prefix or title
            _np.savetxt(os.path.join(values_output_dir, f"{prefix}_shift_values.txt"), shift_values, fmt="%.6f")
            _np.savetxt(os.path.join(values_output_dir, f"{prefix}_fwhm_values.txt"), fwhm_values, fmt="%.6f")
        return

    import matplotlib.pyplot as plt
    counts, bins, _ = plt.hist(plot_array, bins=bin_number, alpha=0.7, density=False, range=value_range, color=color)
    mu, std = norm.fit(plot_array)
    bin_w = bins[1] - bins[0]
    xfit = _np.linspace(bins[0], bins[-1], 250)
    yfit = norm.pdf(xfit, mu, std) * counts.sum() * bin_w
    plt.plot(xfit, yfit, 'k', lw=2)

    txt = f"μ={mu:.2f}\nσ={std:.2f}\nmedian={_np.median(plot_array):.2f}\nn={plot_array.size}"
    plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va='top')

    plt.title(title)
    plt.xlabel("[GHz]")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.show()

    if save_values:
        if values_output_dir is None:
            values_output_dir = os.getcwd()
        os.makedirs(values_output_dir, exist_ok=True)
        prefix = values_filename_prefix or title
        sh_path = os.path.join(values_output_dir, f"{prefix}_shift_values.txt")
        fw_path = os.path.join(values_output_dir, f"{prefix}_fwhm_values.txt")
        _np.savetxt(sh_path, shift_values, fmt="%.6f")
        _np.savetxt(fw_path, fwhm_values, fmt="%.6f")
        print(f"Saved: {sh_path}")
        print(f"Saved: {fw_path}")
