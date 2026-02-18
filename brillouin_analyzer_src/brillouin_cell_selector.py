import csv  # For CSV export
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For plotting (RGB images, contours)
import cv2  # OpenCV for image processing and contour detection
from scipy.interpolate import griddata  # For interpolating NaN values
from scipy.ndimage import median_filter, gaussian_filter  # For applying filters
from scipy.signal import wiener  # For Wiener filtering
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_tv_bregman, denoise_nl_means  # For advanced denoising filters
from pathlib import Path  # For path manipulations
from .aux_func import convert_to_df_map, extract_peak_values
from .data_registry import extract_lateral_step_from_container

def detect_cells(  
    peaks_map,
    title = 'Plot',
    filter_type='median',
    filter_params={'size': 3},
    interpolate_nan=False,
    aggregation='median',
    normalize=True,
    pre_gaussian_blur=True,
    morph_op=True,
    annotate=True,
    save_fig=True,
    fig_path=None,
    save_csv=None,
    cell_area=[0, 1000],  # [min_area, max_area]
    threshold_method='otsu',  # 'otsu', 'adaptive', 'fixed', or 'manual'
    threshold_value=None,      # Threshold value if using 'fixed'
    manual_regions=None,       # List of regions: {'type':'polygon','points':[...] } or {'type':'circle','center':(x,y),'radius':r}
    contour_retrieval_mode=cv2.RETR_EXTERNAL,  # Contour retrieval mode
    contour_smoothing=0.001,  # Approximation factor for contour smoothing
    prominence=0.1,
    shift_range=('auto', 60, 99),
    fwhm_range=('auto', 60, 99),
    scaling_factor='auto',
    mark_all=False,
    pixel_aggregation='mean',   # New parameter for per-pixel aggregation
    plot_heatmap=False,  # Whether to plot heatmaps for shift and FWHM
    heatmap_save_path=None,  # Path to save heatmap figures
    heatmap_colorbar_range=("auto", 60, 99),  # Colorbar range for heatmaps
    heatmap_cmap='jet',  # Colormap for heatmaps
    heatmap_annotate=True  # Whether to annotate heatmaps
):
    """
    Plots an RGB heatmap where:
    - Red channel is Brillouin Left shift (Stokes)
    - Green channel is FWHM
    - Blue channel is Brillouin Right shift (Anti-Stokes)
    Detects cells using OpenCV, marks their contours on the plot, numbers them,
    and returns two dictionaries containing the shift and FWHM maps for each cell,
    along with background shift and FWHM maps.
    
    Parameters:
    -----------
    peaks_map : dict or list
        Dictionary or list of dictionaries containing analysis results for each pixel.
    title : str
        Title for the plot.
    filter_type : str, optional
        Type of filter to apply to the data ('median', 'gaussian', etc.).
    filter_params : dict, optional
        Parameters for the filter function.
    interpolate_nan : bool, optional
        Whether to interpolate over NaN values.
    aggregation : str, optional
        Aggregation method to use ('median', 'mean', 'robust').
    normalize : bool, optional
        Whether to normalize data to [0,1] range for plotting.
    pre_gaussian_blur : bool, optional
        Whether to apply Gaussian blur before thresholding.
    morph_op : bool, optional
        Whether to apply morphological operations after thresholding.
    annotate : bool, optional
        Whether to annotate the plot with statistical information.
    save_fig : bool, optional
        Whether to save the figure.
    fig_path : str, optional
        Path to save the figure if save_fig is True.
    save_csv : str or Path-like, optional
        Base path used to export per-region CSV files containing the median
        Brillouin shift and FWHM. When provided, a CSV is written for each
        detected region and the background (if available).
    cell_area : list of int, optional
        Minimum and maximum area (in pixels) for a contour to be considered a cell.
    threshold_method : str, optional
        Method used for thresholding ('otsu', 'adaptive', 'fixed', or 'manual').
    threshold_value : int or float, optional
        Threshold value if using 'fixed' thresholding.
    manual_regions : list of dict, optional
        Regions to use when threshold_method='manual'. Each dict is either
        {'type':'polygon','points':[ (x1,y1),... ] }
        or {'type':'circle','center':(x,y),'radius':r}.
    contour_retrieval_mode : int, optional
        Mode for contour retrieval (e.g., cv2.RETR_EXTERNAL).
    contour_smoothing : float, optional
        Approximation factor for contour smoothing.
    prominence : float, optional
        Minimum normalized mean intensity (0 to 1) for a contour to be considered a cell.
    shift_range : tuple of floats, optional
        Tuple specifying (shift_min, shift_max) for plotting.
    fwhm_range : tuple of floats, optional
        Tuple specifying (fwhm_min, fwhm_max) for plotting.
    scaling_factor : float, optional
        Scaling factor for the axes.
    mark_all : bool, optional
        If True, treats the entire image as a single cell.
    pixel_aggregation : str, optional
        Method for aggregating pixel values across multiple maps ('mean' or 'median').
    plot_heatmap : bool, optional
        If True, plots heatmaps of shift and FWHM with unselected areas using the lowest color from colorbar.
    heatmap_save_path : str, optional
        Base path for saving heatmap figures (one for shift, one for FWHM).
    heatmap_colorbar_range : tuple, optional
        Colorbar range for heatmaps. Default is ('auto', 60, 99).
    heatmap_cmap : str, optional
        Colormap for heatmaps. Default is 'jet'.
    heatmap_annotate : bool, optional
        Whether to annotate heatmaps with statistical information. Default is True.
    
    Returns:
    --------
    cell_shift_maps : dict
        A dictionary with keys 'stokes', 'anti-stokes', 'all', each containing a list of shift maps for each detected cell.
    cell_fwhm_maps : dict
        A dictionary with keys 'stokes', 'anti-stokes', 'all', each containing a list of FWHM maps for each detected cell.
    background_shift_maps : dict
        A dictionary with keys 'stokes', 'anti-stokes', 'all', each containing the background shift map.
    background_fwhm_maps : dict
        A dictionary with keys 'stokes', 'anti-stokes', 'all', each containing the background FWHM map.
    """
    if isinstance(scaling_factor, str) and scaling_factor.lower() == 'auto':
        lateral_step = extract_lateral_step_from_container(peaks_map)
        if lateral_step is None:
            print('No lateral step found; using scaling_factor=1.')
            scaling_factor = 1.0
        else:
            scaling_factor = float(lateral_step)

    # Handle multiple maps for per‐pixel aggregation
    if isinstance(peaks_map, list):
        peaks_maps_list = peaks_map
    else:
        peaks_maps_list = [peaks_map]

    # collect individual maps
    left_shifts = []
    right_shifts = []
    fwhm_lefts = []
    fwhm_rights = []

    for pmap in peaks_maps_list:
        # convert to df_map
        df_map = convert_to_df_map(pmap)

        # extract per‐map arrays
        ls = extract_peak_values(df_map, peak_type='Brillouin Left', data_type='Shift', aggregation=aggregation)
        rs = extract_peak_values(df_map, peak_type='Brillouin Right', data_type='Shift', aggregation=aggregation)
        fl = extract_peak_values(df_map, peak_type='Brillouin Left', data_type='FWHM', aggregation=aggregation)
        fr = extract_peak_values(df_map, peak_type='Brillouin Right', data_type='FWHM', aggregation=aggregation)

        left_shifts.append(ls)
        right_shifts.append(rs)
        fwhm_lefts.append(fl)
        fwhm_rights.append(fr)

    # aggregate across maps
    stack_ls = np.stack(left_shifts, axis=0)
    stack_rs = np.stack(right_shifts, axis=0)
    stack_fl = np.stack(fwhm_lefts, axis=0)
    stack_fr = np.stack(fwhm_rights, axis=0)

    if pixel_aggregation == 'mean':
        left_shift_map  = np.nanmean(stack_ls, axis=0)
        right_shift_map = np.nanmean(stack_rs, axis=0)
        fwhm_left_map   = np.nanmean(stack_fl, axis=0)
        fwhm_right_map  = np.nanmean(stack_fr, axis=0)
    elif pixel_aggregation == 'median':
        left_shift_map  = np.nanmedian(stack_ls, axis=0)
        right_shift_map = np.nanmedian(stack_rs, axis=0)
        fwhm_left_map   = np.nanmedian(stack_fl, axis=0)
        fwhm_right_map  = np.nanmedian(stack_fr, axis=0)
    else:
        raise ValueError(f"Unknown pixel_aggregation: {pixel_aggregation}")

    # Copy the original data for return values
    left_shift_original = np.copy(left_shift_map)
    right_shift_original = np.copy(right_shift_map)
    all_shift_original = np.nanmean([left_shift_original, right_shift_original], axis=0)

    fwhm_left_original = np.copy(fwhm_left_map)
    fwhm_right_original = np.copy(fwhm_right_map)
    fwhm_all_original = np.nanmean([fwhm_left_original, fwhm_right_original], axis=0)

    # Process each channel separately
    def process_channel(data_map):
        processed_data = np.copy(data_map)
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

        return processed_data

    # Process each channel
    left_shift_processed  = process_channel(left_shift_map)
    right_shift_processed = process_channel(right_shift_map)
    fwhm_left_processed   = process_channel(fwhm_left_map)
    fwhm_right_processed  = process_channel(fwhm_right_map)

    # Apply shift_range for plotting purposes only
    if shift_range is not None:
        # support 'auto', ('auto', p_low, p_high), or (min, max)
        if shift_range == 'auto':
            p_low, p_high = 2, 98
            left_vmin,  left_vmax  = np.nanpercentile(left_shift_processed,  [p_low, p_high])
            right_vmin, right_vmax = np.nanpercentile(right_shift_processed, [p_low, p_high])
        elif isinstance(shift_range, (list, tuple)) and len(shift_range) == 3 and shift_range[0] == 'auto':
            _, p_low, p_high = shift_range
            left_vmin,  left_vmax  = np.nanpercentile(left_shift_processed,  [p_low, p_high])
            right_vmin, right_vmax = np.nanpercentile(right_shift_processed, [p_low, p_high])
        elif isinstance(shift_range, (list, tuple)) and len(shift_range) == 2:
            left_vmin, left_vmax = shift_range
            right_vmin, right_vmax = shift_range
        else:
            raise ValueError("Invalid shift_range. Use None, (min,max), 'auto', or ('auto',p_low,p_high).")
        left_shift_processed  = np.clip(left_shift_processed,  left_vmin,  left_vmax)
        right_shift_processed = np.clip(right_shift_processed, right_vmin, right_vmax)

    # Apply fwhm_range for plotting purposes only
    if fwhm_range is not None:
        # support 'auto', ('auto', p_low, p_high), or (min, max)
        if fwhm_range == 'auto':
            p_low, p_high = 2, 98
            fl_vmin, fl_vmax = np.nanpercentile(fwhm_left_processed,  [p_low, p_high])
            fr_vmin, fr_vmax = np.nanpercentile(fwhm_right_processed, [p_low, p_high])
        elif isinstance(fwhm_range, (list, tuple)) and len(fwhm_range) == 3 and fwhm_range[0] == 'auto':
            _, p_low, p_high = fwhm_range
            fl_vmin, fl_vmax = np.nanpercentile(fwhm_left_processed,  [p_low, p_high])
            fr_vmin, fr_vmax = np.nanpercentile(fwhm_right_processed, [p_low, p_high])
        elif isinstance(fwhm_range, (list, tuple)) and len(fwhm_range) == 2:
            fl_vmin, fl_vmax = fwhm_range
            fr_vmin, fr_vmax = fwhm_range
        else:
            raise ValueError("Invalid fwhm_range. Use None, (min,max), 'auto', or ('auto',p_low,p_high).")
        fwhm_left_processed  = np.clip(fwhm_left_processed,  fl_vmin, fl_vmax)
        fwhm_right_processed = np.clip(fwhm_right_processed, fr_vmin, fr_vmax)

    # Compute the average of shifts and FWHM
    all_shift_processed = np.nanmean([left_shift_processed, right_shift_processed], axis=0)
    fwhm_processed_all = np.nanmean([fwhm_left_processed, fwhm_right_processed], axis=0)

    # Normalize each channel to [0,1] for visualization
    if normalize:
        def normalize_channel(channel_data):
            min_val = np.nanmin(channel_data)
            max_val = np.nanmax(channel_data)
            if max_val - min_val == 0:
                return np.zeros_like(channel_data)
            normalized = (channel_data - min_val) / (max_val - min_val)
            return np.clip(normalized, 0, 1) # Something is here

        left_shift_normalized = normalize_channel(left_shift_processed)
        right_shift_normalized = normalize_channel(right_shift_processed)
        all_shift_normalized = normalize_channel(all_shift_processed)
        fwhm_normalized = normalize_channel(fwhm_processed_all)
    else:
        left_shift_normalized = np.clip(left_shift_processed, 0, 1)
        right_shift_normalized = np.clip(right_shift_processed, 0, 1)
        all_shift_normalized = np.clip(all_shift_processed, 0, 1)
        fwhm_normalized = np.clip(fwhm_processed_all, 0, 1)

    # Combine channels into an RGB image for visualization
    rgb_image = np.stack([left_shift_normalized, fwhm_normalized, right_shift_normalized], axis=-1)

    # Convert the RGB image to uint8 format for OpenCV
    rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

    # Convert to grayscale for contour detection
    gray_image = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2GRAY)
    if pre_gaussian_blur:
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Initialize masks for background
    height, width = gray_image.shape
    cell_mask = np.zeros((height, width), dtype=bool)

    # Initialize dictionaries to hold cell maps
    cell_shift_maps = {'stokes': [], 'anti-stokes': [], 'all': []}
    cell_fwhm_maps = {'stokes': [], 'anti-stokes': [], 'all': []}
    csv_exports = []

    def _safe_nanmedian(array):
        finite_vals = array[np.isfinite(array)]
        if finite_vals.size == 0:
            return np.nan
        return float(np.nanmedian(finite_vals))

    background_shift_maps = {
        'stokes': None,
        'anti-stokes': None,
        'all': None
    }

    background_fwhm_maps = {
        'stokes': None,
        'anti-stokes': None,
        'all': None
    }

    if mark_all:
        # Treat the entire image as a single cell
        cell_mask[:, :] = True  # All pixels belong to the cell

        # Mask the data channels
        left_shift_cell = np.where(cell_mask, left_shift_original, np.nan)
        right_shift_cell = np.where(cell_mask, right_shift_original, np.nan)
        all_shift_cell = np.where(cell_mask, all_shift_original, np.nan)

        fwhm_left_cell = np.where(cell_mask, fwhm_left_original, np.nan)
        fwhm_right_cell = np.where(cell_mask, fwhm_right_original, np.nan)
        fwhm_all_cell = np.where(cell_mask, fwhm_all_original, np.nan)

        # Append to dictionaries
        cell_shift_maps['stokes'].append(left_shift_cell)
        cell_shift_maps['anti-stokes'].append(right_shift_cell)
        cell_shift_maps['all'].append(all_shift_cell)

        cell_fwhm_maps['stokes'].append(fwhm_left_cell)
        cell_fwhm_maps['anti-stokes'].append(fwhm_right_cell)
        cell_fwhm_maps['all'].append(fwhm_all_cell)

        region_label = 'region_1'
        shift_median = _safe_nanmedian(all_shift_cell)
        fwhm_median = _safe_nanmedian(fwhm_all_cell)
        csv_exports.append((region_label, shift_median, fwhm_median))

        # Commented because it's not useful for the current use case
        # # Prepare to plot
        # plt.figure(figsize=(8, 8))
        # plt.imshow(rgb_image, origin='lower', aspect='auto')
        # plt.title(title, fontsize=14)
        # plt.xlabel("X [μm]", fontsize=12)
        # plt.ylabel("Y [μm]", fontsize=12)

        # # Set the ticks based on the scaling factors
        # x_ticks = np.arange(0, rgb_image.shape[1], step=1) * scaling_factor
        # y_ticks = np.arange(0, rgb_image.shape[0], step=1) * scaling_factor
        # plt.xticks(ticks=np.arange(0, rgb_image.shape[1], step=10), labels=np.round(x_ticks[::10], 2), rotation=45)
        # plt.yticks(ticks=np.arange(0, rgb_image.shape[0], step=10), labels=np.round(y_ticks[::10], 2))

        # # Draw rectangle around the entire image to mimic a contour
        # rect = plt.Rectangle((0, 0), width, height, linewidth=2, edgecolor='white', facecolor='none', linestyle='dashed')
        # plt.gca().add_patch(rect)

        # # Plot cell number at the center
        # cX = width // 2
        # cY = height // 2
        # plt.text(
        #     cX,
        #     cY,
        #     '1',
        #     color='yellow',
        #     fontsize=12,
        #     ha='center',
        #     va='center',
        #     backgroundcolor='black',
        #     bbox=dict(facecolor='blue', alpha=0.5)
        # )

    else:
        # Apply thresholding to create binary image based on the specified method
        if threshold_method == 'otsu':
            _, binary_image = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif threshold_method == 'adaptive':
            binary_image = cv2.adaptiveThreshold(
                gray_image,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
        elif threshold_method == 'fixed':
            if threshold_value is None:
                raise ValueError("You must provide threshold_value when using 'fixed' threshold_method.")
            _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_method == 'manual':
            if manual_regions is None:
                raise ValueError("manual_regions must be provided when threshold_method='manual'")
            binary_image = np.zeros(gray_image.shape, dtype=np.uint8)
            for region in manual_regions:
                if region.get('type') == 'polygon':
                    pts = np.array(region['points'], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(binary_image, [pts], 255)
                elif region.get('type') == 'circle':
                    center = tuple(map(int, region['center']))
                    radius = int(region['radius'])
                    cv2.circle(binary_image, center, radius, 255, thickness=-1)
                else:
                    raise ValueError(f"Unknown manual region type: {region.get('type')}")
            contours = []
            for region in manual_regions:
                if region['type'] == 'polygon':
                    pts = np.array(region['points'], dtype=np.int32)
                else:  # circle
                    center = region['center']
                    r = region['radius']
                    angles = np.linspace(0, 2*np.pi, 100)
                    xs = center[0] + r*np.cos(angles)
                    ys = center[1] + r*np.sin(angles)
                    pts = np.vstack([xs, ys]).T.astype(np.int32)
                contours.append(pts)

        else:
            raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
        if morph_op:
            kernel = np.ones((3, 3), np.uint8)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours using the specified retrieval mode
        contours, hierarchy = cv2.findContours(
            binary_image, contour_retrieval_mode, cv2.CHAIN_APPROX_SIMPLE
        )

        # Prepare to plot
        plt.figure(figsize=(12, 12))
        plt.imshow(rgb_image, origin='lower', aspect='auto')
        plt.title(title, fontsize=14)
        plt.xlabel("X [μm]", fontsize=12)
        plt.ylabel("Y [μm]", fontsize=12)
        # Set the ticks based on the scaling factors
        x_ticks = np.arange(0, rgb_image.shape[1], step=1) * scaling_factor
        y_ticks = np.arange(0, rgb_image.shape[0], step=1) * scaling_factor
        plt.xticks(ticks=np.arange(0, rgb_image.shape[1], step=10), labels=np.round(x_ticks[::10], 2), rotation=45)
        plt.yticks(ticks=np.arange(0, rgb_image.shape[0], step=10), labels=np.round(y_ticks[::10], 2))

        cell_counter = 0  # To number cells sequentially
        # collect per‐region stats text
        stats_per_region = []

        for i, contour in enumerate(contours):
            # Filter out contours based on area
            area = cv2.contourArea(contour)
            if threshold_method != 'manual':
                if area < cell_area[0] or area > cell_area[1]:
                    continue  # Skip contours outside the specified area range

            # Get coordinates of contour
            contour = contour.squeeze()
            if len(contour.shape) != 2:
                continue  # Skip invalid contours

            # Apply contour smoothing if specified
            if contour_smoothing > 0:
                epsilon = contour_smoothing * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
                contour = contour.squeeze()

            # Create a mask for the current contour
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

            # Update the cell mask
            cell_mask[mask == 255] = True

            # Compute mean intensity inside the contour
            mean_intensity = cv2.mean(gray_image, mask=mask)[0]
            normalized_mean_intensity = mean_intensity / 255.0

            # If mean intensity is less than prominence, skip the contour
            if normalized_mean_intensity < prominence:
                continue

            # Mask the data channels
            left_shift_cell = np.where(mask == 255, left_shift_original, np.nan)
            right_shift_cell = np.where(mask == 255, right_shift_original, np.nan)
            all_shift_cell = np.nanmean([left_shift_cell, right_shift_cell], axis=0)

            fwhm_left_cell = np.where(mask == 255, fwhm_left_original, np.nan)
            fwhm_right_cell = np.where(mask == 255, fwhm_right_original, np.nan)
            fwhm_all_cell = np.nanmean([fwhm_left_cell, fwhm_right_cell], axis=0)

            region_label = f"region_{cell_counter + 1}"

            # Append to dictionaries
            cell_shift_maps['stokes'].append(left_shift_cell)
            cell_shift_maps['anti-stokes'].append(right_shift_cell)
            cell_shift_maps['all'].append(all_shift_cell)

            cell_fwhm_maps['stokes'].append(fwhm_left_cell)
            cell_fwhm_maps['anti-stokes'].append(fwhm_right_cell)
            cell_fwhm_maps['all'].append(fwhm_all_cell)

            shift_median = _safe_nanmedian(all_shift_cell)
            fwhm_median = _safe_nanmedian(fwhm_all_cell)
            csv_exports.append((region_label, shift_median, fwhm_median))

            # Close the contour loop for plotting
            contour = np.vstack([contour, contour[0]])

            # Draw contour on the plot
            plt.plot(
                contour[:, 0],
                contour[:, 1],
                color='white',
                linewidth=2,
                alpha=0.8,
                linestyle='dashed'
            )

            # Compute centroid for numbering
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = int(contour[:, 0].mean())
                cY = int(contour[:, 1].mean())

            # Plot cell number
            plt.text(
                cX, cY, str(cell_counter+1),
                color='yellow',
                fontsize=12,
                ha='center',
                va='center',
                backgroundcolor='black',
                bbox=dict(facecolor='blue', alpha=0.5)
            )

            # collect per‐cell statistics
            if annotate:
                arr_ls = left_shift_cell[np.isfinite(left_shift_cell)]
                arr_rs = right_shift_cell[np.isfinite(right_shift_cell)]
                arr_all = all_shift_cell[np.isfinite(all_shift_cell)]
                arr_fwhm = fwhm_all_cell[np.isfinite(fwhm_all_cell)]
                stats_per_region.append(
                    f"{cell_counter+1}: LS μ={np.nanmean(arr_ls):.2f}, med={np.nanmedian(arr_ls):.2f}, σ={np.nanstd(arr_ls):.2f}; "
                    f"RS μ={np.nanmean(arr_rs):.2f}, med={np.nanmedian(arr_rs):.2f}, σ={np.nanstd(arr_rs):.2f}; "
                    f"All μ={np.nanmean(arr_all):.2f}, med={np.nanmedian(arr_all):.2f}, σ={np.nanstd(arr_all):.2f}; "
                    f"FWHM μ={np.nanmean(arr_fwhm):.2f}, med={np.nanmedian(arr_fwhm):.2f}, σ={np.nanstd(arr_fwhm):.2f}"
                )

            cell_counter += 1

        # after looping all contours, draw single annotation box
        if annotate and stats_per_region:
            annot_text = "\n".join(stats_per_region)
            plt.text(
                0.01, 0.99,
                annot_text,
                transform=plt.gca().transAxes,
                fontsize=8,
                color='white',
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.6, pad=5)
            )

        # Create background maps by masking out cell regions
        background_shift_maps = {
            'stokes': np.where(cell_mask, np.nan, left_shift_original),
            'anti-stokes': np.where(cell_mask, np.nan, right_shift_original),
            'all': np.where(cell_mask, np.nan, all_shift_original)
        }

        background_fwhm_maps = {
            'stokes': np.where(cell_mask, np.nan, fwhm_left_original),
            'anti-stokes': np.where(cell_mask, np.nan, fwhm_right_original),
            'all': np.where(cell_mask, np.nan, fwhm_all_original)
        }

    background_entry = None
    if background_shift_maps['all'] is not None and background_fwhm_maps['all'] is not None:
        background_entry = (
            'background',
            _safe_nanmedian(background_shift_maps['all']),
            _safe_nanmedian(background_fwhm_maps['all'])
        )

    if save_csv:
        original_path_str = str(save_csv)
        base_path = Path(save_csv).expanduser()
        dir_hint = original_path_str.endswith(('/', '\\'))

        if base_path.suffix.lower() == '.csv':
            stem = base_path.stem
            directory = base_path.parent if base_path.parent != Path('') else Path.cwd()
        elif dir_hint:
            stem = 'cell'
            directory = base_path
        else:
            stem = base_path.name
            directory = base_path.parent if base_path.parent != Path('') else Path.cwd()

        if not stem or stem == '.':
            stem = 'cell'

        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Failed to create CSV output directory: {directory}") from exc

        def _format_value(value):
            return float(value) if np.isfinite(value) else ''

        def _write_file(label, shift_value, fwhm_value):
            output_path = directory / f"{stem}_{label}.csv"
            try:
                with output_path.open('w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['shift', 'fwhm'])
                    writer.writerow([
                        _format_value(shift_value),
                        _format_value(fwhm_value)
                    ])
            except OSError as exc:
                raise OSError(f"Failed to write CSV output: {output_path}") from exc

        for label, shift_value, fwhm_value in csv_exports:
            _write_file(label, shift_value, fwhm_value)

        if background_entry is not None:
            label, shift_value, fwhm_value = background_entry
            _write_file(label, shift_value, fwhm_value)

    # Optionally annotate statistical information
    if annotate:
        # Compute mean, median, std for each channel
        mean_left = np.nanmean(left_shift_original)
        median_left = np.nanmedian(left_shift_original)
        std_left = np.nanstd(left_shift_original)

        mean_right = np.nanmean(right_shift_original)
        median_right = np.nanmedian(right_shift_original)
        std_right = np.nanstd(right_shift_original)

        mean_all_shift = np.nanmean(all_shift_original)
        median_all_shift = np.nanmedian(all_shift_original)
        std_all_shift = np.nanstd(all_shift_original)

        mean_fwhm = np.nanmean(fwhm_all_original)
        median_fwhm = np.nanmedian(fwhm_all_original)
        std_fwhm = np.nanstd(fwhm_all_original)

        textstr = (
            f"Brillouin Left Shift (Stokes):\nMean: {mean_left:.2f}, Median: {median_left:.2f}, Std: {std_left:.2f}\n"
            f"Brillouin Right Shift (Anti-Stokes):\nMean: {mean_right:.2f}, Median: {median_right:.2f}, Std: {std_right:.2f}\n"
            f"Average Shift (All):\nMean: {mean_all_shift:.2f}, Median: {median_all_shift:.2f}, Std: {std_all_shift:.2f}\n"
            f"FWHM:\nMean: {mean_fwhm:.2f}, Median: {median_fwhm:.2f}, Std: {std_fwhm:.2f}"
        )

        plt.text(
            0.95,
            0.05,
            textstr,
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=plt.gca().transAxes,
            color='white',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5, pad=5)
        )

    plt.tight_layout()
    if save_fig and fig_path:
        plt.savefig(fig_path)
    plt.show()

    # Plot heatmaps if requested
    if plot_heatmap:
        from matplotlib import patheffects
        
        def _apply_filter_to_data(data_map, ftype, fparams):
            """Apply the specified filter to the data."""
            filtered = np.copy(data_map)
            if ftype == 'median':
                filtered = median_filter(filtered, **fparams)
            elif ftype == 'gaussian':
                filtered = gaussian_filter(filtered, **fparams)
            elif ftype == 'bilateral':
                filtered = denoise_bilateral(filtered, **fparams)
            elif ftype == 'wiener':
                filtered = wiener(filtered, **fparams)
            elif ftype == 'anisotropic_diffusion':
                filtered = denoise_tv_chambolle(filtered, **fparams)
            elif ftype == 'total_variation':
                filtered = denoise_tv_bregman(filtered, **fparams)
            elif ftype == 'non_local_means':
                filtered = denoise_nl_means(filtered, **fparams)
            return filtered
        
        def _plot_cell_heatmap(data_map, mask, cell_mask, data_type_label, 
                               colorbar_range, cmap, annotate_data, scale, save_path=None):
            """
            Helper function to plot heatmaps with unselected areas shown as lowest color.
            """
            # Create a copy for visualization
            vis_data = np.copy(data_map)
            
            # Determine vmin/vmax from colorbar_range
            vmin, vmax = None, None
            if colorbar_range is not None:
                # Only use valid data points for determining range
                valid_data = vis_data[np.isfinite(vis_data)]
                if len(valid_data) > 0:
                    if colorbar_range == 'auto':
                        p_low, p_high = 2, 98
                        vmin, vmax = np.nanpercentile(valid_data, [p_low, p_high])
                    elif isinstance(colorbar_range, tuple) and len(colorbar_range) == 3 and colorbar_range[0] == 'auto':
                        _, p_low, p_high = colorbar_range
                        vmin, vmax = np.nanpercentile(valid_data, [p_low, p_high])
                    elif isinstance(colorbar_range, (list, tuple)) and len(colorbar_range) == 2:
                        vmin, vmax = colorbar_range
            
            if vmin is None or vmax is None:
                vmin = np.nanmin(vis_data)
                vmax = np.nanmax(vis_data)
            
            # Fill unselected areas with vmin (lowest color from colorbar)
            vis_data[~mask] = vmin
            
            # Create figure
            plt.figure(figsize=(8, 6))
            im = plt.imshow(
                vis_data,
                cmap=cmap,
                origin='lower',
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                interpolation=None
            )
            
            plt.xlabel("μm", fontsize=20)
            plt.ylabel("μm", fontsize=20)
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            
            # Set the ticks based on the scaling factors
            x_ticks = np.arange(0, vis_data.shape[1], step=10) * scale
            y_ticks = np.arange(0, vis_data.shape[0], step=10) * scale
            
            def _format_tick_label(value):
                rounded = int(np.round(value))
                if np.isclose(value, rounded):
                    return str(rounded)
                formatted = f"{value:.2f}".rstrip('0').rstrip('.')
                return formatted
            
            x_tick_labels = [_format_tick_label(val) for val in x_ticks]
            y_tick_labels = [_format_tick_label(val) for val in y_ticks]
            
            plt.xticks(
                ticks=np.arange(0, vis_data.shape[1], step=10),
                labels=x_tick_labels,
                rotation=45
            )
            plt.yticks(
                ticks=np.arange(0, vis_data.shape[0], step=10),
                labels=y_tick_labels
            )
            
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Add colorbar with label
            cbar = plt.colorbar(im)
            cbar_label = f"Brillouin {data_type_label} [GHz]"
            cbar.set_label(cbar_label, fontsize=20)
            cbar.ax.tick_params(labelsize=20)
            
            # Annotate statistical information
            if annotate_data:
                valid_vals = vis_data[np.isfinite(vis_data) & mask]
                if len(valid_vals) > 0:
                    mean_val = np.nanmean(valid_vals)
                    median_val = np.nanmedian(valid_vals)
                    std_val = np.nanstd(valid_vals)
                else:
                    mean_val = median_val = std_val = np.nan
                    
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
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.show()
        
        # Create mask for selected regions (cells)
        heatmap_mask = cell_mask.copy()
        
        # Apply filter to data for heatmaps
        if filter_type is not None and filter_params is not None:
            shift_data_for_heatmap = _apply_filter_to_data(all_shift_original, filter_type, filter_params)
            fwhm_data_for_heatmap = _apply_filter_to_data(fwhm_all_original, filter_type, filter_params)
        else:
            shift_data_for_heatmap = all_shift_original
            fwhm_data_for_heatmap = fwhm_all_original
        
        # Create safe filename from title
        safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('-', '_')
        
        # Plot shift heatmap
        shift_save_path = None
        if heatmap_save_path:
            from pathlib import Path
            base = Path(heatmap_save_path)
            # Check if path is a directory or has a file extension
            if base.suffix:
                # It's a file path, use parent as directory
                directory = base.parent
            else:
                # It's a directory path
                directory = base
            shift_save_path = directory / f"{safe_title}_shift_heatmap.png"
        
        _plot_cell_heatmap(
            shift_data_for_heatmap,
            heatmap_mask,
            cell_mask,
            "Shift",
            heatmap_colorbar_range,
            heatmap_cmap,
            heatmap_annotate,
            scaling_factor,
            save_path=shift_save_path
        )
        
        # Plot FWHM heatmap
        fwhm_save_path = None
        if heatmap_save_path:
            from pathlib import Path
            base = Path(heatmap_save_path)
            if base.suffix:
                directory = base.parent
            else:
                directory = base
            fwhm_save_path = directory / f"{safe_title}_fwhm_heatmap.png"
        
        _plot_cell_heatmap(
            fwhm_data_for_heatmap,
            heatmap_mask,
            cell_mask,
            "FWHM",
            heatmap_colorbar_range,
            heatmap_cmap,
            heatmap_annotate,
            scaling_factor,
            save_path=fwhm_save_path
        )

    return cell_shift_maps, cell_fwhm_maps, background_shift_maps, background_fwhm_maps
