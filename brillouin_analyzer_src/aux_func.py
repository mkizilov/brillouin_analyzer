import numpy as np  # For array handling and mathematical operations
import pandas as pd  # For DataFrame operations
import pickle  # For saving and loading objects to/from files

def save_results_to_file(results_list, filename):
    """
    Save results_list to a file using pickle.

    :param results_list: The results_list to save.
    :param filename: The filename to save the data to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(results_list, f)

def load_results_from_file(filename):
    """
    Load results_list from a file using pickle.

    :param filename: The filename to load the data from.
    :return: The loaded results_list.
    """
    with open(filename, 'rb') as f:
        results_list = pickle.load(f)
    return results_list

def convert_to_df_map(peaks_map):
    """
    Convert the output from get_brillouin_peaks_2d_list into a df_map format.

    Parameters:
    -----------
    peaks_map : dict
        Dictionary returned by get_brillouin_peaks_2d_list where keys are (x, y) coordinates and values are dictionaries.

    Returns:
    --------
    df_map : dict
        A dictionary where keys are (x, y) tuples and values are DataFrames or NaNs.
    """
    df_map = {}
    # Loop through the peaks_map and extract the DataFrames
    for (x, y), result in peaks_map.items():
        if result is not None and isinstance(result, dict) and 'df' in result:
            df_map[(x, y)] = result  # Extract the result dictionary containing the DataFrame
        else:
            df_map[(x, y)] = np.nan  # If no DataFrame exists, mark it as NaN
    return df_map

def extract_peak_values(df_map, peak_type="Brillouin Left", data_type="Shift", aggregation="median", match_type='exact'):
    """
    Extracts the median, mean, or robust mean values of 'Shift' or 'FWHM' for specified peak types from a dictionary of DataFrames.
    
    Parameters:
    -----------
    df_map : dict
        A dictionary where keys are tuples (x, y) and values are dictionaries containing DataFrames.
    peak_type : str
        The type of peak to extract (e.g., 'Brillouin Left', 'Brillouin Right', 'Brillouin', 'Laser').
    data_type : str
        The type of data to extract ('Shift' or 'FWHM').
    aggregation : str
        The aggregation method ('median', 'mean', 'robust', or 'second').
        If 'second', only the second left and right Brillouin peaks will be used.
    match_type : str
        How to match the peak_type ('exact' or 'contains').
    
    Returns:
    --------
    np.ndarray
        A 2D array containing the extracted values.
    """
    # Get the dimensions of the data
    x_dim = max([key[0] for key in df_map.keys()]) + 1
    y_dim = max([key[1] for key in df_map.keys()]) + 1

    # Initialize an array with NaN values
    data_map = np.full((x_dim, y_dim), np.nan)

    # Loop over the dictionary and extract values
    for (i, j), result in df_map.items():
        if result is not None and not isinstance(result, float) and not isinstance(result, tuple):
            # Extract DataFrame
            df = result['df']
            
            if aggregation == "second":
                # For 'second' aggregation, we need both left and right second peaks
                # Get all peaks of each type
                left_peaks = df[df["Peak Type"] == "Brillouin Left"]
                right_peaks = df[df["Peak Type"] == "Brillouin Right"]
                
                # Select the second peak of each type (index 1 if it exists)
                left_second = left_peaks.iloc[[1]] if len(left_peaks) > 1 else pd.DataFrame()
                right_second = right_peaks.iloc[[1]] if len(right_peaks) > 1 else pd.DataFrame()
                
                # Check if both peaks exist
                if not left_second.empty and not right_second.empty:
                    left_value = left_second[data_type].values[0] if len(left_second[data_type].values) > 0 else np.nan
                    right_value = right_second[data_type].values[0] if len(right_second[data_type].values) > 0 else np.nan
                    
                    # Use average of left and right values
                    values = np.array([left_value, right_value])
                    data_map[i, j] = np.nanmean(values)
            else:
                if match_type == 'exact':
                    peak_data = df[df["Peak Type"] == peak_type]
                    if not peak_data.empty:
                        values = peak_data[data_type].values
                        # Apply aggregation
                        data_map[i, j] = aggregate_values(values, aggregation)
                elif match_type == 'contains':
                    # Get unique Peak Types that match
                    matching_peak_types = df["Peak Type"].unique()
                    matching_peak_types = [pt for pt in matching_peak_types if peak_type in pt]
                    aggregated_values = []
                    for pt in matching_peak_types:
                        pt_peak_data = df[df["Peak Type"] == pt]
                        if not pt_peak_data.empty:
                            values = pt_peak_data[data_type].values
                            aggregated_value = aggregate_values(values, aggregation)
                            aggregated_values.append(aggregated_value)
                    if aggregated_values:
                        # Compute mean of the aggregated values, equally weighted
                        data_map[i, j] = np.nanmean(aggregated_values)
                else:
                    raise ValueError(f"Unknown match_type: {match_type}")
    return data_map


def aggregate_values(values, aggregation):
    """
    Helper function to apply aggregation method to values.
    """
    values = values[np.isfinite(values)]  # Ensure values are finite
    if len(values) == 0:
        return np.nan
    if aggregation == "median":
        return np.nanmedian(values)
    elif aggregation == "mean":
        return np.nanmean(values)
    elif aggregation == "robust":
        # Implement the robust mean calculation
        if len(values) > 1:
            # Calculate mean and standard deviation
            mean_value = np.nanmean(values)
            std_dev = np.nanstd(values, ddof=1)
            if std_dev == 0:
                return mean_value  # All values are the same
            # Identify outliers based on standard deviation
            deviations = np.abs(values - mean_value)
            outliers = deviations > 2 * std_dev

            # Exclude outliers
            robust_values = values[~outliers]

            # Compute robust mean
            if len(robust_values) > 0:
                return np.nanmean(robust_values)
            else:
                return np.nanmedian(values)  # Fall back to median if all are outliers
        else:
            # For a single value, use it directly
            return values[0]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
def swap_horizontal_chunks(data, x_split=21):
    """
    Swaps two horizontal chunks of a 2D matrix.
        
    Parameters:
    data: Data structure where elements can be accessed as data[i][x,y]
    x_split: The x-coordinate to split at (default: 21)
        
    Returns:
    A new data structure with the left and right chunks swapped
    """
    import copy
    import numpy as np
        
    # Make a deep copy to avoid modifying the original data
    result = copy.deepcopy(data)
        
    # Process each matrix in the data
    for i in range(len(result)):
        matrix = result[i]
            
        # Find all valid coordinates
        valid_coords = []
        for x in range(100):  # Reasonably high limit to scan for coordinates
            for y in range(100):
                try:
                    _ = matrix[x, y]
                    valid_coords.append((x, y))
                except (IndexError, KeyError, TypeError):
                    continue
            
        if not valid_coords:
            continue  # Skip if no valid coordinates found
            
        # Determine matrix dimensions
        max_x = max(x for x, _ in valid_coords) + 1
        max_y = max(y for _, y in valid_coords) + 1
            
        # Create a new matrix of the same type
        new_matrix = type(matrix)()
            
        # Process and swap chunks
        for x, y in valid_coords:
            if x <= x_split:
                # Move left chunk to right
                new_x = x + (max_x - (x_split + 1))
            else:
                # Move right chunk to left
                new_x = x - (x_split + 1)
                
            # Copy the value to its new position
            new_matrix[new_x, y] = matrix[x, y]
            
        result[i] = new_matrix
        
    return result

def swap_vertical_chunks(data, y_split=21):
    """
    Swaps two vertical chunks of a 2D matrix.
        
    Parameters:
    data: Data structure where elements can be accessed as data[i][x,y]
    y_split: The y-coordinate to split at (default: 21)
        
    Returns:
    A new data structure with the top and bottom chunks swapped
    """
    import copy
    import numpy as np
        
    # Make a deep copy to avoid modifying the original data
    result = copy.deepcopy(data)
        
    # Process each matrix in the data
    for i in range(len(result)):
        matrix = result[i]
            
        # Find all valid coordinates
        valid_coords = []
        for x in range(100):  # Reasonably high limit to scan for coordinates
            for y in range(100):
                try:
                    _ = matrix[x, y]
                    valid_coords.append((x, y))
                except (IndexError, KeyError, TypeError):
                    continue
            
        if not valid_coords:
            continue  # Skip if no valid coordinates found
            
        # Determine matrix dimensions
        max_x = max(x for x, _ in valid_coords) + 1
        max_y = max(y for _, y in valid_coords) + 1
            
        # Create a new matrix of the same type
        new_matrix = type(matrix)()
            
        # Process and swap chunks
        for x, y in valid_coords:
            if y <= y_split:
                # Move top chunk to bottom
                new_y = y + (max_y - (y_split + 1))
            else:
                # Move bottom chunk to top
                new_y = y - (y_split + 1)
                
            # Copy the value to its new position
            new_matrix[x, new_y] = matrix[x, y]
            
        result[i] = new_matrix
        
    return result