
import os
import re
import zipfile
import tempfile
from collections import Counter
from contextlib import contextmanager

import numpy as np
from tqdm.auto import tqdm

from .data_registry import register_lateral_step

_LATERAL_STEP_PATTERN = re.compile(r"Lateral\s*step\s*:?\s*([-+]?\d*[\.,]?\d+)", re.IGNORECASE)


def _parse_lateral_step_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as label_file:
            for line in label_file:
                match = _LATERAL_STEP_PATTERN.search(line)
                if match:
                    value_str = match.group(1).replace(',', '.')
                    try:
                        return float(value_str)
                    except ValueError:
                        continue
    except OSError:
        return None
    return None


def _auto_detect_file_label(directory_path):
    """Automatically detect file_label by examining .asc files in the directory.
    
    Looks for files matching the pattern {file_label}B*_Z*Y*X*.asc and extracts
    the prefix before 'B'.
    
    Returns:
        str or None: The detected file_label, or None if no matching files found.
    """
    if not os.path.isdir(directory_path):
        return None

    file_labels = []
    # Pattern matches: {file_label}B (optional chars) _ (anything) _ Z{num}Y{num}X{num}.asc
    # We're looking for the prefix before 'B'
    pattern = re.compile(r'(.+?)B_Z\d+Y\d+X\d+\.asc$')

    for file_name in os.listdir(directory_path):
        if not file_name.lower().endswith('.asc'):
            continue

        match = pattern.match(file_name)
        if match:
            file_labels.append(match.group(1))

    if not file_labels:
        return None

    # Return the most common file_label (in case there are variations)
    counter = Counter(file_labels)
    return counter.most_common(1)[0][0]


def _find_lateral_step(directory_path, file_label):
    candidate_names = []
    if file_label:
        candidate_names.append(f"{file_label}.txt")
        sanitized_label = file_label.replace(' ', '_')
        if sanitized_label != file_label:
            candidate_names.append(f"{sanitized_label}.txt")
    candidate_names.append('file_label.txt')

    checked_paths = set()
    for name in candidate_names:
        label_path = os.path.join(directory_path, name)
        if label_path in checked_paths:
            continue
        checked_paths.add(label_path)
        if os.path.isfile(label_path):
            value = _parse_lateral_step_from_file(label_path)
            if value is not None:
                return value

    for name in os.listdir(directory_path):
        if not name.lower().endswith('.txt'):
            continue
        label_path = os.path.join(directory_path, name)
        if label_path in checked_paths:
            continue
        value = _parse_lateral_step_from_file(label_path)
        if value is not None:
            return value
    return None


@contextmanager
def _materialize_dataset(directory_path, file_label):
    """Yield a real directory containing the dataset, extracting zip archives on demand."""
    if os.path.isdir(directory_path):
        yield directory_path
        return

    if os.path.isfile(directory_path) and directory_path.lower().endswith('.zip'):
        with zipfile.ZipFile(directory_path) as zf, tempfile.TemporaryDirectory() as tmpdir:
            zf.extractall(tmpdir)

            # Prefer a folder matching the file label (common packaging pattern)
            if file_label:
                label_variants = [file_label, file_label.replace(' ', '_')]
                for variant in label_variants:
                    candidate = os.path.join(tmpdir, variant)
                    if os.path.isdir(candidate):
                        yield candidate
                        return

            # If the archive contains a single top-level directory, drill into it
            top_dirs = [name for name in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, name))]
            # Ignore __MACOSX folder which is common in zips created on macOS
            valid_top_dirs = [d for d in top_dirs if d != '__MACOSX']

            if len(valid_top_dirs) == 1:
                yield os.path.join(tmpdir, valid_top_dirs[0])
                return

            # Search for a directory containing files that start with the file_label
            # This handles cases where the folder name doesn't match the label
            # or files are nested deeper.
            if file_label:
                best_candidate = None
                max_matches = 0
                
                for root, dirs, files in os.walk(tmpdir):
                    # Skip __MACOSX directories during walk
                    if '__MACOSX' in dirs:
                        dirs.remove('__MACOSX')
                    
                    matches = sum(1 for f in files if f.startswith(file_label))
                    if matches > max_matches:
                        max_matches = matches
                        best_candidate = root
                
                if best_candidate and max_matches > 0:
                    yield best_candidate
                    return

            # Fallback: use the extraction root
            yield tmpdir
        return

    raise FileNotFoundError(f"Dataset path '{directory_path}' not found or unsupported format.")


def parse_brillouin_set(directory_path, file_label=None):
    """Parse Brillouin spectroscopy data from a directory or zip file.
    
    Args:
        directory_path: Path to the directory or zip file containing the data.
        file_label: Optional label prefix for the files. If not provided, will be
                   automatically detected from the .asc files in the directory.
    
    Returns:
        numpy.ndarray: 3D array of Brillouin spectra, or None if no files found.
    """
    # Auto-detect file_label if not provided
    if file_label is None:
        # First materialize the dataset to get the working directory
        # This extracts zip files if needed
        with _materialize_dataset(directory_path, file_label) as working_dir:
            # Try to auto-detect file_label from the working directory
            file_label = _auto_detect_file_label(working_dir)
            
            if file_label is None:
                # If not found in top level, search recursively
                for root, dirs, files in os.walk(working_dir):
                    if '__MACOSX' in dirs:
                        dirs.remove('__MACOSX')
                    file_label = _auto_detect_file_label(root)
                    if file_label:
                        break
        
        if file_label is None:
            raise ValueError(
                "Could not automatically detect file_label. "
                "Please provide file_label parameter or ensure the directory contains "
                "files matching the pattern {file_label}B_*_Z*Y*X*.asc"
            )
    
    # Now parse with the detected or provided file_label
    with _materialize_dataset(directory_path, file_label) as working_dir:
        return _parse_brillouin_directory(
            directory_path=working_dir,
            source_path=directory_path,
            file_label=file_label,
        )


def parse_brillouin_set_directory(directory_path, expand_label=False, show_progress=True):
    """Parse Brillouin spectroscopy data from all subdirectories and zip files.
    
    Recursively searches through subdirectories to find directories or zip files
    containing Brillouin spectroscopy data, parses each one, and returns a dictionary
    where keys are file_labels and values are the parsed spectra arrays.
    
    Args:
        directory_path: Path to the root directory to search for data.
        expand_label: If True, prepend folder name to file_label (folder_name_file_label).
                     If False, use only file_label as key. Default is False.
    
    Returns:
        dict: Dictionary mapping file_label strings to numpy.ndarray spectra.
              Empty dictionary if no data found.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory path '{directory_path}' is not a valid directory.")

    results = {}

    # First pass: count how many datasets we will attempt to parse
    total_datasets = 0
    for root, dirs, files in os.walk(directory_path):
        if '__MACOSX' in dirs:
            dirs.remove('__MACOSX')

        has_brillouin_files = any(
            f.lower().endswith('.asc') and 'B' in f.upper()
            for f in files
        )
        zip_files = [f for f in files if f.lower().endswith('.zip')]

        if has_brillouin_files:
            total_datasets += 1
        total_datasets += len(zip_files)

    # Setup progress bar
    pbar = None
    if show_progress and total_datasets > 0:
        pbar = tqdm(
            total=total_datasets,
            desc="Parsing datasets",
            unit="dataset",
        )

    # Second pass: original parsing logic, now with progress updates
    for root, dirs, files in os.walk(directory_path):
        # Skip __MACOSX directories
        if '__MACOSX' in dirs:
            dirs.remove('__MACOSX')

        # Check if current directory has Brillouin data files
        has_brillouin_files = any(
            f.lower().endswith('.asc') and 'B' in f.upper()
            for f in files
        )

        # Check for zip files in current directory
        zip_files = [f for f in files if f.lower().endswith('.zip')]

        # Try to parse directory if it has Brillouin files
        if has_brillouin_files:
            try:
                # Detect file_label first to avoid duplicate materialization
                with _materialize_dataset(root, None) as working_dir:
                    file_label = _auto_detect_file_label(working_dir)
                    if file_label is None:
                        # Try recursive search if not found in top level
                        for walk_root, walk_dirs, walk_files in os.walk(working_dir):
                            if '__MACOSX' in walk_dirs:
                                walk_dirs.remove('__MACOSX')
                            file_label = _auto_detect_file_label(walk_root)
                            if file_label:
                                break
                
                if file_label:
                    # Parse with the detected file_label
                    spectra = parse_brillouin_set(root, file_label=file_label)
                    if spectra is not None:
                        # Optionally expand label with parent folder and current folder name
                        if expand_label:
                            parent_folder = os.path.basename(os.path.dirname(root))
                            current_folder = os.path.basename(root)
                            original_label = f"{parent_folder}_{current_folder}_{file_label}"
                        else:
                            original_label = file_label

                        # Handle duplicate file_labels by appending a suffix
                        label_key = original_label
                        counter = 1
                        while label_key in results:
                            label_key = f"{original_label}_{counter}"
                            counter += 1

                        results[label_key] = spectra
            except (ValueError, FileNotFoundError):
                # Skip directories that can't be parsed
                pass
            finally:
                if pbar is not None:
                    pbar.update(1)

        # Try to parse zip files
        for zip_file in zip_files:
            zip_path = os.path.join(root, zip_file)
            try:
                # Detect file_label first to avoid duplicate materialization
                with _materialize_dataset(zip_path, None) as working_dir:
                    file_label = _auto_detect_file_label(working_dir)
                    if file_label is None:
                        # Try recursive search if not found in top level
                        for walk_root, walk_dirs, walk_files in os.walk(working_dir):
                            if '__MACOSX' in walk_dirs:
                                walk_dirs.remove('__MACOSX')
                            file_label = _auto_detect_file_label(walk_root)
                            if file_label:
                                break
                
                if file_label:
                    # Parse with the detected file_label
                    spectra = parse_brillouin_set(zip_path, file_label=file_label)
                    if spectra is not None:
                        # Optionally expand label with parent folder and zip filename (without extension)
                        if expand_label:
                            parent_folder = os.path.basename(os.path.dirname(zip_path))
                            zip_name = os.path.splitext(zip_file)[0]
                            original_label = f"{parent_folder}_{zip_name}_{file_label}"
                        else:
                            original_label = file_label

                        # Handle duplicate file_labels by appending a suffix
                        label_key = original_label
                        counter = 1
                        while label_key in results:
                            label_key = f"{original_label}_{counter}"
                            counter += 1

                        results[label_key] = spectra
            except (ValueError, FileNotFoundError):
                # Skip zip files that can't be parsed
                pass
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    return results


def _parse_brillouin_directory(directory_path, source_path, file_label):
    data_dict = {}
    files = os.listdir(directory_path)
    lateral_step = _find_lateral_step(directory_path, file_label)
    
    # Regex pattern for matching files: {file_label}B_Z{num}Y{num}X{num}.asc
    file_pattern = re.compile(
        re.escape(file_label) + r'B_Z(\d+)Y(\d+)X(\d+)\.asc$'
    )
    
    for file_name in files:
        match = file_pattern.match(file_name)
        if match:
            z_coord = int(match.group(1))
            y_coord = int(match.group(2))
            x_coord = int(match.group(3))
            file_path = os.path.join(directory_path, file_name)
            
            try:
                data = np.loadtxt(file_path)
                
                # Handle different array shapes
                if data.ndim == 1:
                    print(f"Warning: File {file_name} has only 1 row. Skipping entire dataset '{file_label}'.")
                    return None
                elif data.ndim == 2 and data.shape[1] >= 2:
                    data_dict[(x_coord, y_coord, z_coord)] = data[:, 1]
                else:
                    print(f"Warning: File {file_name} has unexpected shape {data.shape}. Skipping entire dataset '{file_label}'.")
                    return None
            except Exception as e:
                print(f"Warning: Could not load {file_name}: {e}. Skipping entire dataset '{file_label}'.")
                return None
    
    # Check if data_dict is empty
    if not data_dict:
        print(f"No files found in {directory_path} with label {file_label}.")
        return None

    max_x = max([key[0] for key in data_dict.keys()]) + 1
    max_y = max([key[1] for key in data_dict.keys()]) + 1
    max_z = max([key[2] for key in data_dict.keys()]) + 1

    brillouin_spectra = np.zeros((max_x, max_y, max_z), dtype=object)
    for (x_coord, y_coord, z_coord), spectrum in data_dict.items():
        brillouin_spectra[x_coord][y_coord][z_coord] = spectrum

    register_lateral_step(
        source=brillouin_spectra,
        lateral_step=lateral_step,
        directory_path=source_path,
        file_label=file_label,
    )

    return brillouin_spectra


def parse_raman_set(directory_path, file_label):
    with _materialize_dataset(directory_path, file_label) as working_dir:
        data_dict = {}
        files = os.listdir(working_dir)
        # Check if files exist
        for file_name in files:
            if file_name.startswith(file_label + "R") and file_name.endswith('.asc'):
                parts = file_name.split('_')
                coords = parts[-1].replace('.asc', '')
                z_coord = int(coords.split('Z')[1].split('Y')[0])
                y_coord = int(coords.split('Y')[1].split('X')[0])
                x_coord = int(coords.split('X')[1])
                file_path = os.path.join(working_dir, file_name)
                data = np.loadtxt(file_path)
                data_dict[(x_coord, y_coord, z_coord)] = data[:, 1]  # Second column is spectral data
        # Check if data_dict is empty
        if not data_dict:
            print(f"No files found in {working_dir} with label {file_label}.")
            return None
