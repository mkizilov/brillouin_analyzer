# Parsing Module

The `brillouin_parser` module provides functionality to load and parse Brillouin and Raman spectra data from directories or ZIP archives.

## Functions

### `parse_brillouin_set`

```python
def parse_brillouin_set(directory_path, file_label):
    ...
```

Parses a set of Brillouin spectra files.

**Arguments:**

*   `directory_path` (str): Path to the directory or ZIP file containing the data.
*   `file_label` (str): The label used to identify the files (e.g., the prefix of the filenames).

**Returns:**

*   `numpy.ndarray`: A 3D array of object type, where each element contains the spectral data (intensity) for a specific (x, y, z) coordinate. The indices of the array correspond to the spatial coordinates.

**Description:**

This function locates the dataset (extracting it from a ZIP file if necessary), finds the lateral step size, and reads `.asc` files matching the pattern `{file_label}B*_Z...Y...X...asc`. It constructs a 3D array representing the spatial distribution of the spectra.

It also registers the lateral step metadata using `data_registry.register_lateral_step`.

### `parse_raman_set`

```python
def parse_raman_set(directory_path, file_label):
    ...
```

Parses a set of Raman spectra files.

**Arguments:**

*   `directory_path` (str): Path to the directory or ZIP file containing the data.
*   `file_label` (str): The label used to identify the files.

**Returns:**

*   `dict` or `None`: A dictionary where keys are `(x, y, z)` tuples and values are the spectral data (intensity), or `None` if no files are found. (Note: The current implementation returns a dictionary, unlike `parse_brillouin_set` which returns a 3D array. *Check implementation details if this is intended behavior*).

**Description:**

Similar to `parse_brillouin_set`, this function reads `.asc` files matching the pattern `{file_label}R*_Z...Y...X...asc`.

## File Naming Convention

The parser expects files to follow a specific naming convention to extract spatial coordinates:

*   **Brillouin files:** `{file_label}B*_Z{z}Y{y}X{x}.asc`
*   **Raman files:** `{file_label}R*_Z{z}Y{y}X{x}.asc`

Where `{x}`, `{y}`, and `{z}` are integer coordinates.

## Lateral Step Detection

The parser attempts to automatically detect the "Lateral step" size from text files in the directory. It looks for files named:
1.  `{file_label}.txt`
2.  `{sanitized_label}.txt` (spaces replaced by underscores)
3.  `file_label.txt`

It searches for a line matching the pattern `Lateral step : {value}`.

## ZIP Archive Support

The functions support reading data directly from ZIP archives. If `directory_path` points to a `.zip` file, the parser will:
1.  Extract the archive to a temporary directory.
2.  Search for a folder matching `file_label` or its sanitized version.
3.  If not found, check if there is a single top-level directory.
4.  If not found, search for a directory containing files starting with `file_label`.
5.  Use the found directory as the source for parsing.
