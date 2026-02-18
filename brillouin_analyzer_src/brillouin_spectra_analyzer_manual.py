import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from scipy.signal import peak_widths, find_peaks, peak_prominences
from scipy.signal import savgol_filter as _savgol
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import get_context
import warnings, os, numbers
from functools import lru_cache
from .data_registry import BrillouinPeaksMap, get_lateral_step
from sklearn.decomposition import PCA
try:
    import pywt
except ImportError:  # Optional dependency for wavelet filtering
    pywt = None

# Try to import numba for JIT compilation of hot functions
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

warnings.filterwarnings("ignore")

# Pre-bind frequently used numpy functions to avoid attribute lookup overhead
_np_max = np.max
_np_min = np.min
_np_abs = np.abs
_np_sqrt = np.sqrt
_np_clip = np.clip
_np_searchsorted = np.searchsorted
_np_argmax = np.argmax
_np_argmin = np.argmin
_np_isfinite = np.isfinite
_np_nanmedian = np.nanmedian
_np_nanmin = np.nanmin
_np_nanmax = np.nanmax

def _process_pixel_common(brillouin_spectra, coords, worker_kwargs, spectrum_data=None):
    x, y, z = coords
    try:
        local_kwargs = worker_kwargs if spectrum_data is None else {**worker_kwargs, 'spectrum_data': spectrum_data}
        result = analyze_brillouin_spectrum_manual(
            brillouin_spectra=brillouin_spectra,
            x_coord=x,
            y_coord=y,
            z_coord=z,
            **local_kwargs,
        )
    except Exception:
        result = None
    return {'pixel_key': (x, y, z), 'result': result}


def _process_pixel_with_payload(payload):
    coords, spectrum, worker_kwargs = payload
    return _process_pixel_common(None, coords, worker_kwargs, spectrum_data=spectrum)


# ---------- model ----------
# JIT-compiled Lorentzian for performance when numba is available
@njit(cache=True, fastmath=True)
def _lorentzian_core(x, amplitude, center, gamma):
    """JIT-compiled Lorentzian core computation."""
    gamma_sq = gamma * gamma
    return amplitude * gamma_sq / ((x - center)**2 + gamma_sq)


def lorentzian(x, amplitude, center, gamma):
    """Plain Lorentzian (no background). FWHM = 2*gamma."""
    return _lorentzian_core(x, amplitude, center, gamma)


# ---------- helpers ----------
def _normalize_range_pair(start, end, n):
    """Return clipped, sorted (start, end) in [0, n]. If invalid, return None."""
    if start is None or end is None:
        return None
    s = int(min(max(0, start), n))
    e = int(min(max(0, end), n))
    if s == e:
        return None
    if e < s:
        s, e = e, s
    return (s, e)


# ---------- filtering helpers ----------
def _canonical_filter_name(name):
    if name is None:
        return None
    alias_map = {
        'savgol_filter': 'savgol',
        'wavelet_filter': 'wavelet',
        'fourier_filter': 'fourier',
        'fft': 'fourier',
        'pca_filter': 'pca',
        'custom': 'callable',
        'function': 'callable',
    }
    key = str(name).strip().lower()
    return alias_map.get(key, key)


def _normalize_filter_entry(entry):
    if entry is None:
        return None
    if isinstance(entry, dict):
        cfg = dict(entry)
    elif isinstance(entry, str):
        cfg = {'type': entry}
    elif isinstance(entry, (list, tuple)):
        if not entry:
            return None
        if isinstance(entry[0], str):
            cfg = {'type': entry[0]}
            if len(entry) > 1:
                second = entry[1]
                if isinstance(second, dict):
                    cfg.update(second)
                else:
                    cfg['params'] = second
        elif len(entry) == 2 and all(isinstance(v, numbers.Number) for v in entry):
            cfg = {'type': 'savgol', 'params': entry}
        else:
            return None
    else:
        return None

    if 'type' not in cfg and any(k in cfg for k in ('callable', 'function', 'func')):
        cfg['type'] = 'callable'

    enabled = cfg.pop('enabled', True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {'false', '0', 'no'}
    if not enabled:
        return None

    ftype = _canonical_filter_name(cfg.get('type') or cfg.get('name') or cfg.get('filter'))
    if not ftype:
        return None
    cfg['type'] = ftype

    params = cfg.get('params')
    if isinstance(params, dict):
        for key, value in params.items():
            cfg.setdefault(key, value)

    apply_after_cut = cfg.get('apply_after_cut')
    if apply_after_cut is None:
        apply_after_cut = (ftype == 'pca')
    else:
        apply_after_cut = bool(apply_after_cut)
    cfg['apply_after_cut'] = apply_after_cut

    return cfg


def _prepare_filter_sequences(filter_settings):
    if not filter_settings:
        return [], []
    if not isinstance(filter_settings, (list, tuple)):
        sequence = [filter_settings]
    else:
        sequence = list(filter_settings)

    pre_filters, post_filters = [], []
    for entry in sequence:
        cfg = _normalize_filter_entry(entry)
        if not cfg:
            continue
        if cfg.get('apply_after_cut'):
            post_filters.append(cfg)
        else:
            pre_filters.append(cfg)
    return pre_filters, post_filters


def _apply_filter_sequence(signal, sequence):
    arr = np.asarray(signal, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not sequence:
        return arr

    # Build dispatch dict once (faster than repeated string comparisons)
    _filter_dispatch = {
        'savgol': _apply_savgol_filter,
        'wavelet': _apply_wavelet_filter,
        'fourier': _apply_fourier_filter,
        'pca': _apply_pca_filter,
        'callable': _apply_callable_filter,
    }

    for cfg in sequence:
        filter_type = cfg.get('type')
        if not filter_type:
            continue
        filter_func = _filter_dispatch.get(filter_type)
        if filter_func is not None:
            try:
                arr = filter_func(arr, cfg)
            except Exception as exc:
                print(f"Skipping filter '{filter_type}': {exc}")
    return arr


def _apply_savgol_filter(data, config):
    arr = np.asarray(data, dtype=float)
    arr_size = arr.size
    if arr_size < 3:
        return arr

    params = config.get('params')
    window_length = config.get('window_length', config.get('window'))
    polyorder = config.get('polyorder', config.get('order'))
    if window_length is None and isinstance(params, (list, tuple)):
        window_length = params[0] if len(params) > 0 else None
    if polyorder is None and isinstance(params, (list, tuple)):
        polyorder = params[1] if len(params) > 1 else None
    if window_length is None or polyorder is None:
        return arr

    wl = max(3, int(window_length))
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, arr_size - (1 - arr_size % 2))
    poly = int(polyorder)
    if wl <= poly or wl < 3:
        return arr

    extra_kwargs = {k: config[k] for k in ('deriv', 'delta', 'axis', 'mode', 'cval') if k in config}
    return _savgol(arr, wl, poly, **extra_kwargs)


def _apply_wavelet_filter(data, config):
    arr = np.asarray(data, dtype=float)
    if pywt is None:
        raise ImportError("PyWavelets is required for wavelet filtering.")
    if arr.size == 0:
        return arr

    wavelet = config.get('wavelet', 'db4')
    mode = config.get('mode', 'symmetric')
    level = config.get('level')
    try:
        if level is None:
            dec_len = pywt.Wavelet(wavelet).dec_len
            level = pywt.dwt_max_level(arr.size, dec_len)
            level = max(1, min(level, 6))
        coeffs = pywt.wavedec(arr, wavelet=wavelet, mode=mode, level=int(level))
    except Exception as exc:
        raise RuntimeError(f"Wavelet setup failed: {exc}")

    threshold = config.get('threshold')
    if threshold is None and len(coeffs) > 1:
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
        threshold = sigma * np.sqrt(2 * np.log(arr.size)) if arr.size else 0.0
    threshold_mode = config.get('threshold_mode', 'soft')

    filtered_coeffs = [coeffs[0]]
    for detail_coeff in coeffs[1:]:
        if threshold is None:
            filtered_coeffs.append(detail_coeff)
        else:
            filtered_coeffs.append(pywt.threshold(detail_coeff, value=threshold, mode=threshold_mode))

    reconstructed = pywt.waverec(filtered_coeffs, wavelet=wavelet, mode=mode)
    if reconstructed.size > arr.size:
        reconstructed = reconstructed[:arr.size]
    elif reconstructed.size < arr.size:
        padded = np.zeros_like(arr)
        padded[:reconstructed.size] = reconstructed
        reconstructed = padded
    return reconstructed


def _apply_fourier_filter(data, config):
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return arr

    sample_spacing = float(config.get('sample_spacing', 1.0))
    freq = np.fft.rfftfreq(arr.size, d=sample_spacing)
    spectrum = np.fft.rfft(arr)

    pass_type = str(config.get('pass_type', config.get('kind', 'lowpass'))).lower()
    cutoff = config.get('cutoff', 0.1)
    band = config.get('band')
    mask = np.ones_like(freq, dtype=bool)

    if pass_type == 'lowpass':
        mask = freq <= float(cutoff)
    elif pass_type == 'highpass':
        mask = freq >= float(cutoff)
    elif pass_type == 'bandpass' and band:
        low, high = band
        mask = (freq >= float(low)) & (freq <= float(high))
    elif pass_type in {'bandstop', 'notch'} and band:
        low, high = band
        mask = ~((freq >= float(low)) & (freq <= float(high)))

    attenuation = float(config.get('attenuation', 0.0))
    filtered_spectrum = spectrum * mask + attenuation * spectrum * (~mask)
    filtered = np.fft.irfft(filtered_spectrum, n=arr.size)
    return np.real(filtered)


def _apply_pca_filter(data, config):
    arr = np.asarray(data, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    filter_data = config.get('filter_data') or config.get('pca_filter') or config.get('model')
    if filter_data is None:
        return arr

    if isinstance(filter_data, dict):
        model = filter_data.get('pca_model') or filter_data.get('model')
        expected_length = filter_data.get('input_length')
        default_components = filter_data.get('n_components')
    else:
        model = filter_data
        expected_length = getattr(model, 'mean_', None)
        if expected_length is not None:
            expected_length = len(expected_length)
        default_components = getattr(model, 'n_components_', None)
    if expected_length is None and model is not None and hasattr(model, 'mean_'):
        expected_length = len(model.mean_)

    n_components = config.get('n_components', config.get('components_to_use', default_components))
    if model is None:
        raise ValueError("pca_filter requires a fitted PCA model or the output of pca_spectra_filter.")

    target_length = expected_length
    working, original_length, adjust_mode, resolved_length = _match_signal_length(arr, target_length)
    if resolved_length is None:
        resolved_length = working.size

    try:
        coeffs = model.transform(working.reshape(1, -1))
    except Exception as exc:
        raise RuntimeError(f"PCA transform failed: {exc}")

    if n_components is not None and int(n_components) < coeffs.shape[1]:
        n_keep = max(1, int(n_components))
        coeffs[:, n_keep:] = 0.0

    reconstructed = model.inverse_transform(coeffs)[0]
    if adjust_mode == 'pad':
        return reconstructed[:original_length]
    if adjust_mode == 'truncate':
        restored = arr.copy()
        restored[:resolved_length] = reconstructed
        return restored
    return reconstructed


def _apply_callable_filter(data, config):
    arr = np.asarray(data, dtype=float)
    func = config.get('callable') or config.get('function') or config.get('func')
    if not callable(func):
        return arr
    try:
        result = func(arr, config)
    except TypeError:
        result = func(arr)
    return np.asarray(result, dtype=float) if result is not None else arr


def _match_signal_length(signal, target_length):
    arr = np.asarray(signal, dtype=float)
    original_length = arr.size
    if target_length is None:
        return arr.copy(), original_length, None, None
    try:
        target = int(target_length)
    except (TypeError, ValueError):
        return arr.copy(), original_length, None, None
    if target <= 0 or target == original_length:
        return arr.copy(), original_length, None, target
    if original_length > target:
        return arr[:target].copy(), original_length, 'truncate', target
    padded = np.zeros(target, dtype=arr.dtype)
    padded[:original_length] = arr
    return padded, original_length, 'pad', target


# ---------- single-spectrum analysis ----------
def analyze_brillouin_spectrum_manual(
    brillouin_spectra,
    x_coord=None,
    y_coord=None,
    z_coord=None,
    free_spectral_range=29.98,
    laser_peak_ranges=[626, 986, 1269, 1510],
    brillouin_peak_ranges=(6, 11),
    spectrum_cut_range=(350, 1600),
    make_plot=(12, 8),
    debug_plot=False,
    # Lorentzian half-window in GHz (0 -> disable fitting)
    fit_lorentzians=1.5,

    # Optional filter pipeline configuration
    filter_settings=[{'type': 'savgol', 'params': (15, 6)},
    {'type': 'wavelet', 'wavelet': 'db6', 'level': 5},
    {'type': 'fft', 'params': (20, 100)}],
    save_plot=False,

    # Optional constraints for Lorentzian fitting (GHz)
    fwhm_bounds_GHz=(0.8, 4.0),     # (min_fwhm, max_fwhm) or None
    center_slack_GHz=0.5,    # ± slack around nominal center; if None, window-bound
    spectrum_data=None,
    baseline_spectra=True,
    poisson_weighting=False,
    match_brilouin_parameters=True,
    ignore_brilouin_peaks=False,
    ignore_laser_peaks=False,
):
    """
    Analyze a single spectrum at (x,y,z).
    - Supports 'tuple mode' for Brillouin ranges: brillouin_peak_ranges=(a_GHz, b_GHz),
      which means search in [lp-a, lp-b] left and [lp+a, lp+b] right around each laser peak.
    - If fitting is enabled (fit_lorentzians > 0), windows are taken in GHz around each nominal peak.
    - When baseline_spectra is True, the spectrum is shifted so its minimum is zero before analysis.
    - When poisson_weighting is True, Poisson-like weights are applied during Lorentzian fitting.
    - filter_settings accepts an ordered collection (list/tuple) of filter configuration dictionaries.
      Available filter types: 'savgol', 'wavelet', 'fourier', 'pca', and 'callable'. Each entry may
      set apply_after_cut=True to defer filtering until after spectrum_cut_range is applied.
    - ignore_laser_peaks: Can be a bool (backward compatible) or a dict with two keys:
      - 'brillouin_shift_average' (bool): If True, Brillouin shift is calculated as
        (center_right - center_left) / 2 (distance between Brillouin peaks divided by two).
      - 'ignore_laser_for_fit' (bool): If True, the FSR quadratic fit uses midpoints between
        Brillouin peak pairs instead of laser peak positions.
      If a single bool is provided, it's treated as {'brillouin_shift_average': value, 'ignore_laser_for_fit': False}.
    """

    baseline_spectra = bool(baseline_spectra)
    poisson_weighting = bool(poisson_weighting)

    # ---------------- coords & spectrum ----------------
    if spectrum_data is not None:
        spectrum = spectrum_data
    else:
        if brillouin_spectra is None:
            print("No Brillouin spectra provided.")
            return None
        if x_coord is None:
            x_coord = np.random.randint(0, brillouin_spectra.shape[0])
        if y_coord is None:
            y_coord = np.random.randint(0, brillouin_spectra.shape[1])
        if z_coord is None:
            z_coord = np.random.randint(0, brillouin_spectra.shape[2])

        try:
            spectrum = brillouin_spectra[x_coord][y_coord][z_coord]
        except IndexError:
            print(f"Coordinates ({x_coord}, {y_coord}, {z_coord}) are out of bounds.")
            return None
        if spectrum is None or isinstance(spectrum, int):
            print(f"No spectrum found at coordinates ({x_coord}, {y_coord}, {z_coord})")
            return None

    pre_filter_sequence, post_filter_sequence = _prepare_filter_sequences(filter_settings)

    if pre_filter_sequence:
        spectrum = _apply_filter_sequence(spectrum, pre_filter_sequence)

    # ---------------- prep spectrum ----------------
    spectrum = np.asarray(spectrum, dtype=float)
    spectrum[spectrum < 0] = 0.0
    if spectrum_cut_range:
        s0, s1 = spectrum_cut_range
        spectrum = spectrum[s0:s1]
    else:
        s0 = 0
    if post_filter_sequence:
        spectrum = _apply_filter_sequence(spectrum, post_filter_sequence)
    if baseline_spectra:
        spectrum -= np.min(spectrum)
    x_axis = np.arange(len(spectrum))

    # ---------------- find laser peaks (use given index ranges) ----------------
    laser_peaks_indices = []
    laser_peak_index_ranges = []

    def _is_scalar_like(value):
        if value is None:
            return False
        if np.isscalar(value):
            return True
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return True
        return False

    # Map ranges to local coordinates
    def _to_local_scalar(val, offset):
        if val is None: return None
        return val - offset

    def _to_local_range(rng, offset):
        if rng is None: return None
        return (rng[0] - offset, rng[1] - offset)

    local_laser_peak_ranges = laser_peak_ranges
    if laser_peak_ranges:
        if all(_is_scalar_like(entry) for entry in laser_peak_ranges):
            local_laser_peak_ranges = [_to_local_scalar(x, s0) for x in laser_peak_ranges]
        else:
            local_laser_peak_ranges = [_to_local_range(x, s0) for x in laser_peak_ranges]

    local_brillouin_peak_ranges = brillouin_peak_ranges
    tuple_mode = isinstance(brillouin_peak_ranges, tuple) and len(brillouin_peak_ranges) == 2
    if not tuple_mode and brillouin_peak_ranges:
         local_brillouin_peak_ranges = [(_to_local_range(l, s0), _to_local_range(r, s0)) for l, r in brillouin_peak_ranges]

    manual_laser_peaks = bool(local_laser_peak_ranges) and all(_is_scalar_like(entry) for entry in local_laser_peak_ranges)

    if manual_laser_peaks:
        n_points = len(spectrum)
        for entry in local_laser_peak_ranges:
            if entry is None:
                continue
            idx = int(round(float(entry)))
            idx = max(0, min(n_points - 1, idx))
            laser_peaks_indices.append(idx)
            laser_peak_index_ranges.append((idx, min(n_points, idx + 1)))
    else:
        for start, end in local_laser_peak_ranges:
            rng = _normalize_range_pair(start, end, len(spectrum))
            if not rng:
                continue
            s, e = rng
            laser_peak_index_ranges.append((s, e))
            idx = int(np.argmax(spectrum[s:e]) + s)
            laser_peaks_indices.append(idx)

    # ---------------- Brillouin peak detection ----------------
    match_brilouin_parameters = bool(match_brilouin_parameters)
    ignore_brilouin_peaks = bool(ignore_brilouin_peaks)
    
    # Parse ignore_laser_peaks: can be bool or dict with 'brillouin_shift_average' and 'ignore_laser_for_fit'
    if isinstance(ignore_laser_peaks, dict):
        brillouin_shift_average = bool(ignore_laser_peaks.get('brillouin_shift_average', False))
        ignore_laser_for_fit = bool(ignore_laser_peaks.get('ignore_laser_for_fit', False))
    else:
        # Backward compatibility: single bool means brillouin_shift_average only
        brillouin_shift_average = bool(ignore_laser_peaks)
        ignore_laser_for_fit = False
    
    if ignore_brilouin_peaks:
        match_brilouin_parameters = False
    # When using brillouin_shift_average, disable the shared-shift constraint
    if brillouin_shift_average:
        match_brilouin_parameters = False

    tuple_mode = isinstance(brillouin_peak_ranges, tuple) and len(brillouin_peak_ranges) == 2
    brillouin_peaks_indices = []
    per_laser_left_ranges, per_laser_right_ranges = [], []

    if ignore_brilouin_peaks and tuple_mode:
        pass  # handled after axis rescaling
    elif ignore_brilouin_peaks and not tuple_mode:
        for left_range, right_range in local_brillouin_peak_ranges:
            lr = _normalize_range_pair(left_range[0], left_range[1], len(spectrum))
            rr = _normalize_range_pair(right_range[0], right_range[1], len(spectrum))
            per_laser_left_ranges.append(lr)
            per_laser_right_ranges.append(rr)
            if lr:
                s, e = lr
                mid = int((s + e - 1) / 2)
                brillouin_peaks_indices.append(mid)
            if rr:
                s, e = rr
                mid = int((s + e - 1) / 2)
                brillouin_peaks_indices.append(mid)
    elif not tuple_mode:
        for left_range, right_range in local_brillouin_peak_ranges:
            # left
            rng_left = _normalize_range_pair(left_range[0], left_range[1], len(spectrum))
            per_laser_left_ranges.append(rng_left)
            if rng_left:
                s, e = rng_left
                il = int(np.argmax(spectrum[s:e]) + s)
                brillouin_peaks_indices.append(il)
            # right
            rng_right = _normalize_range_pair(right_range[0], right_range[1], len(spectrum))
            per_laser_right_ranges.append(rng_right)
            if rng_right:
                s, e = rng_right
                ir = int(np.argmax(spectrum[s:e]) + s)
                brillouin_peaks_indices.append(ir)

        if len(laser_peaks_indices) != len(brillouin_peak_ranges):
            print("Number of laser peaks and Brillouin peak pairs do not match.")
            return None

    # ---------------- FSR quadratic fit ----------------
    laser_peaks_indices = np.array(laser_peaks_indices, dtype=int)
    N_l = len(laser_peaks_indices)
    expected_positions = np.arange(N_l) * free_spectral_range

    def _resid(params, x, y):
        a, b, c = params
        return a * x**2 + b * x + c - y

    init = [1e-16, 1e-16, 0.0]
    lb   = [1e-16, 1e-16, -np.inf]
    ub   = [np.inf, np.inf,  np.inf]
    
    # Determine which indices to use for FSR fit
    if ignore_laser_for_fit and not tuple_mode and len(brillouin_peaks_indices) == 2 * N_l:
        # Non-tuple mode with ignore_laser_for_fit: use midpoints between Brillouin peaks
        fit_indices = np.array([
            (brillouin_peaks_indices[2*i] + brillouin_peaks_indices[2*i + 1]) / 2.0
            for i in range(N_l)
        ])
    else:
        # Default: use laser peak indices (tuple mode will refit later if ignore_laser_for_fit)
        fit_indices = laser_peaks_indices
    
    res = least_squares(_resid, x0=init, args=(fit_indices, expected_positions), bounds=(lb, ub))
    if not res.success:
        print("Quadratic fit unsuccessful.")
        return None

    a, b, c = [float(v) for v in res.x]
    rescaled_x_axis = a * x_axis**2 + b * x_axis + c  # in GHz

    # If tuple mode, determine Brillouin peaks from GHz windows around each laser
    if tuple_mode:
        a_val, b_val = local_brillouin_peak_ranges
        bi = []
        keep_lp = []
        keep_ranges = []
        keep_left_idx_ranges = []
        keep_right_idx_ranges = []
        for lp, lr in zip(laser_peaks_indices.tolist(), laser_peak_index_ranges):
            lp_pos = rescaled_x_axis[lp]
            # left GHz band [lp-b, lp-a], right GHz band [lp+a, lp+b]
            lg0, lg1 = lp_pos - b_val, lp_pos - a_val
            rg0, rg1 = lp_pos + a_val, lp_pos + b_val

            sl = _np_searchsorted(rescaled_x_axis, min(lg0, lg1))
            el = _np_searchsorted(rescaled_x_axis, max(lg0, lg1))
            sr = _np_searchsorted(rescaled_x_axis, min(rg0, rg1))
            er = _np_searchsorted(rescaled_x_axis, max(rg0, rg1))

            n_spec = len(spectrum)
            sl, el = max(0, sl), min(n_spec, el)
            sr, er = max(0, sr), min(n_spec, er)

            if ignore_brilouin_peaks:
                left_center_freq = float((lg0 + lg1) * 0.5)
                right_center_freq = float((rg0 + rg1) * 0.5)
                il = int(_np_argmin(_np_abs(rescaled_x_axis - left_center_freq))) if len(rescaled_x_axis) else None
                ir = int(_np_argmin(_np_abs(rescaled_x_axis - right_center_freq))) if len(rescaled_x_axis) else None
            else:
                il = ir = None
                if el - sl > 0:
                    seg_left = spectrum[sl:el]
                    pk, _ = find_peaks(seg_left, height=0)
                    il = int(pk[_np_argmax(peak_prominences(seg_left, pk)[0])] + sl) if len(pk) else int(_np_argmax(seg_left) + sl)
                if er - sr > 0:
                    seg_right = spectrum[sr:er]
                    pk, _ = find_peaks(seg_right, height=0)
                    ir = int(pk[_np_argmax(peak_prominences(seg_right, pk)[0])] + sr) if len(pk) else int(_np_argmax(seg_right) + sr)

            if il is not None and ir is not None:
                keep_lp.append(lp)
                keep_ranges.append(lr)
                keep_left_idx_ranges.append((sl, el))
                keep_right_idx_ranges.append((sr, er))
                bi.extend([il, ir])

        laser_peaks_indices = np.array(keep_lp, dtype=int)
        laser_peak_index_ranges = keep_ranges
        brillouin_peaks_indices = bi
        per_laser_left_ranges = keep_left_idx_ranges
        per_laser_right_ranges = keep_right_idx_ranges

        if len(brillouin_peaks_indices) != 2 * len(laser_peaks_indices):
            print("Mismatch in Brillouin peaks count with tuple mode (after filtering).")
            return None
        
        # In tuple mode with ignore_laser_for_fit: redo FSR fit using Brillouin peak midpoints
        if ignore_laser_for_fit and len(brillouin_peaks_indices) == 2 * len(laser_peaks_indices):
            n_pairs = len(laser_peaks_indices)
            midpoint_indices = np.array([
                (brillouin_peaks_indices[2*i] + brillouin_peaks_indices[2*i + 1]) / 2.0
                for i in range(n_pairs)
            ])
            expected_positions_refit = np.arange(n_pairs) * free_spectral_range
            res_refit = least_squares(_resid, x0=[a, b, c], args=(midpoint_indices, expected_positions_refit), bounds=(lb, ub))
            if res_refit.success:
                a, b, c = [float(v) for v in res_refit.x]
                rescaled_x_axis = a * x_axis**2 + b * x_axis + c  # Update rescaled axis

    left_center_bounds = []
    right_center_bounds = []
    total_points = len(rescaled_x_axis)

    for rng in per_laser_left_ranges:
        if rng and total_points:
            s_idx, e_idx = rng
            s_idx = int(max(0, min(total_points - 1, s_idx)))
            e_idx = int(max(0, min(total_points, e_idx)))
            if e_idx <= s_idx:
                e_idx = min(total_points, s_idx + 1)
            segment = rescaled_x_axis[s_idx:e_idx] if e_idx > s_idx else rescaled_x_axis[s_idx:s_idx+1]
            if segment.size == 0:
                segment = np.array([rescaled_x_axis[min(total_points - 1, s_idx)]])
            lo = float(np.nanmin(segment))
            hi = float(np.nanmax(segment))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo = float(rescaled_x_axis[min(total_points - 1, s_idx)])
                hi = float(rescaled_x_axis[min(total_points - 1, max(s_idx, e_idx - 1))])
            if hi < lo:
                lo, hi = hi, lo
            left_center_bounds.append((lo, hi))
        else:
            left_center_bounds.append((float('-inf'), float('inf')))

    for rng in per_laser_right_ranges:
        if rng and total_points:
            s_idx, e_idx = rng
            s_idx = int(max(0, min(total_points - 1, s_idx)))
            e_idx = int(max(0, min(total_points, e_idx)))
            if e_idx <= s_idx:
                e_idx = min(total_points, s_idx + 1)
            segment = rescaled_x_axis[s_idx:e_idx] if e_idx > s_idx else rescaled_x_axis[s_idx:s_idx+1]
            if segment.size == 0:
                segment = np.array([rescaled_x_axis[min(total_points - 1, s_idx)]])
            lo = float(np.nanmin(segment))
            hi = float(np.nanmax(segment))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo = float(rescaled_x_axis[min(total_points - 1, s_idx)])
                hi = float(rescaled_x_axis[min(total_points - 1, max(s_idx, e_idx - 1))])
            if hi < lo:
                lo, hi = hi, lo
            right_center_bounds.append((lo, hi))
        else:
            right_center_bounds.append((float('-inf'), float('inf')))

    laser_center_hard_bounds = []
    for idx_range in laser_peak_index_ranges:
        if not idx_range:
            laser_center_hard_bounds.append(None)
            continue
        s_idx, e_idx = idx_range
        total_points = len(rescaled_x_axis)
        if total_points == 0:
            laser_center_hard_bounds.append(None)
            continue
        s_idx = int(max(0, min(total_points - 1, s_idx)))
        e_idx = int(max(0, min(total_points, e_idx)))
        if e_idx <= s_idx:
            e_idx = min(total_points, s_idx + 1)
        segment = rescaled_x_axis[s_idx:e_idx] if e_idx > s_idx else rescaled_x_axis[s_idx:s_idx+1]
        if segment.size == 0:
            idx_clamp = max(0, min(total_points - 1, s_idx))
            segment = rescaled_x_axis[idx_clamp:idx_clamp+1]
        center_lo = float(np.nanmin(segment))
        center_hi = float(np.nanmax(segment))
        if not np.isfinite(center_lo) or not np.isfinite(center_hi):
            idx_lo = max(0, min(total_points - 1, s_idx))
            idx_hi = max(0, min(total_points - 1, e_idx - 1 if e_idx > 0 else 0))
            center_lo = float(rescaled_x_axis[idx_lo])
            center_hi = float(rescaled_x_axis[idx_hi])
        if center_hi < center_lo:
            center_lo, center_hi = center_hi, center_lo
        laser_center_hard_bounds.append((center_lo, center_hi))
    # ---------------- Lorentzian fit controls ----------------
    perform_lorentzian_fitting = (isinstance(fit_lorentzians, (int, float)) and float(fit_lorentzians) > 0.0)
    fit_half_window_GHz = float(fit_lorentzians) if perform_lorentzian_fitting else 0.0
    fit_laser_lorentzians = perform_lorentzian_fitting and not manual_laser_peaks
    if match_brilouin_parameters and not perform_lorentzian_fitting:
        match_brilouin_parameters = False

    # gamma (FWHM/2) bounds (GHz)
    if fwhm_bounds_GHz is not None:
        fmin, fmax = fwhm_bounds_GHz
        g_lo = max(1e-6, 0.5 * float(fmin)) if (fmin is not None) else 1e-6
        g_hi = (0.5 * float(fmax)) if (fmax is not None) else np.inf
        if not _np_isfinite(g_hi) or g_hi <= g_lo:
            g_hi = np.inf
    else:
        g_lo, g_hi = 1e-6, np.inf

    # Pre-compute values used repeatedly in _fit_one_by_center
    _spectrum_len = len(spectrum)
    _axis_len = len(rescaled_x_axis)
    # Cache the median gradient for fallback (computed once, not per peak)
    _cached_median_gradient = float(np.median(np.gradient(rescaled_x_axis))) if _axis_len > 1 else 1.0

    # helper: fit one Lorentzian around a nominal center using GHz window & constraints
    def _fit_one_by_center(nominal_idx, *, hard_center_bounds=None, return_window=False):
        xc_nom = float(rescaled_x_axis[nominal_idx])
        lo_val = xc_nom - fit_half_window_GHz
        hi_val = xc_nom + fit_half_window_GHz
        w0 = _np_searchsorted(rescaled_x_axis, lo_val, side='left')
        w1 = _np_searchsorted(rescaled_x_axis, hi_val, side='right')
        w0 = max(0, w0); w1 = min(_spectrum_len, max(w1, w0 + 3))  # >= 3 pts

        xw = rescaled_x_axis[w0:w1].copy()
        yw = spectrum[w0:w1].copy()
        if len(xw) < 3:
            w0 = max(0, nominal_idx - 3)
            w1 = min(_spectrum_len, nominal_idx + 4)
            xw = rescaled_x_axis[w0:w1].copy()
            yw = spectrum[w0:w1].copy()

        A0 = float(max(1e-12, _np_max(yw)))
        x0 = float(xw[_np_argmax(yw)])
        try:
            w_samples = float(peak_widths(spectrum, [nominal_idx], rel_height=0.5)[0][0])
        except Exception:
            w_samples = max(1.0, len(xw) / 5.0)
        seg_hi = min(_axis_len - 1, w0 + max(2, len(xw)))
        local_dx = np.median(np.diff(rescaled_x_axis[max(0, w0):seg_hi]))
        local_dx = float(local_dx) if (_np_isfinite(local_dx) and local_dx != 0) else _cached_median_gradient
        fwhm0 = max(1e-6, w_samples * abs(local_dx))
        g0 = float(_np_clip(0.5 * fwhm0, g_lo, g_hi if _np_isfinite(g_hi) else 0.5 * fwhm0))
        hard_lo = hard_hi = None
        if hard_center_bounds is not None:
            hard_lo, hard_hi = hard_center_bounds
            if hard_lo is not None and hard_hi is not None and hard_lo > hard_hi:
                hard_lo, hard_hi = hard_hi, hard_lo

        # center bounds
        if center_slack_GHz is not None and center_slack_GHz > 0:
            c_lo_val = xc_nom - float(center_slack_GHz)
            c_hi_val = xc_nom + float(center_slack_GHz)
        else:
            c_lo_val, c_hi_val = lo_val, hi_val
        if hard_lo is not None:
            c_lo_val = max(c_lo_val, hard_lo)
        if hard_hi is not None:
            c_hi_val = min(c_hi_val, hard_hi)
        if c_lo_val > c_hi_val:
            mid_val = 0.5 * (c_lo_val + c_hi_val)
            c_lo_val = c_hi_val = mid_val
        xw_min, xw_max = float(xw[0]), float(xw[-1])  # xw is already sorted
        c_lo = max(xw_min, c_lo_val)
        c_hi = min(xw_max, c_hi_val)
        if c_hi < c_lo:
            target = float(_np_clip(x0, min(c_lo_val, c_hi_val), max(c_lo_val, c_hi_val)))
            c_lo = c_hi = target
        p0 = [A0, x0, g0]
        bounds = ([0.0, c_lo, g_lo], [np.inf, c_hi, g_hi])

        sigma = None
        if poisson_weighting:
            sigma = _np_sqrt(_np_clip(yw, 1e-6, None))

        try:
            if poisson_weighting and sigma is not None:
                popt, _ = curve_fit(lorentzian, xw, yw, p0=p0, bounds=bounds, sigma=sigma,
                                    absolute_sigma=False, maxfev=20000)
            else:
                popt, _ = curve_fit(lorentzian, xw, yw, p0=p0, bounds=bounds, maxfev=20000)
            A, xc, g = float(popt[0]), float(popt[1]), float(popt[2])
        except Exception:
            A, xc, g = p0

        rmse = np.nan
        if len(xw) and len(yw):
            try:
                model_vals = lorentzian(xw, A, xc, g)
                diff = model_vals - yw
                rmse = float(_np_sqrt(np.mean(diff**2))) if diff.size else np.nan
            except Exception:
                pass

        if return_window:
            return A, xc, g, rmse, xw, yw, sigma
        return A, xc, g, rmse

    def _fit_group_shared(windows, *, side, poisson_weighting=False):
        if not windows:
            return None
        n = len(windows)
        
        # Pre-allocate arrays instead of list appends
        shift_inits = np.empty(n, dtype=float)
        gamma_inits = np.empty(n, dtype=float)
        amp_inits = np.empty(n, dtype=float)
        
        # Pre-extract window data for faster access in residual function
        is_left = (side == 'left')
        laser_centers = np.empty(n, dtype=float)
        window_lengths = np.empty(n, dtype=int)
        
        for i, win in enumerate(windows):
            laser_centers[i] = win['laser_center']
            window_lengths[i] = len(win['x'])
            
            if is_left:
                shift_val = float(win['laser_center'] - win['center_init'])
            else:
                shift_val = float(win['center_init'] - win['laser_center'])
            shift_val = abs(shift_val)
            shift_inits[i] = shift_val if (_np_isfinite(shift_val) and shift_val > 0) else 1.0
            
            gamma_val = float(win['gamma_init'])
            gamma_inits[i] = gamma_val if (_np_isfinite(gamma_val) and gamma_val > 0) else 1.0
            
            amp_val = float(win['amplitude_init'])
            amp_inits[i] = amp_val if (_np_isfinite(amp_val) and amp_val >= 0) else 1.0

        shift0 = max(1e-6, float(np.mean(shift_inits)))
        gamma0 = float(_np_clip(np.mean(gamma_inits), g_lo, g_hi if _np_isfinite(g_hi) else np.mean(gamma_inits)))
        params0 = np.concatenate([[shift0, gamma0], amp_inits])

        shift_lower_bound = 1e-6
        shift_upper_bound = np.inf
        shift_lower_candidates = []
        shift_upper_candidates = []
        bounds_available = True
        for i, win in enumerate(windows):
            center_bounds = win.get('center_bounds')
            if center_bounds is None:
                bounds_available = False
                break
            c_lo, c_hi = center_bounds
            if not (_np_isfinite(c_lo) and _np_isfinite(c_hi)):
                bounds_available = False
                break
            lc = laser_centers[i]
            if is_left:
                lower_val = max(1e-6, lc - c_hi)
                upper_val = max(1e-6, lc - c_lo)
            else:
                lower_val = max(1e-6, c_lo - lc)
                upper_val = max(1e-6, c_hi - lc)
            shift_lower_candidates.append(lower_val)
            shift_upper_candidates.append(upper_val)
        if bounds_available and shift_lower_candidates and shift_upper_candidates:
            shift_lower_bound = max([1e-6] + shift_lower_candidates)
            shift_upper_bound = min([np.inf] + shift_upper_candidates)
            if shift_lower_bound > shift_upper_bound:
                mid = max(1e-6, float(np.mean(shift_inits)))
                shift_lower_bound = shift_upper_bound = mid

        params0[0] = float(_np_clip(params0[0], shift_lower_bound, shift_upper_bound))

        lower_bounds = np.array([shift_lower_bound, g_lo] + [0.0] * n, dtype=float)
        upper_gamma = g_hi if _np_isfinite(g_hi) else np.inf
        upper_bounds = np.array([shift_upper_bound, upper_gamma] + [np.inf] * n, dtype=float)

        # Pre-allocate residual output array and pre-compute sigma if needed
        total_points = int(np.sum(window_lengths))
        residual_out = np.empty(total_points, dtype=float)
        
        # Pre-compute sigma arrays for Poisson weighting
        if poisson_weighting:
            sigma_arrays = []
            for win in windows:
                sig = win.get('sigma')
                if sig is None:
                    sig = _np_sqrt(_np_clip(win['y'], 1e-6, None))
                sigma_arrays.append(_np_clip(sig, 1e-6, None))
        else:
            sigma_arrays = None

        def _residuals(params):
            shift = params[0]
            gamma = params[1]
            amps = params[2:]
            offset = 0
            for i, (amp, win) in enumerate(zip(amps, windows)):
                length = window_lengths[i]
                if is_left:
                    center_val = laser_centers[i] - shift
                else:
                    center_val = laser_centers[i] + shift
                model = lorentzian(win['x'], amp, center_val, gamma)
                resid = model - win['y']
                if poisson_weighting:
                    resid = resid / sigma_arrays[i]
                residual_out[offset:offset + length] = resid
                offset += length
            return residual_out

        try:
            res = least_squares(_residuals, params0, bounds=(lower_bounds, upper_bounds), max_nfev=20000)
            if not res.success:
                return None
            opt = res.x
        except Exception:
            return None

        shift_opt = float(opt[0])
        gamma_opt = float(opt[1])
        amps_opt = [float(v) for v in opt[2:]]
        
        # Vectorized center computation
        if is_left:
            centers_opt = (laser_centers - shift_opt).tolist()
        else:
            centers_opt = (laser_centers + shift_opt).tolist()
        
        # Compute RMSE list
        rmse_list = []
        for amp_val, center_val, win in zip(amps_opt, centers_opt, windows):
            rmse_val = np.nan
            try:
                x_vals = win.get('x')
                y_vals = win.get('y')
                if x_vals is not None and y_vals is not None and len(x_vals) and len(y_vals):
                    model_vals = lorentzian(x_vals, amp_val, center_val, gamma_opt)
                    diff = model_vals - y_vals
                    if diff.size:
                        rmse_val = float(_np_sqrt(np.mean(diff**2)))
            except Exception:
                pass
            rmse_list.append(rmse_val)

        return {
            'shift': shift_opt,
            'gamma': gamma_opt,
            'amplitudes': amps_opt,
            'centers': centers_opt,
            'rmse_list': rmse_list,
        }

    # ---------------- main per-peak loop ----------------
    shifts, left_shifts, right_shifts = [], [], []
    fwhms, fwhms_left_list, fwhms_right_list = [], [], []
    amplitude_laser_list, center_laser_list, gamma_laser_list = [], [], []
    amplitude_left_list,  center_left_list,  gamma_left_list  = [], [], []
    amplitude_right_list, center_right_list, gamma_right_list = [], [], []
    laser_fit_rmse, left_fit_rmse, right_fit_rmse = [], [], []
    left_windows, right_windows = [], []

    for i in range(len(laser_peaks_indices)):
        lp = int(laser_peaks_indices[i])
        li = int(brillouin_peaks_indices[2*i])
        ri = int(brillouin_peaks_indices[2*i + 1])

        if perform_lorentzian_fitting:
            if fit_laser_lorentzians:
                A_l, xc_l, g_l, rmse_l = _fit_one_by_center(
                    lp,
                    hard_center_bounds=laser_center_hard_bounds[i] if i < len(laser_center_hard_bounds) else None,
                )
            else:
                xc_l = float(rescaled_x_axis[lp])
                A_l = float(spectrum[lp])
                g_l = 1.0
                rmse_l = np.nan
            amplitude_laser_list.append(A_l); center_laser_list.append(xc_l); gamma_laser_list.append(g_l)
            laser_fit_rmse.append(rmse_l)

            current_laser_center = center_laser_list[-1]

            if ignore_brilouin_peaks:
                bc_left = float(rescaled_x_axis[li])
                bc_right = float(rescaled_x_axis[ri])
                amplitude_left_list.append(1.0); center_left_list.append(bc_left); gamma_left_list.append(1.0)
                amplitude_right_list.append(1.0); center_right_list.append(bc_right); gamma_right_list.append(1.0)
                left_fit_rmse.append(np.nan)
                right_fit_rmse.append(np.nan)
            else:
                left_bounds_current = left_center_bounds[i] if i < len(left_center_bounds) else (float('-inf'), float('inf'))
                if match_brilouin_parameters:
                    A_L, xc_L, g_L, rmse_L, xw_L, yw_L, sig_L = _fit_one_by_center(li, return_window=True)
                else:
                    A_L, xc_L, g_L, rmse_L = _fit_one_by_center(li)
                amplitude_left_list.append(A_L); center_left_list.append(xc_L); gamma_left_list.append(g_L)
                left_fit_rmse.append(rmse_L)
                if match_brilouin_parameters:
                    left_windows.append({
                        'x': xw_L,
                        'y': yw_L,
                        'laser_center': current_laser_center,
                        'center_init': xc_L,
                        'gamma_init': g_L,
                        'amplitude_init': A_L,
                        'sigma': sig_L,
                        'center_bounds': left_bounds_current,
                    })

                right_bounds_current = right_center_bounds[i] if i < len(right_center_bounds) else (float('-inf'), float('inf'))
                if match_brilouin_parameters:
                    A_R, xc_R, g_R, rmse_R, xw_R, yw_R, sig_R = _fit_one_by_center(ri, return_window=True)
                else:
                    A_R, xc_R, g_R, rmse_R = _fit_one_by_center(ri)
                amplitude_right_list.append(A_R); center_right_list.append(xc_R); gamma_right_list.append(g_R)
                right_fit_rmse.append(rmse_R)
                if match_brilouin_parameters:
                    right_windows.append({
                        'x': xw_R,
                        'y': yw_R,
                        'laser_center': current_laser_center,
                        'center_init': xc_R,
                        'gamma_init': g_R,
                        'amplitude_init': A_R,
                        'sigma': sig_R,
                        'center_bounds': right_bounds_current,
                    })

            brillouin_based_shift = (center_right_list[-1] - center_left_list[-1]) / 2.0
            if brillouin_shift_average:
                # Shift determined purely from Brillouin peak positions
                left_shifts.append(brillouin_based_shift)
                right_shifts.append(brillouin_based_shift)
            else:
                # Shift measured from laser peak position
                left_shifts.append(center_laser_list[-1] - center_left_list[-1])
                right_shifts.append(center_right_list[-1] - center_laser_list[-1])
            shifts.append(brillouin_based_shift)
            fwhms_left = 2.0 * gamma_left_list[-1]
            fwhms_right = 2.0 * gamma_right_list[-1]
            fwhms_left_list.append(fwhms_left)
            fwhms_right_list.append(fwhms_right)
            fwhms.append(0.5 * (fwhms_left + fwhms_right))
        else:
            left_center = float(rescaled_x_axis[li])
            right_center = float(rescaled_x_axis[ri])
            laser_center = float(rescaled_x_axis[lp])
            bbs = (right_center - left_center) / 2.0
            if brillouin_shift_average:
                left_shifts.append(bbs)
                right_shifts.append(bbs)
            else:
                left_shifts.append(laser_center - left_center)
                right_shifts.append(right_center - laser_center)
            shifts.append(bbs)

            if ignore_brilouin_peaks:
                fwhms_left_list.append(1.0)
                fwhms_right_list.append(1.0)
                fwhms.append(1.0)
            else:
                local_dx = np.gradient(rescaled_x_axis)
                wL = peak_widths(spectrum, [li], rel_height=0.5)[0][0]
                wR = peak_widths(spectrum, [ri], rel_height=0.5)[0][0]
                fL = float(wL * abs(local_dx[li]))
                fR = float(wR * abs(local_dx[ri]))
                fwhms_left_list.append(fL); fwhms_right_list.append(fR); fwhms.append(0.5 * (fL + fR))
            laser_fit_rmse.append(np.nan)
            left_fit_rmse.append(np.nan)
            right_fit_rmse.append(np.nan)

    if match_brilouin_parameters and perform_lorentzian_fitting and center_laser_list:
        left_result = _fit_group_shared(left_windows, side='left', poisson_weighting=poisson_weighting) if left_windows else None
        right_result = _fit_group_shared(right_windows, side='right', poisson_weighting=poisson_weighting) if right_windows else None
        n_peaks = len(center_laser_list)

        if left_result and len(left_result['centers']) == n_peaks:
            amplitude_left_list = left_result['amplitudes']
            center_left_list = left_result['centers']
            gamma_left_list = [left_result['gamma']] * n_peaks
            left_shift_val = left_result['shift']
            left_shifts = [left_shift_val] * n_peaks
            fwhm_left_val = 2.0 * left_result['gamma']
            fwhms_left_list = [fwhm_left_val] * n_peaks
            if 'rmse_list' in left_result and len(left_result['rmse_list']) == n_peaks:
                left_fit_rmse = list(left_result['rmse_list'])

        if right_result and len(right_result['centers']) == n_peaks:
            amplitude_right_list = right_result['amplitudes']
            center_right_list = right_result['centers']
            gamma_right_list = [right_result['gamma']] * n_peaks
            right_shift_val = right_result['shift']
            right_shifts = [right_shift_val] * n_peaks
            fwhm_right_val = 2.0 * right_result['gamma']
            fwhms_right_list = [fwhm_right_val] * n_peaks
            if 'rmse_list' in right_result and len(right_result['rmse_list']) == n_peaks:
                right_fit_rmse = list(right_result['rmse_list'])

        if left_result and right_result and len(left_result['centers']) == n_peaks and len(right_result['centers']) == n_peaks:
            shifts = [0.5 * (center_right_list[i] - center_left_list[i]) for i in range(n_peaks)]
            fwhms = [0.5 * (fwhms_left_list[i] + fwhms_right_list[i]) for i in range(n_peaks)]
        elif left_result and len(left_result['centers']) == n_peaks:
            shifts = left_shifts[:]
            fwhms = fwhms_left_list[:]
        elif right_result and len(right_result['centers']) == n_peaks:
            shifts = right_shifts[:]
            fwhms = fwhms_right_list[:]

    # aggregates
    median_shift = np.median(shifts) if shifts else np.nan
    median_fwhm  = np.median(fwhms)  if fwhms  else np.nan

    # optional reindex if fitted centers were collected (keep triplets aligned)
    if perform_lorentzian_fitting and len(center_laser_list):
        order = np.argsort(center_laser_list)
        def _re(L): return list(np.array(L)[order])
        center_laser_list    = _re(center_laser_list)
        amplitude_laser_list = _re(amplitude_laser_list)
        gamma_laser_list     = _re(gamma_laser_list)
        center_left_list     = _re(center_left_list)
        amplitude_left_list  = _re(amplitude_left_list)
        gamma_left_list      = _re(gamma_left_list)
        center_right_list    = _re(center_right_list)
        amplitude_right_list = _re(amplitude_right_list)
        gamma_right_list     = _re(gamma_right_list)
        laser_fit_rmse       = _re(laser_fit_rmse)
        left_fit_rmse        = _re(left_fit_rmse)
        right_fit_rmse       = _re(right_fit_rmse)
        # reorder derived metrics to match
        shifts        = _re(shifts)
        left_shifts   = _re(left_shifts)
        right_shifts  = _re(right_shifts)
        fwhms         = _re(fwhms)
        fwhms_left_list  = _re(fwhms_left_list)
        fwhms_right_list = _re(fwhms_right_list)
        # remap indices from fitted centers (left/right interleaved)
        laser_peaks_indices = [int(np.argmin(np.abs(rescaled_x_axis - c))) for c in center_laser_list]
        new_bpi = []
        for i in range(len(center_laser_list)):
            new_bpi += [
                int(np.argmin(np.abs(rescaled_x_axis - center_left_list[i]))),
                int(np.argmin(np.abs(rescaled_x_axis - center_right_list[i]))),
            ]
        brillouin_peaks_indices = new_bpi

    # dataframe
    def _safe_center(center_list, list_idx, local_idx):
        if center_list is not None and list_idx is not None:
            try:
                if list_idx < len(center_list):
                    return float(center_list[list_idx])
            except Exception:
                pass
        if 0 <= local_idx < len(rescaled_x_axis):
            return float(rescaled_x_axis[local_idx])
        return np.nan

    def _safe_amplitude(amplitude_list, list_idx, local_idx):
        if amplitude_list is not None and list_idx is not None:
            try:
                if list_idx < len(amplitude_list):
                    return float(amplitude_list[list_idx])
            except Exception:
                pass
        if 0 <= local_idx < len(spectrum):
            return float(spectrum[local_idx])
        return np.nan

    def _safe_gamma(gamma_list, list_idx, fwhm_val):
        if gamma_list is not None and list_idx is not None:
            try:
                if list_idx < len(gamma_list):
                    return float(gamma_list[list_idx])
            except Exception:
                pass
        if fwhm_val is not None and np.isfinite(fwhm_val):
            return float(fwhm_val) / 2.0
        return np.nan

    def _peak_value(local_idx):
        if 0 <= local_idx < len(spectrum):
            return float(spectrum[local_idx])
        return np.nan

    def _safe_rmse(values, idx):
        if values is None:
            return np.nan
        try:
            val = values[idx]
        except Exception:
            return np.nan
        try:
            return float(val)
        except Exception:
            return np.nan

    rows = []
    for i in range(len(laser_peaks_indices)):
        l_idx = int(laser_peaks_indices[i])
        bl_idx = int(brillouin_peaks_indices[2*i])
        br_idx = int(brillouin_peaks_indices[2*i+1])

        rows.append({
            'Peak': _peak_value(l_idx),
            'Peak Index': l_idx + s0,
            'Peak Type': 'Laser',
            'Shift': float(shifts[i]),
            'FWHM': float(fwhms[i]),
            'Center (GHz)': _safe_center(center_laser_list, i, l_idx),
            'Amplitude': _safe_amplitude(amplitude_laser_list, i, l_idx),
            'Gamma (GHz)': _safe_gamma(gamma_laser_list, i, fwhms[i]),
            'Fit RMSE': _safe_rmse(laser_fit_rmse, i),
        })
        rows.append({
            'Peak': _peak_value(bl_idx),
            'Peak Index': bl_idx + s0,
            'Peak Type': 'Brillouin Left',
            'Shift': float(left_shifts[i]),
            'FWHM': float(fwhms_left_list[i]),
            'Center (GHz)': _safe_center(center_left_list, i, bl_idx),
            'Amplitude': _safe_amplitude(amplitude_left_list, i, bl_idx),
            'Gamma (GHz)': _safe_gamma(gamma_left_list, i, fwhms_left_list[i]),
            'Fit RMSE': _safe_rmse(left_fit_rmse, i),
        })
        rows.append({
            'Peak': _peak_value(br_idx),
            'Peak Index': br_idx + s0,
            'Peak Type': 'Brillouin Right',
            'Shift': float(right_shifts[i]),
            'FWHM': float(fwhms_right_list[i]),
            'Center (GHz)': _safe_center(center_right_list, i, br_idx),
            'Amplitude': _safe_amplitude(amplitude_right_list, i, br_idx),
            'Gamma (GHz)': _safe_gamma(gamma_right_list, i, fwhms_right_list[i]),
            'Fit RMSE': _safe_rmse(right_fit_rmse, i),
        })
    df = pd.DataFrame(rows)

    result = {
        'df': df,
        'spectrum': spectrum,
        'laser_peaks_indices': [i + s0 for i in laser_peaks_indices],
        'brillouin_peaks_indices': [i + s0 for i in brillouin_peaks_indices],
        'fit_params': (a, b, c),
        'rescaled_x_axis': rescaled_x_axis,
        'shifts': shifts,
        'fwhms': fwhms,
        'fwhms_left_list': fwhms_left_list,
        'fwhms_right_list': fwhms_right_list,
        'left_shifts': left_shifts,
        'right_shifts': right_shifts,
        'median_shift': median_shift,
        'median_fwhm':  median_fwhm,
        'a': a, 'b': b, 'c': c,
        'amplitude_laser_list': amplitude_laser_list,
        'center_laser_list': center_laser_list,
        'gamma_laser_list': gamma_laser_list,
        'amplitude_left_list': amplitude_left_list,
        'center_left_list': center_left_list,
        'gamma_left_list': gamma_left_list,
        'amplitude_right_list': amplitude_right_list,
        'center_right_list': center_right_list,
        'gamma_right_list': gamma_right_list,
        'laser_fit_rmse': laser_fit_rmse,
        'left_fit_rmse': left_fit_rmse,
        'right_fit_rmse': right_fit_rmse,
        'manual_laser_positions': manual_laser_peaks,
        'baseline_spectra': baseline_spectra,
        'poisson_weighting': poisson_weighting,
        'match_brilouin_parameters': match_brilouin_parameters,
        'ignore_brilouin_peaks': ignore_brilouin_peaks,
        'ignore_laser_peaks': ignore_laser_peaks,
        'brillouin_shift_average': brillouin_shift_average,
        'ignore_laser_for_fit': ignore_laser_for_fit,
        'fit_laser_lorentzians': fit_laser_lorentzians,
        'start_index': s0,
    }

    if make_plot:
        plot_brillouin_spectrum(
            result,
            x_coord, y_coord, z_coord,
            make_plot, debug_plot,
            perform_lorentzians_fitting=perform_lorentzian_fitting,
            fit_lorentzians=fit_lorentzians,
            laser_peak_ranges=local_laser_peak_ranges,
            brillouin_peak_ranges=local_brillouin_peak_ranges,
            save_plot=save_plot
        )

    return result


def _prepare_spectra_matrix(brillouin_spectra, cut_range=None, baseline_spectra=False):
    if brillouin_spectra is None:
        return None, 0
    try:
        flat_spectra = brillouin_spectra.ravel()
    except AttributeError:
        try:
            flat_spectra = np.asarray(brillouin_spectra, dtype=object).ravel()
        except Exception:
            flat_spectra = []

    processed = []
    if cut_range:
        s0 = max(0, int(cut_range[0]))
        s1 = int(cut_range[1])
    else:
        s0, s1 = 0, None

    for spectrum in flat_spectra:
        if spectrum is None or isinstance(spectrum, (int, float)):
            continue
        try:
            arr = np.asarray(spectrum, dtype=float)
        except Exception:
            continue
        if arr.size == 0:
            continue
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr[arr < 0] = 0.0
        if s1 is not None:
            local_s1 = min(arr.size, s1)
            if local_s1 <= s0:
                continue
            arr = arr[s0:local_s1]
        if arr.size == 0:
            continue
        if baseline_spectra:
            arr = arr - np.min(arr)
        processed.append(arr)

    if not processed:
        return None, s0 if cut_range else 0

    try:
        matrix = np.vstack(processed)
    except ValueError as exc:
        print(f"Error stacking spectra for PCA: {exc}")
        return None, s0 if cut_range else 0

    return matrix, (s0 if cut_range else 0)


def perform_pca_analysis(
    brillouin_spectra,
    n_components=3,
    pca_params=None,
    peak_finding_params=None,
    peak_windows=None,
    spectrum_cut_range=None,
    baseline_spectra=False,
    figsize=(12, None),
    show_plot=True,
):
    """
    Perform PCA analysis on Brillouin spectra and optionally plot the results.
    
    Parameters
    ----------
    brillouin_spectra : array-like
        3D array of Brillouin spectra (x, y, z, spectrum).
    n_components : int, default=3
        Number of PCA components to compute.
    pca_params : dict, optional
        Additional parameters passed to sklearn.decomposition.PCA.
    peak_finding_params : dict, optional
        Parameters passed to scipy.signal.find_peaks for global peak detection.
        Used when peak_windows is None.
    peak_windows : list of tuples, optional
        List of (start, end) index ranges for window-based peak finding.
        Each window will detect exactly one peak (the maximum within that window).
        When provided, this takes precedence over peak_finding_params.
        Example: [(400, 500), (600, 700), (900, 1000), (1100, 1200)]
    spectrum_cut_range : tuple, optional
        (start, end) indices to cut spectra before analysis.
    baseline_spectra : bool, default=False
        Whether to baseline (subtract minimum) each spectrum.
    figsize : tuple, default=(12, None)
        Figure size. If second element is None, height is calculated as 4 * n_components.
    show_plot : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'pca_model': fitted PCA model
        - 'components': PCA components array
        - 'explained_variance_ratio': variance explained by each component
        - 'peaks_per_component': list of detected peak indices for each component
        - 'x_axis': x-axis values (indices) for plotting
        - 'x_offset': offset applied to indices
    """
    if pca_params is None:
        pca_params = {}
    if peak_finding_params is None:
        peak_finding_params = {}

    matrix, x_offset = _prepare_spectra_matrix(
        brillouin_spectra, 
        cut_range=spectrum_cut_range, 
        baseline_spectra=baseline_spectra
    )
    if matrix is None:
        print("No valid spectra found for PCA.")
        return None

    pca = PCA(n_components=n_components, **pca_params)
    try:
        pca.fit(matrix)
    except Exception as e:
        print(f"PCA failed: {e}")
        return None

    components = pca.components_
    x_axis = np.arange(matrix.shape[1]) + x_offset
    
    # Find peaks for each component
    peaks_per_component = []
    for comp in components:
        if peak_windows is not None:
            # Window-based peak finding: one peak per window
            peaks = _find_peaks_in_windows(comp, peak_windows, x_offset, **peak_finding_params)
        else:
            # Global peak finding using find_peaks
            peaks, _ = find_peaks(comp, **peak_finding_params)
        peaks_per_component.append(peaks)

    # Plotting
    if show_plot:
        fig_height = figsize[1] if figsize[1] is not None else 4 * n_components
        fig, axes = plt.subplots(n_components, 1, figsize=(figsize[0], fig_height), sharex=True)
        if n_components == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            comp = components[i]
            ax.plot(x_axis, comp, label=f"Component {i+1}")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right')

            peaks = peaks_per_component[i]
            if len(peaks) > 0:
                ax.plot(x_axis[peaks], comp[peaks], "x", color='red', markersize=10)
                for p in peaks:
                    ax.text(x_axis[p], comp[p], f"{int(x_axis[p])}", 
                           verticalalignment='bottom', color='red', fontsize=9)
            
            # Show windows if provided
            if peak_windows is not None:
                for start, end in peak_windows:
                    # Adjust window indices relative to cut range
                    local_start = max(0, start - x_offset)
                    local_end = min(len(comp), end - x_offset)
                    if local_end > local_start:
                        ax.axvspan(start, end, color='yellow', alpha=0.15)

        axes[-1].set_xlabel("Index")
        plt.suptitle(f"PCA Analysis ({n_components} components)")
        plt.tight_layout()
        plt.show()

    peaks_per_component_with_offset = [
        np.asarray(peaks) + x_offset if isinstance(peaks, (np.ndarray, list)) else peaks
        for peaks in peaks_per_component
    ]
    return {
        'pca_model': pca,
        'components': components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'peaks_per_component': peaks_per_component,
        'peaks_per_component_with_offset': peaks_per_component_with_offset,
        'x_axis': x_axis,
        'x_offset': x_offset,
    }


def _find_peaks_in_windows(signal, windows, offset=0, **peak_finding_params):
    """
    Find one peak per window in the signal using scipy.signal.find_peaks.
    
    Parameters
    ----------
    signal : array-like
        1D signal array.
    windows : list of tuples
        List of (start, end) index ranges (in original coordinates before cut).
    offset : int
        Offset to convert from original indices to local signal indices.
    **peak_finding_params
        Additional parameters passed to scipy.signal.find_peaks.
    
    Returns
    -------
    list
        List of peak indices (in local signal coordinates).
    """
    peaks = []
    signal = np.asarray(signal)
    n = len(signal)
    
    for start, end in windows:
        # Convert to local indices
        local_start = max(0, int(start - offset))
        local_end = min(n, int(end - offset))
        
        if local_end <= local_start:
            continue
        
        # Extract window signal
        window_signal = signal[local_start:local_end]
        
        # Find peaks within this window using scipy
        window_peaks, _ = find_peaks(window_signal, **peak_finding_params)
        
        if len(window_peaks) == 0:
            # If no peaks found, fall back to maximum
            local_max_idx = np.argmax(window_signal)
            peak_idx = local_start + local_max_idx
        else:
            # Select the peak with the highest amplitude
            peak_heights = window_signal[window_peaks]
            tallest_peak_local_idx = window_peaks[np.argmax(peak_heights)]
            peak_idx = local_start + tallest_peak_local_idx
        
        peaks.append(peak_idx)
    
    return peaks


def pca_spectra_filter(
    brillouin_spectra,
    n_components=3,
    spectrum_cut_range=None,
    baseline_spectra=False,
    pca_params=None,
):
    """Fit a PCA model to spectra and return payload for the pca_filter stage."""
    if pca_params is None:
        pca_params = {}
    matrix, _ = _prepare_spectra_matrix(
        brillouin_spectra,
        cut_range=spectrum_cut_range,
        baseline_spectra=baseline_spectra,
    )
    if matrix is None:
        print("No valid spectra found for PCA filtering.")
        return None

    pca = PCA(n_components=n_components, **pca_params)
    try:
        pca.fit(matrix)
    except Exception as exc:
        print(f"PCA filter training failed: {exc}")
        return None

    return {
        'pca_model': pca,
        'n_components': getattr(pca, 'n_components_', n_components),
        'input_length': matrix.shape[1],
        'cut_range': spectrum_cut_range,
        'baseline_spectra': baseline_spectra,
        'pca_params': pca_params,
    }


# ---------- plotting ----------
def plot_brillouin_spectrum(
    result_dict,
    x_coord,
    y_coord,
    z_coord,
    figsize,
    debug_plot,
    perform_lorentzians_fitting,
    fit_lorentzians,
    laser_peak_ranges,
    brillouin_peak_ranges,
    save_plot=False
):
    rescaled_x_axis = result_dict['rescaled_x_axis']
    spectrum = result_dict['spectrum']
    s0 = result_dict.get('start_index', 0)
    laser_peaks_indices = [i - s0 for i in result_dict['laser_peaks_indices']]
    brillouin_peaks_indices = [i - s0 for i in result_dict['brillouin_peaks_indices']]
    shifts = result_dict['shifts']
    fwhms = result_dict['fwhms']
    fwhms_left_list = result_dict['fwhms_left_list']
    fwhms_right_list = result_dict['fwhms_right_list']
    median_shift = result_dict['median_shift']
    median_fwhm = result_dict['median_fwhm']
    a = result_dict['a']; b = result_dict['b']; c = result_dict['c']
    amplitude_laser_list = result_dict['amplitude_laser_list']
    center_laser_list = result_dict['center_laser_list']
    gamma_laser_list = result_dict['gamma_laser_list']
    amplitude_left_list = result_dict['amplitude_left_list']
    center_left_list = result_dict['center_left_list']
    gamma_left_list = result_dict['gamma_left_list']
    amplitude_right_list = result_dict['amplitude_right_list']
    center_right_list = result_dict['center_right_list']
    gamma_right_list = result_dict['gamma_right_list']
    manual_laser_mode = bool(result_dict.get('manual_laser_positions', False))
    ignore_brilouin_peaks = bool(result_dict.get('ignore_brilouin_peaks', False))
    fit_laser_lorentzians = bool(result_dict.get('fit_laser_lorentzians', perform_lorentzians_fitting))

    x, y, z = x_coord, y_coord, z_coord

    fig = plt.figure(figsize=figsize)

    # Trace
    if debug_plot:
        plt.plot(rescaled_x_axis, spectrum, label=f"Spectrum at (x={x}, y={y}, z={z})", color='blue')
    else:
        plt.plot(rescaled_x_axis, spectrum, label="Collected Signal", color='blue')

    # Fitted Lorentzians
    if perform_lorentzians_fitting:
        plot_fitted_lorentzians(
            amplitude_laser_list, center_laser_list, gamma_laser_list,
            amplitude_left_list, center_left_list, gamma_left_list,
            amplitude_right_list, center_right_list, gamma_right_list,
            plot_laser=fit_laser_lorentzians,
            plot_brilouin=not ignore_brilouin_peaks,
        )

    # Peak markers
    highlight_peaks(
        rescaled_x_axis, spectrum,
        laser_peaks_indices, brillouin_peaks_indices,
        center_laser_list, center_left_list, center_right_list,
        amplitude_laser_list, amplitude_left_list, amplitude_right_list,
        perform_lorentzians_fitting,
        ignore_brilouin_peaks=ignore_brilouin_peaks,
    )

    # Annotate shifts & FWHM
    annotate_shifts_fwhms(
        rescaled_x_axis, spectrum,
        laser_peaks_indices, brillouin_peaks_indices,
        shifts, fwhms_left_list, fwhms_right_list,
        center_left_list, amplitude_left_list,
        center_right_list, amplitude_right_list,
        perform_lorentzians_fitting,
        center_laser_list, amplitude_laser_list,
        ignore_brilouin_peaks=ignore_brilouin_peaks,
    )

    # Distances
    annotate_distances(
        rescaled_x_axis, spectrum,
        laser_peaks_indices, brillouin_peaks_indices,
        center_laser_list, center_left_list, center_right_list,
        perform_lorentzians_fitting
    )

    # Vertical lines
    if perform_lorentzians_fitting:
        laser_positions = center_laser_list
        left_brillouin_positions = center_left_list
        right_brillouin_positions = center_right_list
    else:
        laser_positions = [rescaled_x_axis[idx] for idx in laser_peaks_indices]
        left_brillouin_positions = [rescaled_x_axis[idx] for idx in brillouin_peaks_indices[::2]]
        right_brillouin_positions = [rescaled_x_axis[idx] for idx in brillouin_peaks_indices[1::2]]

    plot_vertical_lines(laser_positions, left_brillouin_positions, right_brillouin_positions)

    # Debug shading (tuple-aware)
    if debug_plot:
        plot_selected_peak_ranges(
            rescaled_x_axis,
            spectrum,
            laser_peak_ranges,
            brillouin_peak_ranges,
            perform_lorentzians_fitting,
            laser_positions=laser_positions,  # IMPORTANT for tuple-mode shading
            manual_laser_mode=manual_laser_mode
        )
        # NEW: visualize Lorentzian fitting windows
        if perform_lorentzians_fitting and isinstance(fit_lorentzians, (int, float)) and fit_lorentzians > 0:
            plot_fitting_windows(
                center_laser_list,
                center_left_list,
                center_right_list,
                half_window=fit_lorentzians
            )

    plt.xlabel("Frequency [GHz]", fontsize=14)
    plt.ylabel("Intensity [Arbitr. Units]", fontsize=14)
    if debug_plot:
        plt.title(
            f"Collected Signal at (x={x}, y={y}, z={z})\n"
            f"Median Shift: {median_shift:.2f} GHz, Median FWHM: {median_fwhm:.2f} GHz\n"
            f"Fitted Quadratic Coefficients: a={a:.4e}, b={b:.4f}, c={c:.2f}"
        )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if isinstance(save_plot, str):
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        plt.close(fig)
    elif save_plot is not True:
        plt.close(fig)
    else:
        plt.show()


def plot_fitted_lorentzians(
    amplitude_laser_list, center_laser_list, gamma_laser_list,
    amplitude_left_list, center_left_list, gamma_left_list,
    amplitude_right_list, center_right_list, gamma_right_list,
    *, plot_laser=True, plot_brilouin=True,
):
    # Rayleigh
    if plot_laser:
        for i in range(len(amplitude_laser_list)):
            A, xc, g = amplitude_laser_list[i], center_laser_list[i], gamma_laser_list[i]
            x_fit = np.linspace(xc - 3*g, xc + 3*g, 100)
            y_fit = lorentzian(x_fit, A, xc, g)
            plt.plot(x_fit, y_fit, color='green', linestyle='--', label='Fitted Rayleigh Peaks' if i == 0 else '')
    # Brillouin (left)
    if plot_brilouin:
        for i in range(len(amplitude_left_list)):
            A, xc, g = amplitude_left_list[i], center_left_list[i], gamma_left_list[i]
            x_fit = np.linspace(xc - 3*g, xc + 3*g, 100)
            y_fit = lorentzian(x_fit, A, xc, g)
            plt.plot(x_fit, y_fit, color='red', linestyle='--', label='Fitted Brillouin Peaks' if i == 0 else '')
    # Brillouin (right)
    if plot_brilouin:
        for i in range(len(amplitude_right_list)):
            A, xc, g = amplitude_right_list[i], center_right_list[i], gamma_right_list[i]
            x_fit = np.linspace(xc - 3*g, xc + 3*g, 100)
            y_fit = lorentzian(x_fit, A, xc, g)
            plt.plot(x_fit, y_fit, color='red', linestyle='--')


def highlight_peaks(
    rescaled_x_axis, spectrum,
    laser_peaks_indices, brillouin_peaks_indices,
    center_laser_list, center_left_list, center_right_list,
    amplitude_laser_list, amplitude_left_list, amplitude_right_list,
    fit_lorentzians, *, ignore_brilouin_peaks=False
):
    if fit_lorentzians:
        if center_laser_list and amplitude_laser_list:
            plt.plot(
                center_laser_list,
                amplitude_laser_list,
                "x",
                label="Rayleigh Peaks",
                color='green',
                markersize=10,
                markeredgewidth=4,
            )

        centers = center_left_list + center_right_list
        if ignore_brilouin_peaks:
            amps = [float(spectrum[idx]) for idx in brillouin_peaks_indices]
        else:
            amps = amplitude_left_list + amplitude_right_list
        if centers and amps:
            plt.plot(
                centers,
                amps,
                "o",
                label="Brillouin Peaks",
                color='red',
                markersize=8,
            )
    else:
        if len(laser_peaks_indices):
            plt.plot(
                rescaled_x_axis[laser_peaks_indices],
                spectrum[laser_peaks_indices],
                "x",
                label="Rayleigh Peaks",
                color='green',
                markersize=10,
                markeredgewidth=4,
            )
        if len(brillouin_peaks_indices):
            plt.plot(
                rescaled_x_axis[brillouin_peaks_indices],
                spectrum[brillouin_peaks_indices],
                "o",
                label="Brillouin Peaks",
                color='red',
                markersize=8,
            )


def annotate_shifts_fwhms(
    rescaled_x_axis, spectrum,
    laser_peaks_indices, brillouin_peaks_indices,
    shifts, fwhms_left_list, fwhms_right_list,
    center_left_list, amplitude_left_list,
    center_right_list, amplitude_right_list,
    fit_lorentzians, center_laser_list, amplitude_laser_list,
    *, ignore_brilouin_peaks=False
):
    for i in range(len(shifts)):
        shift = shifts[i]
        if fit_lorentzians:
            lp_pos = center_laser_list[i]
            lp_amp = amplitude_laser_list[i]
        else:
            lp_idx = laser_peaks_indices[i]
            lp_pos = rescaled_x_axis[lp_idx]
            lp_amp = spectrum[lp_idx]

        plt.annotate(
            f"$\\nu_B$: {shift:.2f} GHz",
            (lp_pos, lp_amp),
            textcoords="offset points", xytext=(0, 10), ha='center', color='black'
        )

    for i in range(len(fwhms_right_list)):
        if fit_lorentzians:
            if ignore_brilouin_peaks:
                r_idx = brillouin_peaks_indices[2*i + 1]
                l_idx = brillouin_peaks_indices[2*i]
                right_amp = spectrum[r_idx]
                left_amp = spectrum[l_idx]
            else:
                right_amp = amplitude_right_list[i]
                left_amp = amplitude_left_list[i]
            plt.annotate(
                f"Γ: {fwhms_right_list[i]:.2f} GHz",
                (center_right_list[i], right_amp),
                textcoords="offset points", xytext=(0, 10), ha='center', color='black'
            )
            plt.annotate(
                f"Γ: {fwhms_left_list[i]:.2f} GHz",
                (center_left_list[i], left_amp),
                textcoords="offset points", xytext=(0, 10), ha='center', color='black'
            )
        else:
            r_idx = brillouin_peaks_indices[2*i + 1]
            l_idx = brillouin_peaks_indices[2*i]
            plt.annotate(
                f"Γ: {fwhms_right_list[i]:.2f} GHz",
                (rescaled_x_axis[r_idx], spectrum[r_idx]),
                textcoords="offset points", xytext=(0, 10), ha='center', color='black'
            )
            plt.annotate(
                f"Γ: {fwhms_left_list[i]:.2f} GHz",
                (rescaled_x_axis[l_idx], spectrum[l_idx]),
                textcoords="offset points", xytext=(0, 10), ha='center', color='black'
            )


def annotate_distances(
    rescaled_x_axis, spectrum,
    laser_peaks_indices, brillouin_peaks_indices,
    center_laser_list, center_left_list, center_right_list,
    fit_lorentzians
):
    max_spectrum = np.max(spectrum)
    arrow_y = -max_spectrum * 0.05

    if fit_lorentzians:
        laser_positions = center_laser_list
        left_brillouin_positions = center_left_list
        right_brillouin_positions = center_right_list
    else:
        laser_positions = [rescaled_x_axis[idx] for idx in laser_peaks_indices]
        left_brillouin_positions = [rescaled_x_axis[idx] for idx in brillouin_peaks_indices[::2]]
        right_brillouin_positions = [rescaled_x_axis[idx] for idx in brillouin_peaks_indices[1::2]]

    for i in range(len(laser_positions)):
        lp_pos = laser_positions[i]
        bp_pos_left = left_brillouin_positions[i]
        bp_pos_right = right_brillouin_positions[i]
        distance_left = np.abs(bp_pos_left - lp_pos)
        distance_right = np.abs(bp_pos_right - lp_pos)

        plt.annotate('', xy=(lp_pos, arrow_y), xytext=(bp_pos_left, arrow_y),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        mid_x_left = (lp_pos + bp_pos_left) / 2
        plt.text(mid_x_left, arrow_y, f"{distance_left:.2f} GHz", color='red', ha='center', va='bottom', fontsize=8)

        plt.annotate('', xy=(lp_pos, arrow_y), xytext=(bp_pos_right, arrow_y),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        mid_x_right = (lp_pos + bp_pos_right) / 2
        plt.text(mid_x_right, arrow_y, f"{distance_right:.2f} GHz", color='red', ha='center', va='bottom', fontsize=8)

    if len(laser_positions) > 1:
        for i in range(len(laser_positions) - 1):
            x1 = laser_positions[i]; x2 = laser_positions[i + 1]
            distance = x2 - x1
            plt.annotate('', xy=(x1, 0), xytext=(x2, 0), arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
            mid_x = (x1 + x2) / 2; mid_y = -max_spectrum * 0.03
            plt.text(mid_x, mid_y, f"{distance:.2f} GHz", color='green', ha='center', va='bottom', fontsize=8)


def plot_vertical_lines(laser_positions, left_brillouin_positions, right_brillouin_positions):
    for lp_pos in laser_positions:
        plt.axvline(x=lp_pos, color='green', linestyle='--', linewidth=1.5, alpha=0.2)
    for bp_pos in left_brillouin_positions:
        plt.axvline(x=bp_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.2)
    for bp_pos in right_brillouin_positions:
        plt.axvline(x=bp_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.2)


def plot_selected_peak_ranges(
    rescaled_x_axis,
    spectrum,
    laser_peak_ranges,
    brillouin_peak_ranges,
    fit_lorentzians,
    laser_positions=None,
    manual_laser_mode=False
):
    """Debug shading for chosen ranges.
    - laser_peak_ranges are index ranges (start, end)
    - brillouin_peak_ranges is either list-of-pairs of index ranges, or (a_GHz, b_GHz) tuple
    """
    n = len(rescaled_x_axis)

    manual_laser_mode = bool(manual_laser_mode)

    # Laser index ranges (clip + sort)
    if manual_laser_mode:
        if laser_positions is not None:
            label_added = False
            for pos in laser_positions:
                if pos is None or not np.isfinite(pos):
                    continue
                label = "Manual Laser Peak" if not label_added else None
                plt.axvline(x=pos, color='green', linestyle=':', linewidth=1.2, label=label)
                label_added = True
    else:
        if laser_peak_ranges:
            for start, end in laser_peak_ranges:
                rng = _normalize_range_pair(start, end, n)
                if rng:
                    s, e = rng
                    plt.axvspan(rescaled_x_axis[s], rescaled_x_axis[e-1], color='green', alpha=0.1)

    # Brillouin ranges
    if isinstance(brillouin_peak_ranges, tuple) and len(brillouin_peak_ranges) == 2:
        # tuple-mode: (a_val, b_val) in GHz around each laser
        if laser_positions is None:
            return
        a_val, b_val = brillouin_peak_ranges
        for lp in laser_positions:
            # left band [lp-b, lp-a]
            l0, l1 = lp - b_val, lp - a_val
            # right band [lp+a, lp+b]
            r0, r1 = lp + a_val, lp + b_val

            sl = np.searchsorted(rescaled_x_axis, min(l0, l1))
            el = np.searchsorted(rescaled_x_axis, max(l0, l1))
            sr = np.searchsorted(rescaled_x_axis, min(r0, r1))
            er = np.searchsorted(rescaled_x_axis, max(r0, r1))

            sl, el = max(0, sl), min(n, el)
            sr, er = max(0, sr), min(n, er)

            if el > sl:
                plt.axvspan(rescaled_x_axis[sl], rescaled_x_axis[el-1], color='red', alpha=0.1)
            if er > sr:
                plt.axvspan(rescaled_x_axis[sr], rescaled_x_axis[er-1], color='red', alpha=0.1)
    else:
        # index ranges: list of (left_range, right_range)
        for (left_range, right_range) in brillouin_peak_ranges:
            lr = _normalize_range_pair(left_range[0], left_range[1], n)
            rr = _normalize_range_pair(right_range[0], right_range[1], n)
            if lr:
                s, e = lr
                plt.axvspan(rescaled_x_axis[s], rescaled_x_axis[e-1], color='red', alpha=0.1)
            if rr:
                s, e = rr
                plt.axvspan(rescaled_x_axis[s], rescaled_x_axis[e-1], color='red', alpha=0.1)


# NEW helper: shade Lorentzian fitting windows (center ± half_window) in debug mode
def plot_fitting_windows(laser_centers, left_centers, right_centers, half_window):
    """Shade the frequency windows used for Lorentzian fitting (debug only)."""
    if half_window <= 0:
        return
    added_label = False
    for c in (laser_centers + left_centers + right_centers):
        if c is None:
            continue
        label = "Lorentzian Fit Window" if not added_label else None
        plt.axvspan(c - half_window, c + half_window, color='purple', alpha=0.08, label=label)
        added_label = True

# ---------- batch over Z ----------
def analyze_brillouin_spectra_manual(
    brillouin_spectra,
    z_coord=None,
    laser_peak_ranges=[626, 986, 1269, 1510],
    brillouin_peak_ranges=(6, 11),
    spectrum_cut_range=(350, 1600),
    free_spectral_range=29.98,
    fit_lorentzians=1.5,       # GHz half-window; 0 -> off
    refit=False,                # keep API; global refit retained
    filter_settings=[{'type': 'savgol', 'params': (15, 6)},
    {'type': 'wavelet', 'wavelet': 'db6', 'level': 5},
    {'type': 'fft', 'params': (20, 100)}],
    # pass-through optional fitting constraints
    fwhm_bounds_GHz=(0.8, 4.0),
    center_slack_GHz=0.5,
    max_workers=None,
    parallel_backend='auto',
    keep_waveforms=False,
    baseline_spectra=True,
    poisson_weighting=False,
    match_brilouin_parameters=True,
    ignore_laser_peaks=False,
    laser_refit=True,
):

    """
    Process one z slice or all z slices.
    Returns:
      - list of peaks_map per z if z_coord is None
      - single peaks_map if z_coord is not None
    Each peaks_map: {(x,y): result}
    Parallel execution can use threads or processes; set max_workers=1 to force sequential mode.
    Plots are not generated in this batch routine; call analyze_brillouin_spectrum_manual for visualization.
    parallel_backend: 'auto' (default), 'thread', or 'process'. The 'process' backend
    ships individual spectra to worker processes so compute-heavy fitting can bypass
    the GIL and usually provides the best throughput when multiple workers are allowed.
    keep_waveforms: when False, remove raw spectrum/rescaled axis from each pixel entry
    to reduce memory once processing (and optional refit) is finished.
    baseline_spectra / poisson_weighting: forwarded to analyze_brillouin_spectrum_manual.
    match_brilouin_parameters: pass-through to per-spectrum analyzer to enforce shared Brillouin
    shifts/FWHMs when enabled.
    ignore_laser_peaks: Can be a bool or dict. See analyze_brillouin_spectrum_manual for details.
    When dict, supports 'brillouin_shift_average' and 'ignore_laser_for_fit' keys.
    laser_refit: when True, run a laser-only pre-pass to lock Rayleigh peak locations via
    their across-spectra median before recomputing full results.
    """
    if brillouin_spectra is None:
        print("No Brillouin spectra provided.")
        return None

    x_dim, y_dim, z_dim = brillouin_spectra.shape
    lateral_step = get_lateral_step(source=brillouin_spectra)
    z_list = range(z_dim) if z_coord is None else [z_coord]
    if any(z >= z_dim or z < 0 for z in z_list):
        print(f"Z-coordinate out of bounds.")
        return None

    # Note: baseline_spectra and poisson_weighting are converted to bool in
    # analyze_brillouin_spectrum_manual, so no need to convert here
    match_brilouin_parameters = bool(match_brilouin_parameters)
    # ignore_laser_peaks can be bool or dict, pass through as-is to per-spectrum analyzer
    laser_refit = bool(laser_refit)
    ignore_brilouin_peaks = False
    total_pixels = x_dim * y_dim * len(z_list)
    print(f"Processing {total_pixels} pixels across {len(z_list)} z slices...")

    # Local bindings for frequently used numpy functions (avoids attribute lookups)
    _np_array = np.array
    _np_arange = np.arange
    _np_median = np.median
    _np_nan = np.nan

    # global refit accumulators
    x_all, y_all, pixel_indices = [], [], []
    pixel_id_map, pixel_id_reverse_map = {}, {}
    best_a_list, best_b_list, best_c_list = [], [], {}
    peaks_map_global = BrillouinPeaksMap(lateral_step=lateral_step)
    current_pixel_id = 0

    # ---------- per-pixel pass ----------
    # Use list comprehension instead of nested loops with conditionals
    pixel_specs = [(x, y, z) for z in z_list for x in range(x_dim) for y in range(y_dim)
                   if brillouin_spectra[x, y, z] is not None]
    skipped_pixels = total_pixels - len(pixel_specs)

    worker_kwargs = dict(
        free_spectral_range=free_spectral_range,
        laser_peak_ranges=laser_peak_ranges,
        brillouin_peak_ranges=brillouin_peak_ranges,
        spectrum_cut_range=spectrum_cut_range,
        make_plot=False,
        fit_lorentzians=fit_lorentzians,
        debug_plot=False,
        filter_settings=filter_settings,
        save_plot=False,
        fwhm_bounds_GHz=fwhm_bounds_GHz,
        center_slack_GHz=center_slack_GHz,
        baseline_spectra=baseline_spectra,
        poisson_weighting=poisson_weighting,
        match_brilouin_parameters=match_brilouin_parameters,
        ignore_brilouin_peaks=False,
        ignore_laser_peaks=ignore_laser_peaks,
    )

    def _run_pixel_pass(local_kwargs, desc):
        if total_pixels == 0:
            return []

        num_specs = len(pixel_specs)
        if max_workers is None:
            default_workers = min(32, (os.cpu_count() or 1) + 4)
            effective_workers = min(num_specs, default_workers) if num_specs else 1
        else:
            try:
                requested_workers = int(max_workers)
            except (TypeError, ValueError):
                requested_workers = 1
            effective_workers = max(1, requested_workers)
            effective_workers = min(effective_workers, num_specs) if num_specs else 1

        backend = (parallel_backend or 'auto').lower()
        if backend not in {'auto', 'thread', 'process'}:
            raise ValueError("parallel_backend must be 'auto', 'thread', or 'process'.")

        use_parallel = effective_workers > 1
        if backend == 'auto':
            backend = 'process' if use_parallel else 'thread'
        if backend == 'process' and not use_parallel:
            backend = 'thread'

        pbar_local = tqdm(total=total_pixels, desc=desc)

        if num_specs:
            if not use_parallel:
                # Pre-allocate list for sequential mode (avoids repeated list resizing)
                entries = [None] * num_specs
                for i, coords in enumerate(pixel_specs):
                    entry = _process_pixel_common(brillouin_spectra, coords, local_kwargs)
                    entries[i] = entry
                    x, y, z = entry['pixel_key']
                    pbar_local.set_description(f"{desc} (x={x}, y={y}, z={z})", refresh=True)
                    pbar_local.update(1)
            else:
                # For parallel modes, collect results as they complete
                entries = []
                if backend == 'thread':
                    def _thread_worker(coords):
                        return _process_pixel_common(brillouin_spectra, coords, local_kwargs)

                    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                        for entry in executor.map(_thread_worker, pixel_specs):
                            entries.append(entry)
                            x, y, z = entry['pixel_key']
                            pbar_local.set_description(f"{desc} (x={x}, y={y}, z={z})", refresh=True)
                            pbar_local.update(1)
                else:
                    try:
                        ctx = get_context('fork')
                    except ValueError:
                        ctx = get_context('spawn')
                    payload_iter = (
                        (coords, brillouin_spectra[coords[0], coords[1], coords[2]], local_kwargs)
                        for coords in pixel_specs
                    )
                    # Optimize chunksize: larger chunks reduce IPC overhead for many tasks
                    # Use ~4 chunks per worker for good load balancing
                    optimal_chunksize = max(1, num_specs // (effective_workers * 4))
                    with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as executor:
                        for entry in executor.map(_process_pixel_with_payload, payload_iter, chunksize=optimal_chunksize):
                            entries.append(entry)
                            x, y, z = entry['pixel_key']
                            pbar_local.set_description(f"{desc} (x={x}, y={y}, z={z})", refresh=True)
                            pbar_local.update(1)
        else:
            entries = []

        if skipped_pixels:
            pbar_local.update(skipped_pixels)
        pbar_local.close()
        return entries

    processed_entries = []

    if laser_refit:
        worker_kwargs_first = dict(worker_kwargs)
        worker_kwargs_first['ignore_brilouin_peaks'] = True
        worker_kwargs_first['match_brilouin_parameters'] = False
        first_pass_entries = _run_pixel_pass(worker_kwargs_first, "Laser pre-pass")

        median_indices = None
        per_peak_values = []
        for entry in first_pass_entries:
            result = entry['result']
            if not result:
                continue
            indices = result.get('laser_peaks_indices')
            if not indices:
                continue
            if not per_peak_values:
                per_peak_values = [[] for _ in range(len(indices))]
            if len(indices) != len(per_peak_values):
                continue
            for idx, val in enumerate(indices):
                per_peak_values[idx].append(int(val))

        if per_peak_values and all(len(vals) for vals in per_peak_values):
            median_indices = [int(round(float(_np_median(vals)))) for vals in per_peak_values]

        worker_kwargs_second = dict(worker_kwargs)
        worker_kwargs_second['ignore_brilouin_peaks'] = False
        worker_kwargs_second['match_brilouin_parameters'] = match_brilouin_parameters
        if median_indices and len(median_indices) == len(worker_kwargs_second['laser_peak_ranges']):
            worker_kwargs_second['laser_peak_ranges'] = median_indices
        processed_entries = _run_pixel_pass(worker_kwargs_second, "Processing Brillouin Shift and FWHM")
        del first_pass_entries
    else:
        worker_kwargs['ignore_brilouin_peaks'] = ignore_brilouin_peaks
        worker_kwargs['match_brilouin_parameters'] = match_brilouin_parameters
        processed_entries = _run_pixel_pass(worker_kwargs, "Processing Brillouin Shift and FWHM")

    # aggregate per-pixel results serially (deterministic order)
    processed_entries.sort(key=lambda item: (item['pixel_key'][2], item['pixel_key'][0], item['pixel_key'][1]))
    for entry in processed_entries:
        pixel_key = entry['pixel_key']
        result = entry['result']
        if result is None:
            peaks_map_global[pixel_key] = _np_nan
            continue

        peaks_map_global[pixel_key] = result
        pixel_id_map[pixel_key] = current_pixel_id
        pixel_id_reverse_map[current_pixel_id] = pixel_key
        current_pixel_id += 1

        a0, b0, c0 = result['fit_params']
        best_a_list.append(a0); best_b_list.append(b0); best_c_list[pixel_key] = c0
        lidx = result['laser_peaks_indices']
        if len(lidx) >= 3:
            y_expected = _np_arange(len(lidx)) * free_spectral_range
            x_all.extend(lidx)
            y_all.extend(y_expected)
            pixel_indices.extend([pixel_id_map[pixel_key]] * len(lidx))

    # ---------- optional global refit (kept to match your API) ----------
    if refit and (not laser_refit) and current_pixel_id > 0:
        print("Performing global refit across all slices...")
        x_all = _np_array(x_all); y_all = _np_array(y_all); pixel_indices = _np_array(pixel_indices)
        N_pixels = current_pixel_id
        x_all_squared = x_all ** 2

        def global_residuals(params, x2, x, y):
            a, b, c = params
            return y - (a * x2 + b * x + c)

        a_init = _np_median(best_a_list) if best_a_list else 1e-16
        b_init = _np_median(best_b_list) if best_b_list else 1e-16
        c_init = _np_median(list(best_c_list.values())) if best_c_list else 0.0

        result = least_squares(global_residuals, x0=[a_init, b_init, c_init],
                               bounds=([1e-16, 1e-16, -np.inf], [np.inf, np.inf, np.inf]),
                               args=(x_all_squared, x_all, y_all))
        if result.success:
            print("Global refit successful.")
            a_global, b_global, c_global = [float(v) for v in result.x]
            N_pixels = current_pixel_id

            # helper: invert old mapping to get an index for an old GHz center
            def _solve_for_x(a_old, b_old, c_old, center_old_GHz, N):
                A = float(a_old); B = float(b_old); C = float(c_old) - float(center_old_GHz)
                # handle near-linear case safely
                if abs(A) < 1e-30:
                    if abs(B) < 1e-30:
                        return np.nan
                    x = -C / B
                    return x if 0 <= x < N else np.nan
                disc = B*B - 4*A*C
                if disc < 0:
                    return np.nan
                s = np.sqrt(disc)
                x1 = (-B + s) / (2*A)
                x2 = (-B - s) / (2*A)
                for cand in (x1, x2):
                    if 0 <= cand < N:
                        return float(cand)
                return np.nan

            # pass 1: build new rescaled axes and refit-derived per-pixel results
            for pid in range(N_pixels):
                x_c, y_c, z_c = pixel_id_reverse_map[pid]
                pixel_key = (x_c, y_c, z_c)
                fitted = peaks_map_global[pixel_key]
                if isinstance(fitted, float) and np.isnan(fitted):
                    continue

                spectrum = np.asarray(fitted['spectrum'], dtype=float)
                N = len(spectrum)
                x_axis = _np_arange(N)
                rescaled_x_axis_new = a_global * x_axis**2 + b_global * x_axis + c_global

                # stash for consumers/plots
                # (we will put this into the 'refitted' dict at the end)
                a_old, b_old, c_old = fitted.get('fit_params', (a_global, b_global, c_global))

                # was Lorentzian fitting used originally?
                used_lorentz = (
                    len(fitted.get('center_laser_list', [])) > 0 or
                    len(fitted.get('center_left_list',  [])) > 0 or
                    len(fitted.get('center_right_list', [])) > 0
                )

                if used_lorentz:
                    # Remap old centers (GHz under old axis) -> index under old axis -> GHz under global axis
                    old_laser  = fitted.get('center_laser_list', [])
                    old_left   = fitted.get('center_left_list',  [])
                    old_right  = fitted.get('center_right_list', [])

                    new_laser  = []
                    new_left   = []
                    new_right  = []

                    for i in range(len(old_laser)):
                        # laser
                        xi = _solve_for_x(a_old, b_old, c_old, old_laser[i], N)
                        if np.isnan(xi): xi = float(fitted['laser_peaks_indices'][i])
                        new_laser.append(a_global*xi*xi + b_global*xi + c_global)

                        # left
                        xi = _solve_for_x(a_old, b_old, c_old, old_left[i], N)
                        if np.isnan(xi): xi = float(fitted['brillouin_peaks_indices'][2*i])
                        new_left.append(a_global*xi*xi + b_global*xi + c_global)

                        # right
                        xi = _solve_for_x(a_old, b_old, c_old, old_right[i], N)
                        if np.isnan(xi): xi = float(fitted['brillouin_peaks_indices'][2*i+1])
                        new_right.append(a_global*xi*xi + b_global*xi + c_global)

                    # recompute shifts/FWHMs (keep gamma-based FWHM)
                    gamma_left  = fitted.get('gamma_left_list',  [])
                    gamma_right = fitted.get('gamma_right_list', [])
                    shifts = []
                    left_shifts = []
                    right_shifts = []
                    fwhms = []
                    fwhms_left_list = []
                    fwhms_right_list = []

                    for i in range(len(new_laser)):
                        left_shifts.append(new_laser[i] - new_left[i])
                        right_shifts.append(new_right[i] - new_laser[i])
                        shifts.append((new_right[i] - new_left[i]) / 2.0)

                        fL = 2.0 * (gamma_left[i]  if i < len(gamma_left)  else np.nan)
                        fR = 2.0 * (gamma_right[i] if i < len(gamma_right) else np.nan)
                        fwhms_left_list.append(fL)
                        fwhms_right_list.append(fR)
                        fwhms.append(np.nanmean([fL, fR]))

                    # update dataframe
                    df = fitted['df'].copy()
                    df.loc[df['Peak Type']=='Laser',           'Shift'] = shifts
                    df.loc[df['Peak Type']=='Laser',           'FWHM']  = fwhms
                    df.loc[df['Peak Type']=='Brillouin Left',  'Shift'] = left_shifts
                    df.loc[df['Peak Type']=='Brillouin Left',  'FWHM']  = fwhms_left_list
                    df.loc[df['Peak Type']=='Brillouin Right', 'Shift'] = right_shifts
                    df.loc[df['Peak Type']=='Brillouin Right', 'FWHM']  = fwhms_right_list
                    df.loc[df['Peak Type']=='Laser',           'Center (GHz)'] = new_laser
                    df.loc[df['Peak Type']=='Brillouin Left',  'Center (GHz)'] = new_left
                    df.loc[df['Peak Type']=='Brillouin Right', 'Center (GHz)'] = new_right
                    df.loc[df['Peak Type']=='Laser',           'Gamma (GHz)'] = (np.asarray(fwhms, dtype=float) / 2.0)
                    df.loc[df['Peak Type']=='Brillouin Left',  'Gamma (GHz)'] = (np.asarray(fwhms_left_list, dtype=float) / 2.0)
                    df.loc[df['Peak Type']=='Brillouin Right', 'Gamma (GHz)'] = (np.asarray(fwhms_right_list, dtype=float) / 2.0)

                    refitted = fitted.copy()
                    refitted.update({
                        'df': df,
                        'rescaled_x_axis': rescaled_x_axis_new,
                        'a': a_global, 'b': b_global, 'c': c_global,
                        'fit_params': (a_global, b_global, c_global),
                        'center_laser_list': new_laser,
                        'center_left_list':  new_left,
                        'center_right_list': new_right,
                        'shifts': shifts,
                        'left_shifts': left_shifts,
                        'right_shifts': right_shifts,
                        'fwhms': fwhms,
                        'fwhms_left_list': fwhms_left_list,
                        'fwhms_right_list': fwhms_right_list,
                        'median_shift': np.nanmedian(shifts) if len(shifts) else np.nan,
                        'median_fwhm':  np.nanmedian(fwhms)  if len(fwhms)  else np.nan,
                    })
                    peaks_map_global[pixel_key] = refitted

                else:
                    # No Lorentzian fitting: recompute from indices on the new axis
                    lp_idx = np.array(fitted['laser_peaks_indices'], dtype=int)
                    bpi    = np.array(fitted['brillouin_peaks_indices'], dtype=int)
                    left_idx  = bpi[::2]
                    right_idx = bpi[1::2]

                    lp_pos   = rescaled_x_axis_new[lp_idx]
                    left_pos = rescaled_x_axis_new[left_idx]
                    right_pos= rescaled_x_axis_new[right_idx]

                    left_shifts  = lp_pos - left_pos
                    right_shifts = right_pos - lp_pos
                    shifts       = (right_pos - left_pos)/2.0

                    # FWHM from peak_widths * local dx under the new axis
                    local_dx = np.gradient(rescaled_x_axis_new)
                    fL = peak_widths(spectrum, left_idx,  rel_height=0.5)[0] * np.abs(local_dx[left_idx])
                    fR = peak_widths(spectrum, right_idx, rel_height=0.5)[0] * np.abs(local_dx[right_idx])
                    fwhms_left_list  = fL.astype(float).tolist()
                    fwhms_right_list = fR.astype(float).tolist()
                    fwhms            = (fL + fR)/2.0

                    df = fitted['df'].copy()
                    df.loc[df['Peak Type']=='Laser',           'Shift'] = shifts
                    df.loc[df['Peak Type']=='Laser',           'FWHM']  = fwhms
                    df.loc[df['Peak Type']=='Brillouin Left',  'Shift'] = left_shifts
                    df.loc[df['Peak Type']=='Brillouin Left',  'FWHM']  = fL
                    df.loc[df['Peak Type']=='Brillouin Right', 'Shift'] = right_shifts
                    df.loc[df['Peak Type']=='Brillouin Right', 'FWHM']  = fR
                    df.loc[df['Peak Type']=='Laser',           'Center (GHz)'] = lp_pos
                    df.loc[df['Peak Type']=='Brillouin Left',  'Center (GHz)'] = left_pos
                    df.loc[df['Peak Type']=='Brillouin Right', 'Center (GHz)'] = right_pos
                    df.loc[df['Peak Type']=='Laser',           'Gamma (GHz)'] = (np.asarray(fwhms, dtype=float) / 2.0)
                    df.loc[df['Peak Type']=='Brillouin Left',  'Gamma (GHz)'] = (np.asarray(fL, dtype=float) / 2.0)
                    df.loc[df['Peak Type']=='Brillouin Right', 'Gamma (GHz)'] = (np.asarray(fR, dtype=float) / 2.0)

                    refitted = fitted.copy()
                    refitted.update({
                        'df': df,
                        'rescaled_x_axis': rescaled_x_axis_new,
                        'a': a_global, 'b': b_global, 'c': c_global,
                        'fit_params': (a_global, b_global, c_global),
                        'shifts': shifts.tolist(),
                        'left_shifts': left_shifts.tolist(),
                        'right_shifts': right_shifts.tolist(),
                        'fwhms': fwhms.astype(float).tolist(),
                        'fwhms_left_list': fwhms_left_list,
                        'fwhms_right_list': fwhms_right_list,
                        'median_shift': np.nanmedian(shifts) if len(shifts) else np.nan,
                        'median_fwhm':  np.nanmedian(fwhms)  if len(fwhms)  else np.nan,
                    })
                    peaks_map_global[pixel_key] = refitted
        else:
            print("Global refit unsuccessful or not performed.")
    else:
        print("No pixels to refit.")

    if not keep_waveforms:
        def _strip_waveforms(payload):
            if not isinstance(payload, dict):
                return

            spectrum = payload.pop('spectrum', None)
            rescaled_axis = payload.pop('rescaled_x_axis', None)

            if 'spectrum_length' not in payload:
                if spectrum is not None:
                    payload['spectrum_length'] = int(len(spectrum))
                elif rescaled_axis is not None:
                    payload['spectrum_length'] = int(len(rescaled_axis))
                else:
                    length_guess = None
                    for idx_list in (
                        payload.get('laser_peaks_indices'),
                        payload.get('brillouin_peaks_indices'),
                    ):
                        if isinstance(idx_list, (list, tuple, np.ndarray)) and len(idx_list):
                            try:
                                length_guess = int(np.max(idx_list)) + 1
                                break
                            except Exception:
                                continue
                    if length_guess is not None:
                        payload['spectrum_length'] = length_guess

            if spectrum is not None and 'spectrum_length' not in payload:
                payload['spectrum_length'] = int(len(spectrum))

            payload['waveform_retained'] = False

        for entry in peaks_map_global.values():
            _strip_waveforms(entry)

    # Return structure
    if z_coord is None:
        out = []
        for z in z_list:
            peaks_map = BrillouinPeaksMap(lateral_step=lateral_step)
            for x in range(x_dim):
                for y in range(y_dim):
                    key = (x, y, z)
                    if key in peaks_map_global:
                        peaks_map[(x, y)] = peaks_map_global[key]
            out.append(peaks_map)
        return out
    else:
        peaks_map = BrillouinPeaksMap(lateral_step=lateral_step)
        for x in range(x_dim):
            for y in range(y_dim):
                key = (x, y, z_coord)
                if key in peaks_map_global:
                    peaks_map[(x, y)] = peaks_map_global[key]
        return peaks_map
