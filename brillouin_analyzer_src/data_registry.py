"""Utility registry for dataset-level metadata such as lateral step."""
from __future__ import annotations

from typing import Any, Iterable, Optional
import weakref

_LATERAL_STEP_BY_ID: dict[int, tuple[weakref.ReferenceType[Any] | None, float]] = {}
_LATERAL_STEP_BY_LABEL: dict[str, float] = {}
_LATERAL_STEP_BY_PATH: dict[str, float] = {}


class BrillouinPeaksMap(dict):
    """Dictionary-like container that carries the lateral step value."""

    def __init__(self, *args: Any, lateral_step: Optional[float] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lateral_step: Optional[float] = lateral_step

    def copy(self) -> "BrillouinPeaksMap":
        copied = super().copy()
        return BrillouinPeaksMap(copied, lateral_step=self.lateral_step)

    def get_lateral_step(self, default: Optional[float] = None) -> Optional[float]:
        return self.lateral_step if self.lateral_step is not None else default


def _register_for_source(source: Any, lateral_step: float) -> None:
    if source is None:
        return
    key = id(source)

    def _cleanup(_: weakref.ReferenceType[Any]) -> None:
        _LATERAL_STEP_BY_ID.pop(key, None)

    try:
        ref = weakref.ref(source, _cleanup)
    except TypeError:
        ref = None
    _LATERAL_STEP_BY_ID[key] = (ref, lateral_step)


def register_lateral_step(
    *,
    source: Any = None,
    lateral_step: Optional[float],
    directory_path: Optional[str] = None,
    file_label: Optional[str] = None,
) -> None:
    """Register the lateral step value for a dataset.

    Parameters
    ----------
    source:
        The in-memory object representing the dataset (e.g., spectra array).
    lateral_step:
        The lateral step in micrometers.
    directory_path:
        Directory where the dataset resides.
    file_label:
        Label used when parsing the dataset.
    """
    if lateral_step is None:
        return

    lateral_step = float(lateral_step)
    _register_for_source(source, lateral_step)

    if directory_path:
        _LATERAL_STEP_BY_PATH[directory_path] = lateral_step
    if file_label:
        _LATERAL_STEP_BY_LABEL[file_label] = lateral_step


def get_lateral_step(
    *,
    source: Any = None,
    directory_path: Optional[str] = None,
    file_label: Optional[str] = None,
) -> Optional[float]:
    """Retrieve a stored lateral step value if available."""
    if source is not None:
        entry = _LATERAL_STEP_BY_ID.get(id(source))
        if entry:
            ref, value = entry
            if ref is None:
                return value
            ref_obj = ref()
            if ref_obj is source:
                return value
            if ref_obj is None:
                _LATERAL_STEP_BY_ID.pop(id(source), None)

    if file_label is not None and file_label in _LATERAL_STEP_BY_LABEL:
        return _LATERAL_STEP_BY_LABEL[file_label]

    if directory_path is not None and directory_path in _LATERAL_STEP_BY_PATH:
        return _LATERAL_STEP_BY_PATH[directory_path]

    return None


def extract_lateral_step_from_container(peaks_container: Any) -> Optional[float]:
    """Attempt to extract the lateral step from a peaks container or list of containers."""
    if isinstance(peaks_container, BrillouinPeaksMap):
        return peaks_container.lateral_step

    if isinstance(peaks_container, dict) and hasattr(peaks_container, "lateral_step"):
        return getattr(peaks_container, "lateral_step")

    if isinstance(peaks_container, Iterable) and not isinstance(peaks_container, (str, bytes)):
        for item in peaks_container:
            value = extract_lateral_step_from_container(item)
            if value is not None:
                return value

    return None
