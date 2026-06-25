"""gradlint: gradient-scheme QC and b-vector repair for diffusion MRI."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from ._gradlint import version as _core_version
from .api import audit, detect_flip, inspect, recompute_bval, repair
from .report import Report, load_report, render_html, render_markdown

__version__ = _pkg_version("gradlint")

__all__ = [
    "__version__",
    "core_version",
    "Report",
    "load_report",
    "render_markdown",
    "render_html",
    "inspect",
    "audit",
    "detect_flip",
    "repair",
    "recompute_bval",
]


def core_version() -> str:
    return _core_version()
