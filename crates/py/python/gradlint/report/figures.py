"""Static matplotlib figures for the report, returned as base64 PNG data URIs.

Imported lazily by the HTML renderer so the Markdown path needs no matplotlib.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .model import Report  # noqa: E402

STATUS_COLORS = {"PASS": "#1a8a3a", "WARN": "#c77800", "FLAG": "#c0271b"}
_DPI = 110
_GLYPH_DPI = 160

# Fixed relative-margin cut-off for an automatic repair (mirrors the Rust core).
FLAG_THRESHOLD = 0.02


def build_figures(
    report: Report, *, glyphs: dict[str, Any] | None = None
) -> dict[str, str]:
    """Render every figure the report has data for, keyed by slug."""
    figures: dict[str, str] = {}
    if report.shells is not None:
        figures["shells"] = shell_histogram(report)
    if report.scheme.directions:
        figures["sphere"] = direction_sphere(report)
    if report.b0_drift is not None and len(report.b0_drift.indices) > 1:
        figures["drift"] = b0_drift_plot(report)
    if report.flip is not None:
        if report.flip.ranking:
            figures["coherence"] = coherence_bar(report)
        figures["margin"] = margin_gauge(report)
        if glyphs is not None:
            from .glyphs import glyph_figure

            figures["glyphs"] = _encode(
                glyph_figure(report, glyphs), tight=False, dpi=_GLYPH_DPI
            )
    return figures


def shell_histogram(report: Report) -> str:
    """Volume counts per detected shell (b0 plus each diffusion shell)."""
    shells = report.shells.shells if report.shells else []
    labels = ["b0" if s.is_b0 else f"b≈{s.nominal_b:g}" for s in shells]
    counts = [s.count for s in shells]
    colors = ["#6c757d" if s.is_b0 else "#2b6cb0" for s in shells]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bars = ax.bar(labels, counts, color=colors)
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_ylabel("volumes")
    ax.set_title("Shell structure")
    ax.margins(y=0.18)
    return _encode(fig)


def direction_sphere(report: Report) -> str:
    """Unit gradient directions on the sphere, with antipodal mirrors."""
    dirs = np.asarray(report.scheme.directions, dtype=float)
    bvals = np.asarray(report.scheme.bvals, dtype=float)
    norms = np.linalg.norm(dirs, axis=1)
    keep = norms > 1e-6
    unit = dirs[keep] / norms[keep, None]

    fig = plt.figure(figsize=(4.4, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 12j]
    ax.plot_wireframe(
        np.cos(u) * np.sin(v),
        np.sin(u) * np.sin(v),
        np.cos(v),
        color="#dddddd",
        linewidth=0.4,
    )
    if unit.size:
        c = bvals[keep]
        ax.scatter(unit[:, 0], unit[:, 1], unit[:, 2], c=c, cmap="viridis", s=14)
        ax.scatter(-unit[:, 0], -unit[:, 1], -unit[:, 2], c=c, cmap="viridis", s=14)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Gradient directions")
    return _encode(fig)


def b0_drift_plot(report: Report) -> str:
    """Mean b0 signal across the series with the fitted trend line."""
    drift = report.b0_drift
    assert drift is not None
    x = np.asarray(drift.indices, dtype=float)
    y = np.asarray(drift.mean_signal, dtype=float)
    intercept = y.mean() - drift.slope * x.mean()

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(x, y, "o-", color="#2b6cb0", label="mean b0 signal")
    ax.plot(x, drift.slope * x + intercept, "--", color="#c77800", label="trend")
    ax.set_xlabel("volume index")
    ax.set_ylabel("mean signal")
    ax.set_title(f"b0 drift ({drift.relative_drift * 100:.1f}%)")
    ax.legend(fontsize=8)
    return _encode(fig)


def coherence_bar(report: Report) -> str:
    """Coherence index across candidate transforms, best/identity highlighted."""
    flip = report.flip
    assert flip is not None
    ranking = flip.ranking
    labels = [c.label for c in ranking]
    values = [c.coherence for c in ranking]
    colors = []
    for c in ranking:
        if c.label == flip.best.label:
            colors.append("#c0271b" if not c.is_identity else "#1a8a3a")
        elif c.is_identity:
            colors.append("#1a8a3a")
        else:
            colors.append("#b0bec5")

    width = max(5.2, len(labels) * 0.16)
    fig, ax = plt.subplots(figsize=(width, 3.0))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("coherence index")
    ax.set_title("Candidate ranking")
    return _encode(fig)


def margin_gauge(report: Report) -> str:
    """Confidence margin against the auto-repair threshold, with the decision band."""
    flip = report.flip
    assert flip is not None
    margin = flip.relative_margin * 100.0
    threshold = FLAG_THRESHOLD * 100.0
    hi = max(margin, threshold) * 1.35
    bar_color = STATUS_COLORS.get(report.status, "#444444")

    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    ax.axvspan(0.0, threshold, color="#fbe6cf", zorder=0)  # repair withheld
    ax.axvspan(threshold, hi, color="#e3f0e6", zorder=0)  # confident repair
    ax.axvline(threshold, color="#c77800", lw=1.2, ls="--", zorder=1)
    ax.barh([0], [margin], height=0.5, color=bar_color, zorder=2)
    ax.text(
        threshold,
        0.55,
        f"{threshold:g}% repair threshold",
        color="#c77800",
        fontsize=7,
        ha="center",
        va="bottom",
    )
    ax.annotate(
        f"{margin:.2f}%  {flip.decision}",
        xy=(margin, 0),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlim(0.0, hi)
    ax.set_ylim(-0.6, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("relative confidence margin (%)")
    ax.set_title("Repair margin vs threshold")
    return _encode(fig)


def save_figures(
    report: Report,
    out_dir: Path,
    *,
    glyphs: dict[str, Any] | None = None,
) -> list[Path]:
    """Write every available report figure as a PNG into ``out_dir``."""
    return save_figure_data(build_figures(report, glyphs=glyphs), out_dir)


def save_figure_data(figures: dict[str, str], out_dir: Path) -> list[Path]:
    """Write pre-rendered figure data URIs as PNGs into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for slug, uri in figures.items():
        payload = base64.b64decode(uri.split(",", 1)[1])
        path = out_dir / f"{slug}.png"
        path.write_bytes(payload)
        written.append(path)
    return written


def _encode(fig: plt.Figure, *, tight: bool = True, dpi: int = _DPI) -> str:
    buffer = io.BytesIO()
    if tight:
        fig.tight_layout()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight" if tight else None,
    )
    plt.close(fig)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"
