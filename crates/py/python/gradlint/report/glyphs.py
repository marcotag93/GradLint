from __future__ import annotations

import itertools
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from .model import Report, correction_presentation

RADIUS_MM = 28.0
STRIDE_MM = 2.5
GLYPH_LENGTH_MM = 3.2
FA_THRESHOLD = 0.2
LAYOUT_GAP = 0.08
TITLE_GAP = 0.10
ROW_GAP = 0.10
TITLE_HEIGHT = 0.13


def glyph_figure(report: Report, payload: dict[str, Any]) -> plt.Figure:
    if report.flip is None:
        raise ValueError("glyph rendering requires a flip result")
    shape = tuple(int(value) for value in payload["shape"])
    affine = np.asarray(payload["affine"], dtype=float)
    voxel_sizes = np.asarray(payload["voxel_sizes"], dtype=float)
    frame_map = np.asarray(payload["frame_map"], dtype=float)
    v1 = np.frombuffer(payload["v1"], dtype=np.float32).reshape((*shape, 3))
    fa = np.frombuffer(payload["fa"], dtype=np.float32).reshape(shape)
    s0 = np.frombuffer(payload["s0"], dtype=np.float32).reshape(shape)
    mask = np.frombuffer(payload["mask"], dtype=np.uint8).reshape(shape).astype(bool)

    axes, signs = canonical_orientation(affine)
    voxel_sizes = voxel_sizes[list(axes)]
    background = canonical_scalar(s0, axes, signs)
    fa = canonical_scalar(fa, axes, signs)
    mask = canonical_scalar(mask, axes, signs)

    foreground = background >= otsu_threshold(background)
    center = brainstem_center(foreground)
    crop, crop_center = crop_bounds(center, voxel_sizes, background.shape)
    glyph_mask = mask[crop] & (fa[crop] >= FA_THRESHOLD)
    current = canonical_vectors(v1, frame_map, axes, signs, crop)
    best_map = frame_map @ np.asarray(report.flip.best.matrix, dtype=float)
    best = canonical_vectors(v1, best_map, axes, signs, crop)
    return render_figure(
        report,
        background,
        crop,
        current,
        best,
        glyph_mask,
        center,
        crop_center,
        voxel_sizes,
    )


def canonical_orientation(
    affine: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    spatial = affine[:3, :3]
    norms = np.linalg.norm(spatial, axis=0)
    rotation = spatial / norms
    axes = max(
        itertools.permutations(range(3)),
        key=lambda permutation: sum(
            abs(rotation[world_axis, native_axis])
            for world_axis, native_axis in enumerate(permutation)
        ),
    )
    signs = tuple(
        1 if rotation[world_axis, native_axis] >= 0.0 else -1
        for world_axis, native_axis in enumerate(axes)
    )
    return axes, signs


def canonical_scalar(
    values: np.ndarray, axes: tuple[int, ...], signs: tuple[int, ...]
) -> np.ndarray:
    canonical = np.transpose(values, axes=axes)
    for axis, sign in enumerate(signs):
        if sign < 0:
            canonical = np.flip(canonical, axis=axis)
    return canonical


def canonical_vectors(
    values: np.ndarray,
    matrix: np.ndarray,
    axes: tuple[int, ...],
    signs: tuple[int, ...],
    crop: tuple[slice, slice, slice],
) -> np.ndarray:
    canonical = np.transpose(values, axes=(*axes, 3))
    for axis, sign in enumerate(signs):
        if sign < 0:
            canonical = np.flip(canonical, axis=axis)
    transformed = transform_vectors(canonical[crop], matrix)
    return transformed[..., list(axes)] * np.asarray(signs, dtype=float)


def transform_vectors(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = values @ matrix.T
    norms = np.linalg.norm(transformed, axis=-1)
    np.divide(
        transformed,
        norms[..., None],
        out=transformed,
        where=norms[..., None] > 1e-9,
    )
    return transformed


def otsu_threshold(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).ravel()
    low = float(np.min(flat))
    high = float(np.max(flat))
    if high <= low:
        return low
    histogram, edges = np.histogram(flat, bins=256, range=(low, high))
    weights = histogram.astype(float)
    indices = np.arange(256, dtype=float)
    background = np.cumsum(weights)
    foreground = flat.size - background
    background_sum = np.cumsum(weights * indices)
    total_sum = background_sum[-1]
    valid = (background > 0.0) & (foreground > 0.0)
    variance = np.full(256, -1.0)
    difference = np.zeros(256)
    difference[valid] = (
        background_sum[valid] / background[valid]
        - (total_sum - background_sum[valid]) / foreground[valid]
    )
    variance[valid] = background[valid] * foreground[valid] * difference[valid] ** 2
    index = int(np.argmax(variance))
    return float((edges[index] + edges[index + 1]) / 2.0)


def brainstem_center(foreground: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(foreground)
    if coordinates.size == 0:
        raise ValueError("the b0 foreground mask is empty")
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    return np.rint(low + np.asarray([0.5, 0.42, 0.28]) * (high - low)).astype(int)


def crop_bounds(
    center: np.ndarray, voxel_sizes: np.ndarray, shape: tuple[int, ...]
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    radius = np.maximum(1, np.ceil(RADIUS_MM / voxel_sizes).astype(int))
    start = np.maximum(0, center - radius)
    stop = np.minimum(np.asarray(shape), center + radius + 1)
    crop = tuple(
        slice(int(first), int(last)) for first, last in zip(start, stop, strict=True)
    )
    return crop, center - start


def layout_ratios() -> tuple[tuple[float, ...], tuple[float, ...]]:
    widths = (LAYOUT_GAP, 1.0, LAYOUT_GAP, 1.0, LAYOUT_GAP, 1.0, LAYOUT_GAP)
    heights = (
        TITLE_GAP,
        TITLE_HEIGHT,
        TITLE_GAP,
        1.0,
        ROW_GAP,
        1.0,
        ROW_GAP,
        1.0,
        LAYOUT_GAP,
    )
    return widths, heights


def render_figure(
    report: Report,
    background: np.ndarray,
    crop: tuple[slice, slice, slice],
    current: np.ndarray,
    best: np.ndarray,
    glyph_mask: np.ndarray,
    center: np.ndarray,
    crop_center: np.ndarray,
    voxel_sizes: np.ndarray,
) -> plt.Figure:
    assert report.flip is not None
    correction = correction_presentation(report)
    assert correction is not None
    views = (
        ("Sagittal", 0, 1, 2),
        ("Coronal", 1, 0, 2),
        ("Axial", 2, 0, 1),
    )
    positive = background[np.isfinite(background) & (background > 0.0)]
    limits = (
        float(np.percentile(positive, 2.0)) if positive.size else 0.0,
        float(np.percentile(positive, 98.0)) if positive.size else 1.0,
    )
    widths, heights = layout_ratios()
    width = 13.0
    figure = plt.figure(
        figsize=(width, width * sum(heights) / sum(widths)), facecolor="white"
    )
    grid = figure.add_gridspec(
        len(heights),
        len(widths),
        width_ratios=widths,
        height_ratios=heights,
        hspace=0.0,
        wspace=0.0,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
    )
    title_axis = figure.add_subplot(grid[1, 1:6])
    title_axis.set_axis_off()
    panel_rows = (3, 5, 7)
    panel_columns = (1, 3, 5)
    panels = [
        [figure.add_subplot(grid[row, column]) for column in panel_columns]
        for row in panel_rows
    ]
    titles = (
        "INPUT TABLE — current convention",
        correction.glyph_title,
        "ANATOMICAL LOCATION",
    )
    cropped_background = background[crop]
    for row, (view, fixed, horizontal, vertical) in enumerate(views):
        for column, (title, vectors) in enumerate(
            zip(titles[:2], (current, best), strict=True)
        ):
            add_glyph_panel(
                panels[row][column],
                cropped_background,
                vectors,
                glyph_mask,
                fixed,
                horizontal,
                vertical,
                int(crop_center[fixed]),
                voxel_sizes,
                limits,
            )
            detail = f"RAS voxel {int(center[fixed])}"
            if column == 1:
                detail = f"GradLint best: {report.flip.best.label} | {detail}"
            panels[row][column].set_title(
                f"{view} — {title}\n{detail}", fontsize=9.0, pad=4.0
            )
        add_locator_panel(
            panels[row][2],
            background,
            center,
            crop,
            fixed,
            horizontal,
            vertical,
            voxel_sizes,
            limits,
        )
        panels[row][2].set_title(f"{view} — {titles[2]}", fontsize=9.0, pad=4.0)
    title_axis.text(
        0.5,
        0.5,
        "DTI principal-direction glyph QC | "
        f"{correction.summary.rstrip('.')} | "
        f"margin {report.flip.relative_margin * 100.0:.2f}% | "
        f"b≈{report.flip.working_b:g}\n"
        "ROI: inferior midline (brainstem heuristic) | canonical RAS | "
        "color = |R|, |A|, |S|",
        fontsize=12,
        ha="center",
        va="center",
    )
    return figure


def add_glyph_panel(
    axis: plt.Axes,
    background: np.ndarray,
    vectors: np.ndarray,
    glyph_mask: np.ndarray,
    fixed_axis: int,
    horizontal_axis: int,
    vertical_axis: int,
    fixed_index: int,
    voxel_sizes: np.ndarray,
    limits: tuple[float, float],
) -> None:
    background_2d = np.take(background, fixed_index, axis=fixed_axis)
    vectors_2d = np.take(vectors, fixed_index, axis=fixed_axis)
    mask_2d = np.take(glyph_mask, fixed_index, axis=fixed_axis)
    width, height = background_2d.shape
    extent = panel_extent(width, height, horizontal_axis, vertical_axis, voxel_sizes)
    axis.imshow(
        background_2d.T,
        origin="lower",
        cmap="gray",
        vmin=limits[0],
        vmax=limits[1],
        extent=extent,
        interpolation="nearest",
    )
    strides = [
        max(1, math.floor(STRIDE_MM / voxel_sizes[index] + 0.5))
        for index in (horizontal_axis, vertical_axis)
    ]
    segments: list[list[tuple[float, float]]] = []
    colors: list[np.ndarray] = []
    for horizontal in range(0, width, strides[0]):
        for vertical in range(0, height, strides[1]):
            if not mask_2d[horizontal, vertical]:
                continue
            vector = vectors_2d[horizontal, vertical]
            projected = np.asarray(
                [vector[horizontal_axis], vector[vertical_axis]], dtype=float
            )
            if np.linalg.norm(projected) < 0.05:
                continue
            point = np.asarray(
                [
                    (horizontal + 0.5) * voxel_sizes[horizontal_axis],
                    (vertical + 0.5) * voxel_sizes[vertical_axis],
                ]
            )
            delta = 0.5 * GLYPH_LENGTH_MM * projected
            segments.append([tuple(point - delta), tuple(point + delta)])
            colors.append(np.clip(np.abs(vector), 0.0, 1.0))
    if segments:
        axis.add_collection(LineCollection(segments, colors=colors, linewidths=0.9))
    finish_panel(axis, fixed_axis, extent)


def add_locator_panel(
    axis: plt.Axes,
    background: np.ndarray,
    center: np.ndarray,
    crop: tuple[slice, slice, slice],
    fixed_axis: int,
    horizontal_axis: int,
    vertical_axis: int,
    voxel_sizes: np.ndarray,
    limits: tuple[float, float],
) -> None:
    image = np.take(background, int(center[fixed_axis]), axis=fixed_axis)
    width, height = image.shape
    extent = panel_extent(width, height, horizontal_axis, vertical_axis, voxel_sizes)
    axis.imshow(
        image.T,
        origin="lower",
        cmap="gray",
        vmin=limits[0],
        vmax=limits[1],
        extent=extent,
        interpolation="bilinear",
    )
    horizontal_crop = crop[horizontal_axis]
    vertical_crop = crop[vertical_axis]
    axis.add_patch(
        Rectangle(
            (
                horizontal_crop.start * voxel_sizes[horizontal_axis],
                vertical_crop.start * voxel_sizes[vertical_axis],
            ),
            (horizontal_crop.stop - horizontal_crop.start)
            * voxel_sizes[horizontal_axis],
            (vertical_crop.stop - vertical_crop.start) * voxel_sizes[vertical_axis],
            edgecolor="#ffcc33",
            facecolor="none",
            linewidth=1.5,
        )
    )
    axis.plot(
        (center[horizontal_axis] + 0.5) * voxel_sizes[horizontal_axis],
        (center[vertical_axis] + 0.5) * voxel_sizes[vertical_axis],
        marker="+",
        color="#ff4d4d",
        markersize=7,
        markeredgewidth=1.2,
    )
    finish_panel(axis, fixed_axis, extent)


def panel_extent(
    width: int,
    height: int,
    horizontal_axis: int,
    vertical_axis: int,
    voxel_sizes: np.ndarray,
) -> tuple[float, float, float, float]:
    return (
        0.0,
        width * voxel_sizes[horizontal_axis],
        0.0,
        height * voxel_sizes[vertical_axis],
    )


def finish_panel(
    axis: plt.Axes,
    fixed_axis: int,
    extent: tuple[float, float, float, float],
) -> None:
    width = extent[1] - extent[0]
    height = extent[3] - extent[2]
    side = max(width, height)
    horizontal_center = (extent[0] + extent[1]) / 2.0
    vertical_center = (extent[2] + extent[3]) / 2.0
    axis.set_xlim(horizontal_center - side / 2.0, horizontal_center + side / 2.0)
    axis.set_ylim(vertical_center - side / 2.0, vertical_center + side / 2.0)
    axis.set_aspect("equal")
    axis.set_box_aspect(1.0)
    axis.set_facecolor("black")
    axis.set_xticks([])
    axis.set_yticks([])
    add_orientation_labels(axis, fixed_axis)


def add_orientation_labels(axis: plt.Axes, fixed_axis: int) -> None:
    labels = {
        0: ("P", "A", "I", "S"),
        1: ("L", "R", "I", "S"),
        2: ("L", "R", "P", "A"),
    }
    left, right, bottom, top = labels[fixed_axis]
    style = {
        "color": "white",
        "fontsize": 8,
        "fontweight": "bold",
        "bbox": {
            "facecolor": "black",
            "alpha": 0.55,
            "edgecolor": "none",
            "pad": 1.5,
        },
        "transform": axis.transAxes,
        "zorder": 5,
    }
    axis.text(0.02, 0.5, left, ha="left", va="center", **style)
    axis.text(0.98, 0.5, right, ha="right", va="center", **style)
    axis.text(0.5, 0.02, bottom, ha="center", va="bottom", **style)
    axis.text(0.5, 0.98, top, ha="center", va="top", **style)
