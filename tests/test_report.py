from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gradlint import Report, load_report, render_html, render_markdown


def sample_glyphs() -> dict[str, Any]:
    shape = (9, 9, 9)
    grid = np.indices(shape, dtype=float)
    center = (np.asarray(shape, dtype=float) - 1.0)[:, None, None, None] / 2.0
    radius = np.sqrt(np.sum((grid - center) ** 2, axis=0))
    v1 = np.zeros((*shape, 3), dtype=np.float32)
    v1[..., 0] = 1.0
    fa = np.full(shape, 0.7, dtype=np.float32)
    s0 = np.maximum(0.0, 1000.0 - radius * 120.0).astype(np.float32)
    mask = (radius <= 3.5).astype(np.uint8)
    return {
        "shape": shape,
        "affine": np.diag([-2.0, 2.0, 2.0, 1.0]).tolist(),
        "voxel_sizes": [2.0, 2.0, 2.0],
        "frame_map": np.eye(3).tolist(),
        "v1": v1.tobytes(),
        "fa": fa.tobytes(),
        "s0": s0.tobytes(),
        "mask": mask.tobytes(),
    }


def sample_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tool": "gradlint",
        "tool_version": "0.1.0",
        "timestamp": "2026-06-13T12:00:00Z",
        "status": "FLAG",
        "inputs": [
            {"path": "dwi.bvec", "sha256": "a" * 64, "bytes": 128},
        ],
        "scheme": {
            "directions": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "bvals": [0.0, 1000.0, 1000.0, 1000.0],
        },
        "shells": {
            "b0_threshold": 50.0,
            "tolerance": 0.05,
            "shells": [
                {
                    "nominal_b": 0.0,
                    "mean_b": 0.0,
                    "min_b": 0.0,
                    "max_b": 0.0,
                    "count": 1,
                    "indices": [0],
                    "is_b0": True,
                },
                {
                    "nominal_b": 1000.0,
                    "mean_b": 1000.0,
                    "min_b": 1000.0,
                    "max_b": 1000.0,
                    "count": 3,
                    "indices": [1, 2, 3],
                    "is_b0": False,
                },
            ],
            "b0": {"count": 1, "indices": [0], "spacings": []},
            "non_integer_bvals": [],
        },
        "b0_drift": {
            "indices": [0, 10, 20],
            "mean_signal": [100.0, 98.0, 96.0],
            "slope": -0.2,
            "relative_drift": 0.04,
        },
        "angular": {
            "duplicate_angle_deg": 5.0,
            "shells": [
                {
                    "nominal_b": 1000.0,
                    "count": 3,
                    "electrostatic_energy": 12.5,
                    "condition_number": 1.7,
                    "meets_dti_minimum": False,
                    "meets_dti_recommended": False,
                    "meets_csd_minimum": False,
                    "duplicates": [],
                }
            ],
        },
        "flip": {
            "working_b": 1000.0,
            "n_wm_voxels": 42,
            "ranking": [
                {
                    "label": "-x+y+z",
                    "matrix": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "is_identity": False,
                    "coherence": 0.91,
                    "n_samples": 80,
                },
                {
                    "label": "+x+y+z",
                    "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "is_identity": True,
                    "coherence": 0.60,
                    "n_samples": 80,
                },
            ],
            "best": {
                "label": "-x+y+z",
                "matrix": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "is_identity": False,
                "coherence": 0.91,
                "n_samples": 80,
            },
            "runner_up": {
                "label": "+x+y+z",
                "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "is_identity": True,
                "coherence": 0.60,
                "n_samples": 80,
            },
            "identity_coherence": 0.60,
            "margin": 0.31,
            "relative_margin": 0.34,
            "decision": "FLAG",
            "recommended_transform": [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "recommended_label": "-x+y+z",
        },
        "repair": {
            "matrix": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "label": "-x+y+z",
            "outputs": ["dwi.repaired.bvec", "dwi.repaired.bval"],
        },
        "notes": ["auto-repaired from a clear flip"],
    }


def report_with_correction_state(
    decision: str, repair_outputs: list[str] | None
) -> Report:
    data = copy.deepcopy(sample_report())
    data["status"] = decision
    data["flip"]["decision"] = decision
    if decision == "PASS":
        data["flip"]["best"] = copy.deepcopy(data["flip"]["ranking"][1])
    if repair_outputs is None:
        data.pop("repair")
    else:
        data["repair"]["outputs"] = repair_outputs
    return Report.from_dict(data)


@pytest.mark.parametrize(
    ("decision", "outputs", "kind", "glyph_title", "summary"),
    [
        (
            "PASS",
            None,
            "unchanged",
            "CURRENT TABLE — no correction needed",
            "PASS — no correction applied.",
        ),
        (
            "WARN",
            None,
            "withheld",
            "BEST CANDIDATE PREVIEW — not applied",
            "WARN — correction withheld because confidence is below the threshold.",
        ),
        (
            "WARN",
            ["fixed.bvec", "fixed.bval"],
            "force_applied",
            "CORRECTED TABLE — force-applied",
            "WARN — correction applied explicitly with --force-repair; "
            "verdict remains WARN.",
        ),
        (
            "FLAG",
            ["fixed.bvec", "fixed.bval"],
            "applied",
            "CORRECTED TABLE — applied",
            "FLAG — correction applied.",
        ),
        (
            "FLAG",
            None,
            "recommended",
            "BEST CANDIDATE PREVIEW — not applied",
            "FLAG — correction recommended but not applied.",
        ),
        (
            "WARN",
            [],
            "dry_run",
            "PROPOSED CORRECTION — dry run, not written",
            "WARN — proposed correction computed in dry-run mode; no files written.",
        ),
        (
            "FLAG",
            [],
            "dry_run",
            "PROPOSED CORRECTION — dry run, not written",
            "FLAG — proposed correction computed in dry-run mode; no files written.",
        ),
    ],
)
def test_correction_presentation_states(
    decision: str,
    outputs: list[str] | None,
    kind: str,
    glyph_title: str,
    summary: str,
) -> None:
    from gradlint.report.model import correction_presentation

    presentation = correction_presentation(
        report_with_correction_state(decision, outputs)
    )

    assert presentation is not None
    assert presentation.kind == kind
    assert presentation.glyph_title == glyph_title
    assert presentation.summary == summary


def test_scheme_warn_with_pass_flip_needs_no_gradient_correction() -> None:
    from gradlint.report.model import correction_presentation

    data = copy.deepcopy(sample_report())
    data["status"] = "WARN"
    data["flip"]["decision"] = "PASS"
    data["flip"]["best"] = copy.deepcopy(data["flip"]["ranking"][1])
    data.pop("repair")

    presentation = correction_presentation(Report.from_dict(data))

    assert presentation is not None
    assert presentation.kind == "unchanged"
    assert presentation.summary == (
        "WARN — no gradient correction needed; warning comes from scheme QC."
    )


@pytest.mark.parametrize(
    ("decision", "outputs", "summary"),
    [
        ("PASS", None, "PASS — no correction applied."),
        (
            "WARN",
            None,
            "WARN — correction withheld because confidence is below the threshold.",
        ),
        (
            "WARN",
            ["fixed.bvec", "fixed.bval"],
            "WARN — correction applied explicitly with --force-repair; "
            "verdict remains WARN.",
        ),
        ("FLAG", ["fixed.bvec", "fixed.bval"], "FLAG — correction applied."),
        ("FLAG", None, "FLAG — correction recommended but not applied."),
        (
            "FLAG",
            [],
            "FLAG — proposed correction computed in dry-run mode; no files written.",
        ),
    ],
)
def test_html_reports_correction_action(
    decision: str, outputs: list[str] | None, summary: str
) -> None:
    html = render_html(
        report_with_correction_state(decision, outputs), with_figures=False
    )

    assert "Correction status" in html
    assert summary in html


def test_model_parses_full_report() -> None:
    report = Report.from_dict(sample_report())
    assert report.status == "FLAG"
    assert report.flip is not None
    assert report.flip.best.label == "-x+y+z"
    assert len(report.flip.ranking) == 2
    assert report.shells is not None
    assert report.shells.shells[1].count == 3
    assert report.repair is not None
    assert report.repair.outputs[0] == "dwi.repaired.bvec"


def test_load_report_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(sample_report()), encoding="utf-8")
    report = load_report(path)
    assert report.tool_version == "0.1.0"
    assert report.angular is not None
    assert report.angular.shells[0].condition_number == 1.7


def test_optional_sections_default_to_none() -> None:
    minimal = {
        "schema_version": 1,
        "tool": "gradlint",
        "tool_version": "0.1.0",
        "timestamp": "2026-06-13T12:00:00Z",
        "status": "PASS",
        "scheme": {"directions": [[1.0, 0.0, 0.0]], "bvals": [1000.0]},
    }
    report = Report.from_dict(minimal)
    assert report.flip is None
    assert report.shells is None
    assert report.inputs == []


def test_markdown_contains_key_sections() -> None:
    md = render_markdown(Report.from_dict(sample_report()))
    assert "FLAG" in md
    assert "Flip / permutation detection" in md
    assert "`-x+y+z`" in md
    assert "Applied repair" in md
    assert "b-value / shell structure" in md
    assert "Angular scheme quality" in md


def test_html_without_figures_is_self_contained() -> None:
    html = render_html(Report.from_dict(sample_report()), with_figures=False)
    assert html.startswith("<!DOCTYPE html>")
    assert "gradlint · FLAG" in html
    assert "#c0271b" in html  # FLAG banner color
    assert "<table>" in html
    assert "data:image/png" not in html


def test_html_with_figures_embeds_base64_images() -> None:
    pytest.importorskip("matplotlib")
    html = render_html(Report.from_dict(sample_report()), with_figures=True)
    assert "data:image/png;base64," in html
    assert html.count("<img") >= 3


def test_build_figures_includes_margin_gauge() -> None:
    pytest.importorskip("matplotlib")
    from gradlint.report.figures import build_figures

    figures = build_figures(Report.from_dict(sample_report()))
    assert {"shells", "sphere", "drift", "coherence", "margin"} <= set(figures)
    assert all(uri.startswith("data:image/png;base64,") for uri in figures.values())


def test_glyph_figure_is_opt_in_and_embedded_in_html() -> None:
    pytest.importorskip("matplotlib")
    from gradlint.report.figures import build_figures

    report = Report.from_dict(sample_report())
    assert "glyphs" not in build_figures(report)
    figures = build_figures(report, glyphs=sample_glyphs())
    assert figures["glyphs"].startswith("data:image/png;base64,")
    html = render_html(report, glyphs=sample_glyphs())
    assert 'alt="tensor glyph orientation"' in html


@pytest.mark.parametrize(
    ("decision", "outputs", "expected"),
    [
        ("PASS", None, "CURRENT TABLE — no correction needed"),
        ("WARN", None, "BEST CANDIDATE PREVIEW — not applied"),
        (
            "WARN",
            ["fixed.bvec", "fixed.bval"],
            "CORRECTED TABLE — force-applied",
        ),
        ("FLAG", ["fixed.bvec", "fixed.bval"], "CORRECTED TABLE — applied"),
        ("FLAG", None, "BEST CANDIDATE PREVIEW — not applied"),
        ("FLAG", [], "PROPOSED CORRECTION — dry run, not written"),
    ],
)
def test_glyph_title_reports_correction_action(
    decision: str, outputs: list[str] | None, expected: str
) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from gradlint.report.glyphs import glyph_figure

    figure = glyph_figure(
        report_with_correction_state(decision, outputs), sample_glyphs()
    )
    text = "\n".join(
        [axis.get_title() for axis in figure.axes]
        + [item.get_text() for axis in figure.axes for item in axis.texts]
    )
    plt.close(figure)

    assert expected in text


def test_glyph_vectors_are_transformed_after_cropping() -> None:
    from gradlint.report.glyphs import canonical_vectors, transform_vectors

    values = np.arange(5 * 6 * 7 * 3, dtype=np.float32).reshape(5, 6, 7, 3)
    matrix = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    axes = (1, 0, 2)
    signs = (-1, 1, -1)
    crop = (slice(1, 4), slice(1, 5), slice(2, 6))

    expected = transform_vectors(values, matrix)
    expected = np.transpose(expected, axes=(*axes, 3))
    for axis, sign in enumerate(signs):
        if sign < 0:
            expected = np.flip(expected, axis=axis)
    expected = expected[..., list(axes)] * np.asarray(signs, dtype=float)

    np.testing.assert_allclose(
        canonical_vectors(values, matrix, axes, signs, crop), expected[crop]
    )


def test_force_replaces_in_place_cli_flag() -> None:
    from gradlint.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "repair",
            "--dwi",
            "dwi.nii.gz",
            "--bvec",
            "dwi.bvec",
            "--bval",
            "dwi.bval",
            "--out-bvec",
            "out.bvec",
            "--out-bval",
            "out.bval",
            "--force",
        ]
    )
    assert args.force is True
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "repair",
                "--dwi",
                "dwi.nii.gz",
                "--out-bvec",
                "out.bvec",
                "--out-bval",
                "out.bval",
                "--in-place",
            ]
        )


def test_save_figures_writes_png_files(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from gradlint.report.figures import save_figures

    written = save_figures(Report.from_dict(sample_report()), tmp_path / "figs")
    assert written and (tmp_path / "figs" / "margin.png") in written
    for path in written:
        assert path.suffix == ".png"
        assert path.read_bytes().startswith(b"\x89PNG")
