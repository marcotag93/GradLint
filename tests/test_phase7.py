from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

import gradlint as gg

DIRECTIONS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.7071, 0.7071, 0.0],
    [0.7071, 0.0, 0.7071],
    [0.0, 0.7071, 0.7071],
    [0.5774, 0.5774, 0.5774],
]
BVALS = [0.0] + [1000.0] * 7


def write_nifti(path: str, data: np.ndarray, voxel: float = 2.0) -> None:
    data = np.asarray(data, dtype="<f4")
    dim = ([data.ndim] + list(data.shape) + [1] * 7)[:8]
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, *[int(d) for d in dim])
    struct.pack_into("<h", hdr, 70, 16)  # datatype float32
    struct.pack_into("<h", hdr, 72, 32)  # bitpix
    struct.pack_into("<8f", hdr, 76, 1.0, voxel, voxel, voxel, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", hdr, 108, 352.0)  # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)  # scl_slope
    struct.pack_into("<h", hdr, 254, 1)  # sform_code
    # Radiological (negative-determinant) affine, so the FSL frame map is the
    # identity and the phantom — synthesized directly in voxel space — is audited
    # in its own convention (a positive determinant would add an x-flip).
    struct.pack_into("<4f", hdr, 280, -voxel, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", hdr, 296, 0.0, voxel, 0.0, 0.0)
    struct.pack_into("<4f", hdr, 312, 0.0, 0.0, voxel, 0.0)
    hdr[344:348] = b"n+1\x00"
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(data.tobytes(order="F"))


def crossing_phantom(n: int = 23, radius: float = 1.0) -> np.ndarray:
    c = (n - 1) / 2.0
    axes = [np.array(a, float) for a in ([3.0, 1.0, 2.0], [-1.0, 3.0, 1.0])]
    axes = [a / np.linalg.norm(a) for a in axes]
    tensors = [np.eye(3) * 0.3e-3 + (1.7e-3 - 0.3e-3) * np.outer(a, a) for a in axes]
    grad = np.array(DIRECTIONS, float)
    bvals = np.array(BVALS, float)
    out = np.zeros((n, n, n, len(BVALS)), dtype=float)
    for x in range(n):
        for y in range(n):
            for z in range(n):
                p = np.array([x - c, y - c, z - c])
                acc = np.zeros(len(BVALS))
                count = 0
                for a, d in zip(axes, tensors, strict=True):
                    perp = p - (p @ a) * a
                    if np.linalg.norm(perp) <= radius:
                        q = np.sum((grad @ d) * grad, axis=1)
                        acc += np.exp(-bvals * q)
                        count += 1
                if count:
                    out[x, y, z] = 1000.0 * acc / count
    return out


def write_bvec(path: str, directions: list[list[float]]) -> None:
    rows = [[d[axis] for d in directions] for axis in range(3)]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


def write_scheme(directory: str, directions: list[list[float]]) -> tuple[str, str]:
    bvec = os.path.join(directory, "dwi.bvec")
    bval = os.path.join(directory, "dwi.bval")
    write_bvec(bvec, directions)
    with open(bval, "w", encoding="utf-8") as f:
        f.write(" ".join(str(v) for v in BVALS) + "\n")
    return bvec, bval


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    d = str(tmp_path_factory.mktemp("dwi"))
    bvec, bval = write_scheme(d, DIRECTIONS)
    dwi = os.path.join(d, "dwi.nii")
    write_nifti(dwi, crossing_phantom())
    flipped = [[-x, y, z] for x, y, z in DIRECTIONS]
    bvec_flipped = os.path.join(d, "dwi_flipped.bvec")
    write_bvec(bvec_flipped, flipped)
    return {
        "dir": d,
        "bvec": bvec,
        "bval": bval,
        "dwi": dwi,
        "bvec_flipped": bvec_flipped,
    }


def test_inspect_returns_scheme_only(dataset: dict[str, str]) -> None:
    report = gg.inspect(dataset["bvec"], dataset["bval"])
    assert report.status == "PASS"
    assert report.shells is not None
    assert report.flip is None


def test_inspect_emits_scheme_note(dataset: dict[str, str]) -> None:
    report = gg.inspect(dataset["bvec"], dataset["bval"])
    # The 7-direction shell is below the recommended 30: an advisory note, but
    # not a severe issue, so the status stays PASS even without --strict.
    assert report.status == "PASS"
    assert any("recommended" in n for n in report.notes)


def test_strict_promotes_severe_scheme_to_warn(tmp_path: Path) -> None:
    # 5 directions in the b=1000 shell, below the DTI minimum of 6: a severe
    # scheme issue that --strict promotes from PASS to WARN.
    bvec = tmp_path / "few.bvec"
    bval = tmp_path / "few.bval"
    write_bvec(str(bvec), DIRECTIONS[:6])
    bval.write_text(" ".join(str(v) for v in BVALS[:6]) + "\n", encoding="utf-8")

    lenient = gg.inspect(str(bvec), str(bval))
    assert lenient.status == "PASS"
    assert any("DTI minimum" in n for n in lenient.notes)

    strict = gg.inspect(str(bvec), str(bval), strict=True)
    assert strict.status == "WARN"


def test_audit_without_image_skips_flip(dataset: dict[str, str]) -> None:
    report = gg.audit(dataset["bvec"], dataset["bval"])
    assert report.flip is None
    assert report.b0_drift is None


def test_audit_with_image_includes_flip(dataset: dict[str, str]) -> None:
    report = gg.audit(dataset["bvec"], dataset["bval"], dwi=dataset["dwi"])
    assert report.flip is not None
    assert len(report.flip.ranking) == 48
    assert report.b0_drift is not None
    # A consistent table must never be auto-repaired (specificity first).
    assert report.status != "FLAG"


def test_detect_flip_returns_ranking(dataset: dict[str, str]) -> None:
    report = gg.detect_flip(dataset["dwi"], dataset["bvec"], dataset["bval"])
    assert report.flip is not None
    assert len(report.flip.ranking) == 48


def test_margin_threshold_changes_verdict_not_candidate(
    dataset: dict[str, str], tmp_path: Path
) -> None:
    import json

    from gradlint.cli import main

    baseline = gg.detect_flip(dataset["dwi"], dataset["bvec_flipped"], dataset["bval"])
    assert baseline.flip is not None
    assert not baseline.flip.best.is_identity
    margin = baseline.flip.relative_margin
    assert 0 < margin < 1
    low_value = margin / 2
    high_value = (margin + 1) / 2
    low = gg.detect_flip(
        dataset["dwi"],
        dataset["bvec_flipped"],
        dataset["bval"],
        margin_threshold=low_value,
    )
    report_path = tmp_path / "high_threshold.json"
    exit_code = main(
        [
            "detect-flip",
            "--dwi",
            dataset["dwi"],
            "--bvec",
            dataset["bvec_flipped"],
            "--bval",
            dataset["bval"],
            "--margin-threshold",
            str(high_value),
            "--report",
            str(report_path),
        ]
    )
    high = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 3
    assert low.status == "FLAG"
    assert high["status"] == "WARN"
    assert low.flip is not None
    assert low.flip.best.label == baseline.flip.best.label
    assert high["flip"]["best"]["label"] == baseline.flip.best.label
    assert low.flip.margin_threshold == low_value
    assert high["flip"]["margin_threshold"] == high_value
    assert high["flip"]["relative_margin"] == pytest.approx(margin)

    low_bvec = tmp_path / "low.bvec"
    low_bval = tmp_path / "low.bval"
    low_repair = gg.audit(
        dwi=dataset["dwi"],
        out_bvec=str(low_bvec),
        out_bval=str(low_bval),
        bvec=dataset["bvec_flipped"],
        bval=dataset["bval"],
        margin_threshold=low_value,
        apply_repair=True,
    )
    high_bvec = tmp_path / "high.bvec"
    high_bval = tmp_path / "high.bval"
    high_repair = gg.audit(
        dwi=dataset["dwi"],
        out_bvec=str(high_bvec),
        out_bval=str(high_bval),
        bvec=dataset["bvec_flipped"],
        bval=dataset["bval"],
        margin_threshold=high_value,
        apply_repair=True,
    )
    assert low_repair.status == "FLAG"
    assert low_bvec.exists() and low_bval.exists()
    assert high_repair.status == "WARN"
    assert not high_bvec.exists() and not high_bval.exists()


def test_invalid_margin_threshold_is_rejected(dataset: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="margin threshold"):
        gg.detect_flip(
            dataset["dwi"],
            dataset["bvec_flipped"],
            dataset["bval"],
            margin_threshold=float("nan"),
        )


def test_cli_figures_include_glyph_png_and_html(
    dataset: dict[str, str], tmp_path: Path
) -> None:
    from gradlint.cli import main

    output = tmp_path / "figures"
    code = main(
        [
            "detect-flip",
            "--dwi",
            dataset["dwi"],
            "--bvec",
            dataset["bvec"],
            "--bval",
            dataset["bval"],
            "--figures",
            str(output),
        ]
    )

    assert code in {0, 3}
    assert (output / "glyphs.png").read_bytes().startswith(b"\x89PNG")
    assert 'alt="tensor glyph orientation"' in (output / "report.html").read_text(
        encoding="utf-8"
    )


def test_audit_rejects_gradient_volume_mismatch(dataset: dict[str, str]) -> None:
    short_bvec = os.path.join(dataset["dir"], "short.bvec")
    short_bval = os.path.join(dataset["dir"], "short.bval")
    write_bvec(short_bvec, DIRECTIONS[:-1])
    with open(short_bval, "w", encoding="utf-8") as f:
        f.write(" ".join(str(v) for v in BVALS[:-1]) + "\n")
    with pytest.raises(ValueError, match="volume mismatch"):
        gg.audit(short_bvec, short_bval, dwi=dataset["dwi"])


def test_audit_rejects_mask_grid_mismatch(dataset: dict[str, str]) -> None:
    mask = os.path.join(dataset["dir"], "mask_wrong.nii")
    write_nifti(mask, np.ones((3, 3, 3), dtype="<f4"))
    with pytest.raises(ValueError, match="mask grid mismatch"):
        gg.audit(dataset["bvec"], dataset["bval"], dwi=dataset["dwi"], mask=mask)


def test_audit_repair_dry_run_never_writes(dataset: dict[str, str]) -> None:
    out_bvec = os.path.join(dataset["dir"], "dry.bvec")
    out_bval = os.path.join(dataset["dir"], "dry.bval")
    report = gg.audit(
        dwi=dataset["dwi"],
        out_bvec=out_bvec,
        out_bval=out_bval,
        bvec=dataset["bvec_flipped"],
        bval=dataset["bval"],
        dry_run=True,
    )
    assert report.flip is not None
    assert not Path(out_bvec).exists()
    assert not Path(out_bval).exists()


def test_audit_repair_consistent_table_writes_nothing(dataset: dict[str, str]) -> None:
    out_bvec = os.path.join(dataset["dir"], "noop.bvec")
    out_bval = os.path.join(dataset["dir"], "noop.bval")
    report = gg.audit(
        dwi=dataset["dwi"],
        out_bvec=out_bvec,
        out_bval=out_bval,
        bvec=dataset["bvec"],
        bval=dataset["bval"],
        apply_repair=True,
    )
    assert report.status != "FLAG"
    assert report.repair is None
    assert not Path(out_bvec).exists()


def _amplitude_scheme(directory: str) -> tuple[str, str]:
    # Constant nominal bval (3000); the true weighting hides in |g| = sqrt(b / 3000).
    ratios = [1.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0]
    axes = [0, 1, 2, 0, 1, 2]
    directions = [[0.0, 0.0, 0.0]]
    for axis, ratio in zip(axes, ratios, strict=True):
        d = [0.0, 0.0, 0.0]
        d[axis] = ratio**0.5
        directions.append(d)
    bvals = [0.0] + [3000.0] * len(ratios)
    bvec = os.path.join(directory, "amp.bvec")
    bval = os.path.join(directory, "amp.bval")
    write_bvec(bvec, directions)
    with open(bval, "w", encoding="utf-8") as f:
        f.write(" ".join(str(v) for v in bvals) + "\n")
    return bvec, bval


def test_inspect_clean_table_has_unit_norms(dataset: dict[str, str]) -> None:
    report = gg.inspect(dataset["bvec"], dataset["bval"])
    assert report.status == "PASS"
    assert report.norm_stats is not None
    assert report.norm_stats.non_unit_count == 0


def test_inspect_warns_on_amplitude_encoded(tmp_path: Path) -> None:
    bvec, bval = _amplitude_scheme(str(tmp_path))
    report = gg.inspect(bvec, bval)
    assert report.status == "WARN"
    assert report.norm_stats is not None
    assert report.norm_stats.non_unit_fraction > 0.5
    assert any("amplitude-encoded" in n for n in report.notes)


def test_inspect_strict_hard_errors_on_amplitude_encoded(tmp_path: Path) -> None:
    bvec, bval = _amplitude_scheme(str(tmp_path))
    with pytest.raises(ValueError, match="amplitude-encoded"):
        gg.inspect(bvec, bval, strict=True)


def test_recompute_bval_recovers_multishell(tmp_path: Path) -> None:
    bvec, bval = _amplitude_scheme(str(tmp_path))
    out_bvec = os.path.join(str(tmp_path), "rec.bvec")
    out_bval = os.path.join(str(tmp_path), "rec.bval")
    summary = gg.recompute_bval(out_bvec, out_bval, bvec=bvec, bval=bval)
    after = sorted(s["nominal_b"] for s in summary["after"])
    assert after == [0.0, 1000.0, 2000.0, 3000.0]
    report = gg.inspect(out_bvec, out_bval)
    assert report.status == "PASS"
    assert report.norm_stats is not None
    assert report.norm_stats.non_unit_count == 0


def test_recompute_force_overwrites_inputs_and_keeps_backups(tmp_path: Path) -> None:
    from gradlint.cli import main

    bvec, bval = _amplitude_scheme(str(tmp_path))
    original_bvec = Path(bvec).read_bytes()
    original_bval = Path(bval).read_bytes()

    code = main(
        [
            "recompute-bval",
            "--bvec",
            bvec,
            "--bval",
            bval,
            "--force",
        ]
    )

    assert code == 0
    assert Path(f"{bvec}.bak").read_bytes() == original_bvec
    assert Path(f"{bval}.bak").read_bytes() == original_bval
    assert Path(bvec).read_bytes() != original_bvec
    assert Path(bval).read_bytes() != original_bval


def test_all_b0_no_false_positive(tmp_path: Path) -> None:
    bvec = os.path.join(str(tmp_path), "b0.bvec")
    bval = os.path.join(str(tmp_path), "b0.bval")
    write_bvec(bvec, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with open(bval, "w", encoding="utf-8") as f:
        f.write("0 0\n")
    report = gg.inspect(bvec, bval, strict=True)
    assert report.status == "PASS"
    assert report.norm_stats is not None
    assert report.norm_stats.non_b0_count == 0


def test_thread_pools_preserve_reports_and_repairs(
    dataset: dict[str, str], tmp_path: Path
) -> None:
    from gradlint import _gradlint

    kwargs = dict(
        dwi=dataset["dwi"],
        bvec=dataset["bvec_flipped"],
        bval=dataset["bval"],
    )
    baseline_json, default_profile = _gradlint.audit_profiled(**kwargs)
    baseline = json.loads(baseline_json)
    expected_scores = {
        row["label"]: row["coherence"] for row in baseline["flip"]["ranking"]
    }
    repaired = []
    for threads in (1, 2, 4, 8, 16, 1):
        raw, profile = _gradlint.audit_profiled(**kwargs, threads=threads)
        report = json.loads(raw)
        assert profile["threads"] == threads
        assert report.keys() == baseline.keys()
        assert report["status"] == baseline["status"]
        assert report["flip"]["decision"] == baseline["flip"]["decision"]
        assert report["flip"]["margin"] == pytest.approx(
            baseline["flip"]["margin"], abs=1e-12
        )
        scores = {row["label"]: row["coherence"] for row in report["flip"]["ranking"]}
        assert scores == pytest.approx(expected_scores, abs=1e-12)
        bvec = tmp_path / f"{len(repaired)}.bvec"
        bval = tmp_path / f"{len(repaired)}.bval"
        gg.audit(
            out_bvec=str(bvec), out_bval=str(bval), **kwargs, threads=threads,
            apply_repair=True,
        )
        repaired.append((bvec.read_bytes(), bval.read_bytes()))
        withheld_bvec = tmp_path / f"warn-{len(repaired)}.bvec"
        withheld_bval = tmp_path / f"warn-{len(repaired)}.bval"
        withheld = gg.audit(
            out_bvec=str(withheld_bvec),
            out_bval=str(withheld_bval),
            **kwargs,
            threads=threads,
            margin_threshold=1.0,
            apply_repair=True,
        )
        assert withheld.status == "WARN"
        assert not withheld_bvec.exists()
        assert not withheld_bval.exists()
    assert all(files == repaired[0] for files in repaired)
    _, profile = _gradlint.audit_profiled(**kwargs)
    assert profile["threads"] == default_profile["threads"]


def test_nthreads_cli_and_invalid_count(
    dataset: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    from gradlint.cli import main

    args = [
        "audit",
        "--bvec",
        dataset["bvec"],
        "--bval",
        dataset["bval"],
        "--dwi",
        dataset["dwi"],
        "--profile",
    ]
    for count, arguments in (
        (1, ["--nthreads", "1"] + args),
        (2, args + ["--nthreads", "2"]),
    ):
        assert main(arguments) == 0
        assert f"threads: {count}" in capsys.readouterr().out
    assert main(args + ["--nthreads", "0"]) == 1
    assert main(["audit", "--bids", dataset["dir"], "--nthreads", "0"]) == 1
    assert "positive integer" in capsys.readouterr().err
    with pytest.raises(ValueError, match="positive integer"):
        gg.audit(dataset["bvec"], dataset["bval"], threads=0)

@pytest.mark.parametrize("margin_threshold", [0.000001, 1.0])
@pytest.mark.parametrize("apply_repair", [False, True])
@pytest.mark.parametrize("force_repair", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_audit_repair_write_consent_matrix(
    dataset: dict[str, str],
    tmp_path: Path,
    margin_threshold: float,
    apply_repair: bool,
    force_repair: bool,
    dry_run: bool,
) -> None:
    from gradlint.cli import _report_exit_code

    paths = [tmp_path / name for name in ("out.bvec", "out.bval", "out.b", "prov.json")]
    before = Path(dataset["bvec_flipped"]).read_bytes()
    outputs = (
        dict(
            out_bvec=str(paths[0]),
            out_bval=str(paths[1]),
            out_grad=str(paths[2]),
            provenance=str(paths[3]),
        )
        if apply_repair or dry_run
        else {}
    )
    report = gg.audit(
        dwi=dataset["dwi"],
        **outputs,
        bvec=dataset["bvec_flipped"],
        bval=dataset["bval"],
        margin_threshold=margin_threshold,
        apply_repair=apply_repair,
        force_repair=force_repair,
        dry_run=dry_run,
        in_place=True,
        threads=2,
    )
    assert report.flip is not None
    decision = "FLAG" if margin_threshold < 1.0 else "WARN"
    assert report.flip.decision == decision
    written = apply_repair and not dry_run and (decision == "FLAG" or force_repair)
    assert all(path.exists() == written for path in paths)
    assert _report_exit_code(report) == (0 if written else 3)
    assert report.status == ("FLAG" if written and decision == "FLAG" else "WARN")
    if written:
        assert report.repair is not None
        assert len(report.repair.outputs) == 3
        np.testing.assert_array_equal(np.loadtxt(paths[1]), np.loadtxt(dataset["bval"]))
        world = np.loadtxt(paths[2])
        expected_world = np.loadtxt(paths[0]).T * [-1, 1, 1]
        np.testing.assert_allclose(world[:, :3], expected_world, atol=1e-6)
        np.testing.assert_array_equal(world[:, 3], np.loadtxt(paths[1]))
        provenance = json.loads(paths[3].read_text())
        assert any("--apply-repair" in note for note in provenance["notes"])
    else:
        assert report.repair is None or not report.repair.outputs
        assert not list(tmp_path.glob("*.bak*"))
    assert Path(dataset["bvec_flipped"]).read_bytes() == before


@pytest.mark.parametrize("apply_repair", [False, True])
@pytest.mark.parametrize("force_repair", [False, True])
def test_audit_repair_cli_write_consent(
    dataset: dict[str, str], tmp_path: Path, apply_repair: bool, force_repair: bool
) -> None:
    from gradlint.cli import main
    from gradlint.report.model import correction_presentation

    report_path = tmp_path / "report.json"
    output = tmp_path / "fixed.bvec"
    args = [
        "audit",
        "--dwi",
        dataset["dwi"],
        "--bvec",
        dataset["bvec_flipped"],
        "--bval",
        dataset["bval"],
        "--out-bvec",
        str(output),
        "--out-bval",
        str(tmp_path / "fixed.bval"),
        "--report",
        str(report_path),
        "--margin-threshold",
        "0.000001",
        "--nthreads",
        "2",
    ]
    if not apply_repair:
        start = args.index("--out-bvec")
        del args[start:start + 4]
    if apply_repair:
        args.append("--apply-repair")
    if force_repair:
        args.append("--force-repair")
    if not apply_repair and not force_repair:
        args.extend(["--figures", str(tmp_path / "figures")])
    assert main(args) == (0 if apply_repair else 3)
    assert output.exists() == apply_repair
    report = gg.load_report(report_path)
    presentation = correction_presentation(report)
    assert presentation is not None
    assert presentation.kind == ("applied" if apply_repair else "withheld")
    if not apply_repair:
        assert "--apply-repair" in presentation.summary
        assert "--apply-repair" in gg.render_html(report, with_figures=False)


@pytest.mark.parametrize("apply_repair", [False, True])
def test_audit_repair_count_error_cannot_be_overridden(
    dataset: dict[str, str], tmp_path: Path, apply_repair: bool
) -> None:
    bvec = tmp_path / "short.bvec"
    bval = tmp_path / "short.bval"
    write_bvec(str(bvec), DIRECTIONS[:-1])
    bval.write_text(" ".join(str(v) for v in BVALS[:-1]))
    with pytest.raises(ValueError, match="volume mismatch"):
        gg.audit(
            dwi=dataset["dwi"],
            out_bvec=str(tmp_path / "out.bvec") if apply_repair else None,
            out_bval=str(tmp_path / "out.bval") if apply_repair else None,
            bvec=str(bvec),
            bval=str(bval),
            apply_repair=apply_repair,
            force_repair=True,
            in_place=True,
        )
    assert not (tmp_path / "out.bvec").exists()


@pytest.mark.parametrize("apply_repair", [False, True])
@pytest.mark.parametrize("margin_threshold", [0.000001, 1.0])
def test_native_glyph_repair_respects_write_consent(
    dataset: dict[str, str],
    tmp_path: Path,
    apply_repair: bool,
    margin_threshold: float,
) -> None:
    from gradlint import _gradlint
    from gradlint.cli import _report_exit_code
    from gradlint.report.model import Report, correction_presentation

    output = tmp_path / "out.bvec"
    outputs = (
        dict(out_bvec=str(output), out_bval=str(tmp_path / "out.bval"))
        if apply_repair
        else {}
    )
    text, glyphs = _gradlint.audit_with_glyphs(
        dataset["dwi"],
        **outputs,
        bvec=dataset["bvec_flipped"],
        bval=dataset["bval"],
        apply_repair=apply_repair,
        force_repair=True,
        margin_threshold=margin_threshold,
        threads=2,
    )
    report = Report.from_json(text)
    assert glyphs
    assert output.exists() == apply_repair
    assert _report_exit_code(report) == (0 if apply_repair else 3)
    presentation = correction_presentation(report)
    assert presentation is not None
    assert presentation.kind == (
        ("applied" if margin_threshold < 1.0 else "force_applied")
        if apply_repair
        else "withheld"
    )


@pytest.mark.parametrize("extra", [[], ["--profile"], ["--profile", "--figures"]])
def test_audit_blocks_flag_without_changing_detection(
    dataset: dict[str, str], tmp_path: Path, extra: list[str]
) -> None:
    from gradlint.cli import main

    expected = gg.detect_flip(
        dataset["dwi"],
        dataset["bvec_flipped"],
        dataset["bval"],
        margin_threshold=0.000001,
        threads=2,
    )
    if "--figures" in extra:
        extra = [*extra, str(tmp_path / "figures")]
    report_path = tmp_path / "audit.json"
    assert (
        main(
            [
                "audit",
                "--dwi",
                dataset["dwi"],
                "--bvec",
                dataset["bvec_flipped"],
                "--bval",
                dataset["bval"],
                "--margin-threshold",
                "0.000001",
                "--nthreads",
                "2",
                "--report",
                str(report_path),
                *extra,
            ]
        )
        == 3
    )
    actual = gg.load_report(report_path)
    assert actual.status == "WARN"
    assert actual.flip is not None and expected.flip is not None
    assert actual.flip.decision == expected.flip.decision == "FLAG"
    assert actual.flip.best.label == expected.flip.best.label
    assert actual.flip.relative_margin == pytest.approx(expected.flip.relative_margin)
    assert {r.label: r.coherence for r in actual.flip.ranking} == pytest.approx(
        {r.label: r.coherence for r in expected.flip.ranking}, abs=1e-12
    )
    assert actual.repair is None


@pytest.mark.parametrize(
    "options, message",
    [
        ({"apply_repair": True}, "require --dwi"),
        ({"dwi": "missing.nii", "apply_repair": True}, "require --out-bvec"),
        ({"out_bvec": "unused.bvec"}, "output paths require"),
    ],
)
def test_audit_rejects_invalid_requests_before_io(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gg.audit(bvec="missing.bvec", bval="missing.bval", **options)


@pytest.mark.parametrize("mode", ["--bids", "--profile"])
@pytest.mark.parametrize(
    "option", ["--apply-repair", "--dry-run", "--force", "--force-repair"]
)
def test_audit_rejects_batch_or_profile_repair(mode: str, option: str) -> None:
    from gradlint.cli import main

    args = ["audit", mode]
    if mode == "--bids":
        args.append("missing")
    assert main([*args, option]) == 1


def test_bids_blocks_unapplied_flag(dataset: dict[str, str], tmp_path: Path) -> None:
    import shutil

    from gradlint.cli import main

    directory = tmp_path / "sub-test" / "dwi"
    directory.mkdir(parents=True)
    for key, suffix in [("dwi", "nii"), ("bvec_flipped", "bvec"), ("bval", "bval")]:
        shutil.copyfile(dataset[key], directory / f"sub-test_dwi.{suffix}")
    assert (
        main(
            [
                "audit",
                "--bids",
                str(tmp_path),
                "--margin-threshold",
                "0.000001",
                "--nthreads",
                "2",
            ]
        )
        == 3
    )
    output = tmp_path / "derivatives" / "gradlint"
    summary = json.loads((output / "dataset_summary.json").read_text())
    assert summary["status"] == "WARN"
    report = json.loads(next(output.glob("*_report.json")).read_text())
    assert report["status"] == "WARN"
    assert report["flip"]["decision"] == "FLAG"
    assert not report.get("repair")


def test_audit_cli_writes_only_with_consent_and_force_controls_backups(
    dataset: dict[str, str], tmp_path: Path
) -> None:
    from gradlint.cli import main

    bvec, bval = tmp_path / "out.bvec", tmp_path / "out.bval"
    bvec.write_text("existing output")
    before = Path(dataset["bvec_flipped"]).read_bytes()
    args = [
        "audit",
        "--dwi",
        dataset["dwi"],
        "--bvec",
        dataset["bvec_flipped"],
        "--bval",
        dataset["bval"],
        "--margin-threshold",
        "0.000001",
        "--out-bvec",
        str(bvec),
        "--out-bval",
        str(bval),
        "--nthreads",
        "2",
    ]
    assert main(args + ["--force"]) == 1
    assert bvec.read_text() == "existing output"
    assert not Path(f"{bvec}.bak").exists()
    assert main(args + ["--apply-repair"]) == 1
    assert main(args + ["--apply-repair", "--force"]) == 0
    assert Path(f"{bvec}.bak").read_text() == "existing output"
    assert Path(dataset["bvec_flipped"]).read_bytes() == before
    assert bval.exists()


@pytest.mark.parametrize("world_input", [False, True])
def test_unified_audit_cross_format(
    dataset: dict[str, str], tmp_path: Path, world_input: bool
) -> None:
    outputs = dict(
        out_bvec=str(tmp_path / "fixed.bvec"),
        out_bval=str(tmp_path / "fixed.bval"),
        out_grad=str(tmp_path / "fixed.b"),
    )
    if world_input:
        source = tmp_path / "input.b"
        directions = np.loadtxt(dataset["bvec_flipped"]).T * [-1, 1, 1]
        np.savetxt(source, np.column_stack((directions, BVALS)))
        inputs = dict(grad=str(source))
    else:
        inputs = dict(bvec=dataset["bvec_flipped"], bval=dataset["bval"])
    report = gg.audit(
        dwi=dataset["dwi"],
        **inputs,
        **outputs,
        apply_repair=True,
        margin_threshold=0.000001,
        threads=2,
    )
    assert report.status == "FLAG"
    corrected = np.loadtxt(outputs["out_bvec"]).T
    truth = np.array(DIRECTIONS)[1:]
    cosine = np.sum(corrected[1:] * truth, axis=1) / (
        np.linalg.norm(corrected[1:], axis=1) * np.linalg.norm(truth, axis=1)
    )
    np.testing.assert_allclose(np.abs(cosine), 1.0, atol=1e-6)
    world = np.loadtxt(outputs["out_grad"])
    np.testing.assert_allclose(world[:, :3], corrected * [-1, 1, 1], atol=1e-6)
    np.testing.assert_array_equal(world[:, 3], BVALS)
