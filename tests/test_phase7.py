from __future__ import annotations

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


def test_repair_dry_run_never_writes(dataset: dict[str, str]) -> None:
    out_bvec = os.path.join(dataset["dir"], "dry.bvec")
    out_bval = os.path.join(dataset["dir"], "dry.bval")
    report = gg.repair(
        dataset["dwi"],
        out_bvec,
        out_bval,
        bvec=dataset["bvec_flipped"],
        bval=dataset["bval"],
        dry_run=True,
    )
    assert report.flip is not None
    assert not Path(out_bvec).exists()
    assert not Path(out_bval).exists()


def test_repair_consistent_table_writes_nothing(dataset: dict[str, str]) -> None:
    out_bvec = os.path.join(dataset["dir"], "noop.bvec")
    out_bval = os.path.join(dataset["dir"], "noop.bval")
    report = gg.repair(
        dataset["dwi"],
        out_bvec,
        out_bval,
        bvec=dataset["bvec"],
        bval=dataset["bval"],
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
