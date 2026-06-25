# GradLint

[![CI](https://github.com/marcotag93/GradLint/actions/workflows/ci.yml/badge.svg)](https://github.com/marcotag93/GradLint/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gradlint.svg)](https://pypi.org/project/gradlint/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Fast, standalone **gradient-scheme quality control**, **b-vector flip / axis-permutation
detection**, and **auto-repair** for diffusion MRI.

GradLint is a quality-control *gate*. It runs in seconds before a diffusion pipeline starts: it catches the silent wrong-`bvec` error,
audits the b-value and gradient scheme, and — when a flip is found — emits a corrected
gradient table together with a machine-readable provenance log. The input gradient files
are never modified in place unless explicitly requested.

## Install

```bash
pip install gradlint
```

This installs the single `gradlint` command — every subcommand (`inspect`, `audit`,
`detect-flip`, `repair`, `recompute-bval`) and every flag — plus the importable `gradlint`
Python package.
The command is a thin wrapper; all computation runs in the native Rust core compiled into
the wheel. The published wheels are pure-Rust and need no C toolchain.

### Build from source

The build backend is [maturin](https://www.maturin.rs/) (pulled in automatically), so a
Rust toolchain (≥ 1.74) is the only prerequisite:

```bash
pip install ./crates/py
```

### Faster decompression (`libdeflate`)

On large, fine-resolution scans (e.g. HCP 1.25 mm, 288 volumes) the audit is bound by
gzip decompression of the DWI. The opt-in `libdeflate` feature roughly halves the
decompress stage with **byte-identical output** (same verdict, same report). It is off by
default. Enable it at build time via either toolchain:

```bash
# cargo
cargo build --release -p gradlint-cli --features libdeflate

# pip
pip install ./crates/py --config-settings=build-args="--features libdeflate"
```

The feature compiles a small C library. On ~2 mm data the audit is already a few seconds,
so most users do not need it.

### Linker note for source builds

This applies to **every** source build — `cargo`, `pip install ./crates/py`, with or
without `libdeflate`. If linking fails with `rust-lld: error: undefined symbol: getauxval`,
a non-system linker is earlier on your `PATH` (common with FSL or conda toolchains). Export
the system linker first, then re-run the install or build command:

```bash
export PATH="/usr/bin:$PATH" CC=/usr/bin/gcc CXX=/usr/bin/g++
export RUSTFLAGS="-C linker=/usr/bin/gcc -C link-self-contained=no -C linker-features=-lld"
```

The variables must be exported in the same shell before building; a fresh shell needs them
again. The pre-built PyPI wheels are unaffected — this is only relevant when compiling from
source.

## Usage

GradLint is a single command-line tool with one importable Python package behind it.
Every subcommand and flag below — including `--bids`, `--profile`, and `--figures` — is
available from the same `gradlint` install; there is no separate or feature-reduced build.

### Command line

```bash
# Inspect the gradient scheme only (no image needed)
gradlint inspect --bvec dwi.bvec --bval dwi.bval

# Full audit: scheme QC + b-vector flip detection, write a JSON report
gradlint audit --bvec dwi.bvec --bval dwi.bval --dwi dwi.nii.gz \
  --mask wm.nii.gz --report report.json

# Repair a detected flip (writes a corrected copy; never touches the input)
gradlint repair --bvec dwi.bvec --bval dwi.bval --dwi dwi.nii.gz \
  --out-bvec dwi.fixed.bvec --out-bval dwi.fixed.bval --provenance prov.json

# Opt-in: recover b-values hidden in amplitude-encoded bvec norms (separate from repair)
gradlint recompute-bval --bvec dwi.bvec --bval dwi.bval \
  --out-bvec dwi.bvec.recovered --out-bval dwi.bval.recovered

# BIDS dataset: audit every *_dwi and write derivatives under derivatives/gradlint/
gradlint audit --bids /data/my_bids

# Audit with a per-stage timing breakdown (decompress / convert / fit / coherence)
gradlint audit --bvec dwi.bvec --bval dwi.bval --dwi dwi.nii.gz --profile

# Audit and write the HTML report, standard figures, and tensor glyph QC
gradlint audit --bvec dwi.bvec --bval dwi.bval --dwi dwi.nii.gz \
  --figures qc_figures
```

`audit` and `repair` rank all 48 candidate gradient conventions by fibre coherence,
mark the current table and the best-scoring convention, and end with a one-line verdict.
When a flip is detected, the fix is reported as a per-axis remap (e.g. `x=-x, y=+y, z=+z`).
Progress streams to stderr; the machine-readable verdict goes to stdout, so the run is
safe to pipe.

### Outcomes

GradLint returns one of three verdicts. The exit code is intended for pipeline gating:
`0` means the workflow may continue, while `3` means GradLint found a WARN condition
that was not repaired and should be reviewed.

| Verdict        | Exit | Meaning                                                                                                                                                                                                                                                                              |
| -------------- | :---: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **PASS** | `0` | The gradient table is consistent with the image; no flip detected. Nothing is written.                                                                                                                                                                                               |
| **WARN** | `3` when unrepaired, `0` with `repair --force-repair` | A likely issue with low confidence — the best convention's margin is below the auto-repair threshold. The repair is withheld unless `repair --force-repair` is used. With `--strict`, a severe scheme-quality finding also raises PASS to WARN. |
| **FLAG** | `0` | A b-vector flip / axis-permutation is detected with high confidence. `repair` writes the corrected `bvec`/`bval` and a provenance log, allowing the downstream pipeline to continue with the repaired table. |

A false repair on good data is the worst outcome, so detection is deliberately
specificity-first: when the margin is ambiguous GradLint emits WARN rather than an
automatic repair.

The `FLAG` verdict marks a flagged gradient table, not a tool failure. It is the
verdict everywhere — terminal, rendered reports, and the `report.json` `status`
field — and maps to exit code `0`; use the corrected outputs from `repair`, not the
original gradients, in downstream steps. `audit` and `detect-flip` do not write a
repair, so production pipelines that should auto-correct should gate on `repair`.

**Masks.** Flip detection is most sensitive over white matter. With no `--mask`, GradLint
derives a foreground-gated FA white-matter proxy automatically. A supplied `--mask` is
used as-is; a focused WM (FA-thresholded) mask gives a wider detection margin than a loose
whole-brain mask. A mask whose grid does not match the DWI, or a gradient table whose
direction count differs from the number of DWI volumes, is rejected with a clear error —
GradLint never fits on mismatched inputs.

**Scheme quality.** Beyond flip detection, every `inspect`/`audit` emits advisory notes
for scheme issues (too few directions, an ill-conditioned DTI design matrix, duplicate
directions, non-integer b-values, b0 drift). These are advisory by default — the verdict
follows the flip decision. Pass `--strict` to gate on them: a *severe* scheme issue then
promotes PASS to WARN (it never changes a flip verdict, so it cannot trigger a repair).

**Amplitude-encoded bvecs.** Some exports leave the b-vector columns non-unit-length, with
the b-value hidden in the vector norm (`b ∝ |g|²`) while the `bval` file holds only a single
nominal value. Trusting that `bval` silently corrupts downstream tensors. GradLint reports
the non-b0 unit-norm statistics (count, fraction, min/max/mean norm) in the JSON, and when a
majority of non-b0 directions deviate from unit length it raises **WARN** (with `--strict`, a
hard error and non-zero exit) naming the likely cause. GradLint does **not** auto-correct it;
recover the b-values explicitly with `recompute-bval` (below). The `--norm-tolerance` flag
tunes the per-direction deviation threshold (default `0.05`).

**Frame handling (FSL ↔ MRtrix).** Each gradient format is interpreted in its own stored
frame: FSL `bvec` is image-relative (with the affine-determinant x-flip convention), MRtrix
`.b` is in world/scanner coordinates. `repair` reads the input in its own frame, detects the
flip there, and emits **each requested output in that format's correct frame**, converting
across formats through the image affine. So you can feed `--bvec/--bval` and ask for
`--out-grad` (a true world `.b`), or feed `--grad` and ask for `--out-bvec` (a true FSL bvec),
and both are correct — matching MRtrix's own `mrinfo -export_grad_{mrtrix,fsl}`. Same-format
output is numerically unchanged. `--dwi` (already required by `repair`) supplies the
affine. When the image's FSL and world frames genuinely diverge (positive determinant
with a multi-axis rotation, e.g. `diag(-2,-2,2)`), an advisory note is added so a
cross-format hand-off is not applied blindly. (`recompute-bval` takes no image and so
keeps same-format output.)

### Options

All options are long form: value-taking options are `--name value`, switches are bare
`--name`, and they combine on one command line as in the examples above. Run
`gradlint --help`, or `gradlint <command> --help`, for the authoritative per-command list.

**Gradient input** (one is required):

- `--bvec FILE --bval FILE` — FSL-style gradient table.
- `--grad FILE` — MRtrix `.b` gradient table (used instead of `--bvec`/`--bval`).

**Image input** (`audit`, `detect-flip`, `repair`):

- `--dwi FILE` — the diffusion-weighted NIfTI (`.nii` or `.nii.gz`).
- `--mask FILE` — optional brain/WM mask; a WM mask gives the most sensitive detection.

**Scheme-QC tuning:**

- `--tolerance F` — relative tolerance for shell clustering (default `0.05`).
- `--b0-threshold F` — b-value at/below which a volume counts as a b0 (default `50`).
- `--shell B` — working shell for the DTI fit (default: auto-selected).
- `--step VOXELS` — coherence sampling step; flip detection only (default: auto from voxel size).
- `--strict` — let a *severe* scheme finding promote PASS to WARN, and a majority amplitude-encoded bvec finding become a hard error.
- `--norm-tolerance F` — per-direction unit-norm tolerance for the amplitude-encoded bvec check (default `0.05`).

**Output:**

- `--report FILE` — write the canonical JSON report.
- `--figures DIR` — write a self-contained `report.html` plus figure PNGs. Image-based commands also write `glyphs.png`: sagittal, coronal, and axial DTI principal-direction views comparing the input table with GradLint's best candidate, plus anatomical locator views. Titles distinguish an applied correction from a withheld, previewed, dry-run, or unnecessary correction. Rendering uses matplotlib and jinja2, imported **only** when this flag is given; without it, GradLint neither transfers tensor fields to Python nor loads plotting dependencies.

**Repair only:**

- `--out-bvec FILE` / `--out-bval FILE` / `--out-grad FILE` — destinations for the corrected gradients.
- `--provenance FILE` — write a JSON provenance log of the repair.
- `--dry-run` — report what would change without writing anything.
- `--force` — allow existing corrected output paths to be overwritten (a `.bak` backup is kept).
- `--force-repair` — apply the best convention on a thin-margin WARN (write-side only; the verdict stays WARN, but the process exits `0` because a repair was produced).

**Recompute-bval only** (opt-in b-value recovery from amplitude-encoded bvecs; never run from `repair`):

- `--out-bvec FILE` / `--out-bval FILE` / `--out-grad FILE` — destinations for the recovered gradients (unit-normalized bvec + recovered bval). Defaults derive a `.recovered` sibling of the input.
- `--b0-threshold F` — b-value at/below which a volume counts as a b0 (default `50`).
- `--provenance FILE` — write a JSON provenance log of the recovery.
- `--dry-run` — report the recovery (per-shell before/after) without writing anything.
- `--force` — overwrite the input gradients (a `.bak` backup is kept).

**Batch / profiling** (`audit`):

- `--bids DIR` — discover and audit every `*_dwi` in a BIDS dataset, writing per-run and dataset reports under `derivatives/gradlint/` (both the `.bvec`/`.bval` and HCP `.bvecs`/`.bvals` spellings are recognised, and a grid-matching sibling mask is used automatically).
- `--profile` — print a per-stage timing breakdown (decompress / convert / fit / coherence); requires `--dwi`.

**Global:**

- `-h`, `--help` — usage.
- `-v`, `--version` — package and core version, plus the `libdeflate` build status (`on`/`off`).

Terminal output is coloured on a TTY and plain when piped (honours `NO_COLOR`).

### Python

```python
import gradlint as gg

report = gg.audit("dwi.bvec", "dwi.bval", dwi="dwi.nii.gz")
print(report.status, report.flip.best.label)

# Render the canonical report to self-contained HTML (figures embedded)
open("report.html", "w").write(gg.render_html(report))
```

The standard report figures can also be rendered from a previously saved
`--report` JSON, with no recomputation and no re-reading of the DWI:

```python
from pathlib import Path

from gradlint import load_report, render_html
from gradlint.report.figures import save_figures

report = load_report("report.json")          # written by `gradlint ... --report`
open("report.html", "w").write(render_html(report))
save_figures(report, Path("figures"))         # individual PNGs
```

The tensor glyph figure is intentionally generated only during an image-based
CLI run with `--figures`: its fitted tensor field is not stored in `report.json`.
In `glyphs.png`, the first column is the input table before correction, the
second visualizes GradLint's best candidate, and the third shows the crop
location in the full anatomical slice. The second-column title states whether
that candidate was applied, force-applied from WARN, withheld, shown only as a
FLAG preview, computed during a dry run, or unnecessary on PASS. The HTML report
uses the same correction status. Both glyph columns use the same voxels and
background, so differences reflect only the candidate gradient transform.

### Example notebook

A runnable [quickstart notebook](examples/quickstart.ipynb) walks through the full
workflow — inspect, audit, repair, and verify — on a public diffusion MRI example,
downloaded automatically on first run. It needs only
`pip install gradlint dipy`.

## Layout

```
crates/core   Rust core: I/O, metrics, DTI fit, coherence index, repair, BIDS batch
crates/cli    Internal reference binary over the same core (not the published CLI)
crates/py     Python bindings (PyO3 / maturin), the gradlint CLI, and report rendering
tests/        Cross-language smoke and integration tests
```

Data flows one way: the Rust core computes every metric and emits a canonical JSON
report, which the Python layer renders to Markdown/HTML. With `--figures`, the core also
returns the already-fitted tensor field needed for glyph rendering; it is not fitted a
second time. No metric logic is duplicated in Python.

## Validation

GradLint was validated against MRtrix3 `dwigradcheck` on three real multi-shell DWI
cohorts — HCP1200 (N = 105, 1.25 mm), a single-site acquisition (N = 57, 2 mm), and the
OpenNeuro multiple-sclerosis dataset ds007908 (N = 28, 1.5 mm). Each tool's native flip
check was run on the real acquired table of every subject (specificity), and on
known-good data corrupted with an injected flip across a severity ladder of Rician noise
and direction-count reduction (sensitivity).

Across the three cohorts GradLint produced **0 false repairs on 133 real gradient
tables**, recovered the injected convention on **100 % of 1049 graded corrupted cases**,
and agreed with `dwigradcheck` on the recovered orientation in **100 %** of comparisons
(antipode-aware), while running **~4–6× faster** per subject. On oblique affines, GradLint
resolves the gradient table into the image voxel frame, so its corrected b-vectors
reproduce the ground-truth orientation exactly.

Frame-aware emission was also tested on the undegraded injected-flip case from every
subject in these cohorts (N = 190). Repairs were run from format-native FSL and MRtrix
inputs, with both output formats requested. Both routes recovered the expected correction
and passed every numerical output comparison in **190/190 subjects**. Cross-format files
matched MRtrix `mrinfo` exports within **1.20e-6** per component, and the two routes agreed
within **1.0e-6** in both output formats. Same-format FSL output was byte-identical in
190/190 cases. MRtrix output was numerically identical in 190/190 and byte-identical in
190/190.

## Development

Requires a Rust toolchain (≥ 1.74), Python (≥ 3.10), and
[maturin](https://www.maturin.rs/).

```bash
# Rust
cargo test --workspace --exclude gradlint-py
cargo clippy --workspace --exclude gradlint-py -- -D warnings

# Python bindings: editable dev install into the active virtual environment
maturin develop --release -m crates/py/Cargo.toml
pytest
```

`maturin develop` rebuilds the native extension into the current virtual environment on
each call — use it while iterating on the Rust core.

## Citing GradLint

If you use GradLint in your research, please cite the software release. Citation metadata
is provided in [CITATION.cff](CITATION.cff) and is resolved automatically by GitHub's
"Cite this repository" button.

> A manuscript describing GradLint is currently in preparation. This section will be
> updated with the article citation once it is published.

## License

MIT — see [LICENSE](LICENSE).
