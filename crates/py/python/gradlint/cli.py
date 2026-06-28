"""Command-line interface for the ``gradlint`` Python package.

The single user-facing front-end: a thin wrapper over the native bindings
exposing every subcommand and flag behind the ``gradlint`` command.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from typing import Any

from . import __version__, _gradlint
from .report.model import Report

_EXIT = {"PASS": 0, "WARN": 3, "FLAG": 0}

# Pastel xterm-256 verdict palette, shared with the Rust CLI.
_VERDICT_COLOR = {"PASS": 151, "WARN": 222, "FLAG": 210}
_REPAIR_COLOR = 117

_EPILOG = """\
flags (see `gradlint <command> -h` for the per-command list):
  gradients  --bvec FILE  --bval FILE  --grad FILE (MRtrix .b)
  image      --dwi FILE  --mask FILE   [audit/detect-flip/repair]
  scheme     --tolerance F  --b0-threshold F  --shell B  --step VOXELS  --strict
  output     --report FILE  --figures DIR (HTML report + figure PNGs)
  batch      --bids DIR (audit a whole BIDS tree)  --profile (per-stage timing) [audit]
  repair     --out-bvec FILE  --out-bval FILE  --out-grad FILE  --provenance FILE
             --dry-run  --force  --force-repair
  global     -h/--help  -v/--version

exit codes: 0 PASS/FLAG/repaired-WARN, 3 unrepaired WARN
"""

# Author / contact for `-v`/`--version`. Name mirrors pyproject authors;
# affiliation + institutional email aren't standard package-metadata fields.
_AUTHOR = "Marco Tagliaferri"
_ROLE = "PhD candidate in Cognitive Neuroscience"
_AFFILIATION = "Center for Mind/Brain Sciences (CIMeC), University of Trento, Italy"
_EMAILS = ("marco.tagliaferri@unitn.it", "marco.tagliaferri93@gmail.com")


def _version_text() -> str:
    libdeflate = "on" if "libdeflate" in _gradlint.build_features() else "off"
    emails = "  ".join(f"<{e}>" for e in _EMAILS)
    return (
        f"gradlint {__version__} (core {_gradlint.version()}, "
        f"libdeflate: {libdeflate})\n"
        "\n"
        f"Author: "
        "\n"
        f"{_AUTHOR} — {_ROLE}\n"
        f"{_AFFILIATION}\n"
        f"{emails}"
    )


@functools.lru_cache(maxsize=1)
def _color_enabled() -> bool:
    if os.environ.get("CLICOLOR_FORCE"):
        return True
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _paint(text: str, code: int) -> str:
    if _color_enabled():
        return f"\x1b[38;5;{code}m{text}\x1b[0m"
    return text


def _add_gradients(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bvec", help="FSL bvec file")
    p.add_argument("--bval", help="FSL bval file")
    p.add_argument("--grad", help="MRtrix .b table, instead of --bvec/--bval")


def _add_scheme_opts(
    p: argparse.ArgumentParser, *, step: bool = True, norm_tol: bool = True
) -> None:
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="relative b-value tolerance for shell clustering (default 0.05)",
    )
    p.add_argument(
        "--b0-threshold",
        type=float,
        default=50.0,
        help="b-values at or below this are treated as b0 (default 50)",
    )
    p.add_argument(
        "--shell",
        type=float,
        default=None,
        help="working-shell b-value for flip detection (auto if omitted)",
    )
    # --step only affects flip detection, which needs an image; omit it for inspect.
    if step:
        p.add_argument(
            "--step",
            type=float,
            default=None,
            help="coherence step in voxels (default scales ~4 mm with voxel size)",
        )
    # --norm-tolerance drives the amplitude-encoded bvec check; detect-flip skips it.
    if norm_tol:
        p.add_argument(
            "--norm-tolerance",
            type=float,
            default=0.05,
            help="unit-norm tolerance for amplitude-encoded bvec check (default 0.05)",
        )
    p.add_argument("--report", help="write the canonical report.json here")
    p.add_argument(
        "--figures",
        metavar="DIR",
        help="write a self-contained HTML report and the figure PNGs to this directory",
    )


def _add_strict(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--strict",
        action="store_true",
        help="promote severe scheme-quality issues to WARN (notes emitted regardless)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradlint",
        description="Gradient-scheme QC and b-vector flip detection for diffusion MRI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=_version_text(),
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_inspect = sub.add_parser(
        "inspect",
        help="scheme QC only (shells + angular metrics); no image needed",
        allow_abbrev=False,
    )
    _add_gradients(p_inspect)
    _add_scheme_opts(p_inspect, step=False)
    _add_strict(p_inspect)

    p_audit = sub.add_parser(
        "audit", help="scheme QC plus b-vector flip detection", allow_abbrev=False
    )
    _add_gradients(p_audit)
    p_audit.add_argument("--dwi", help="DWI NIfTI (required for flip detection)")
    p_audit.add_argument("--mask", help="white-matter / brain mask NIfTI")
    p_audit.add_argument(
        "--bids",
        metavar="DIR",
        help="audit every *_dwi under a BIDS dataset root and write derivatives",
    )
    p_audit.add_argument(
        "--profile",
        action="store_true",
        help="print a per-stage timing breakdown (decompress/convert/fit/coherence; "
        "requires --dwi)",
    )
    _add_scheme_opts(p_audit)
    _add_strict(p_audit)

    p_flip = sub.add_parser(
        "detect-flip",
        help="flip detection only (requires an image)",
        allow_abbrev=False,
    )
    p_flip.add_argument("--dwi", required=True, help="DWI NIfTI")
    _add_gradients(p_flip)
    p_flip.add_argument("--mask", help="white-matter / brain mask NIfTI")
    _add_scheme_opts(p_flip, norm_tol=False)

    p_repair = sub.add_parser(
        "repair",
        help="audit and write a corrected table when a flip is flagged",
        allow_abbrev=False,
    )
    p_repair.add_argument("--dwi", required=True, help="DWI NIfTI")
    _add_gradients(p_repair)
    p_repair.add_argument("--mask", help="white-matter / brain mask NIfTI")
    p_repair.add_argument("--out-bvec", required=True, help="corrected bvec output")
    p_repair.add_argument("--out-bval", required=True, help="corrected bval output")
    p_repair.add_argument("--out-grad", help="also write a corrected MRtrix .b table")
    p_repair.add_argument("--provenance", help="write a provenance log here")
    p_repair.add_argument(
        "--dry-run", action="store_true", help="compute the repair but write nothing"
    )
    p_repair.add_argument(
        "--force",
        action="store_true",
        help="allow existing output files to be overwritten (keeps .bak backups)",
    )
    p_repair.add_argument(
        "--force-repair",
        action="store_true",
        help="apply the correction on a WARN (thin-margin) decision too "
        "(verdict stays WARN)",
    )
    _add_scheme_opts(p_repair)
    _add_strict(p_repair)

    p_recompute = sub.add_parser(
        "recompute-bval",
        help="recover amplitude-encoded b-values into a corrected bval + unit bvec",
        allow_abbrev=False,
    )
    _add_gradients(p_recompute)
    p_recompute.add_argument(
        "--b0-threshold",
        type=float,
        default=50.0,
        help="b-values at or below this are treated as b0 (default 50)",
    )
    p_recompute.add_argument("--out-bvec", help="corrected bvec output")
    p_recompute.add_argument("--out-bval", help="corrected bval output")
    p_recompute.add_argument(
        "--out-grad", help="also write a corrected MRtrix .b table"
    )
    p_recompute.add_argument("--provenance", help="write a provenance log here")
    p_recompute.add_argument(
        "--dry-run", action="store_true", help="compute the recovery but write nothing"
    )
    p_recompute.add_argument(
        "--force",
        action="store_true",
        help="overwrite the inputs (keeps .bak backups)",
    )

    return parser


def _print_summary(report: Report) -> None:
    status = _paint(report.status, _VERDICT_COLOR.get(report.status, 245))
    print(f"status: {status}")
    if report.shells is not None:
        dwi = sum(s.count for s in report.shells.shells if not s.is_b0)
        print(
            f"scheme: {len(report.shells.shells)} shells, "
            f"{report.shells.b0.count} b0 + {dwi} DWI volumes"
        )
    if report.flip is not None:
        f = report.flip
        verdict = _paint(f.decision, _VERDICT_COLOR.get(f.decision, 245))
        print(
            f"flip: {verdict} — best {f.best.label} "
            f"(margin {f.relative_margin * 100:.2f}%, {f.n_wm_voxels} WM voxels)"
        )
    if report.repair is not None:
        out = ", ".join(report.repair.outputs) or "(dry-run, not written)"
        print(f"repair: {_paint(report.repair.label, _REPAIR_COLOR)} -> {out}")
    for note in report.notes:
        print(f"note: {note}")


def _write_figures(
    report: Report, out_dir: str, glyphs: dict[str, Any] | None = None
) -> None:
    # Lazy import: matplotlib/jinja2 load only when --figures is used.
    from pathlib import Path

    from .report.figures import build_figures, save_figure_data
    from .report.html import render_html

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    figures = build_figures(report, glyphs=glyphs)
    (dest / "report.html").write_text(
        render_html(report, figure_data=figures), encoding="utf-8"
    )
    pngs = save_figure_data(figures, dest)
    print(f"figures: {dest} ({len(pngs)} PNG + report.html)")


def _print_profile(profile: dict[str, float]) -> None:
    print("profile (seconds):")
    for stage in ("decompress", "convert", "fit", "coherence", "other", "total"):
        print(f"  {stage:<10} {profile[stage]:8.3f}")


def _emit_bids(summary_json: str) -> int:
    import json

    summary = json.loads(summary_json)
    for entry in summary["results"]:
        status = _paint(entry["status"], _VERDICT_COLOR.get(entry["status"], 245))
        print(f"{entry['name']}: {status}")
    print(f"summary: {summary['summary_path']}")
    return int(summary["exit_code"])


def _recovered_sibling(path: str) -> str:
    from pathlib import Path

    p = Path(path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}.recovered{p.suffix}"))
    return str(p.with_name(f"{p.name}.recovered"))


def _emit_recovery(summary: dict[str, Any], dry_run: bool) -> None:
    tag = " (dry-run, nothing written)" if dry_run else ""
    print(
        f"recompute-bval{tag}: b_nominal={summary['b_nominal']:.0f}, "
        f"max |g|={summary['max_norm']:.4f}"
    )

    def fmt(shells: list[dict[str, Any]]) -> str:
        return ", ".join(f"b{s['nominal_b']:.0f}×{s['count']}" for s in shells)

    print(f"  shells before: {fmt(summary['before'])}")
    print(f"  shells after:  {fmt(summary['after'])}")
    print(f"  recovered non-b0 b-values: {summary['b_min']:.0f}–{summary['b_max']:.0f}")
    for path in summary["outputs"]:
        print(f"  wrote: {path}")


def _run_recompute(args: argparse.Namespace) -> int:
    import json

    if args.force:
        if not (args.bvec and args.bval):
            print("error: --force requires --bvec/--bval inputs", file=sys.stderr)
            return 1
        out_bvec, out_bval = args.bvec, args.bval
    elif args.out_bvec and args.out_bval:
        out_bvec, out_bval = args.out_bvec, args.out_bval
    elif args.bvec and args.bval:
        out_bvec = _recovered_sibling(args.bvec)
        out_bval = _recovered_sibling(args.bval)
    else:
        print(
            "error: specify --out-bvec/--out-bval (or --bvec/--bval to derive them)",
            file=sys.stderr,
        )
        return 1
    try:
        summary = _gradlint.recompute_bval(
            out_bvec,
            out_bval,
            bvec=args.bvec,
            bval=args.bval,
            grad=args.grad,
            out_grad=args.out_grad,
            provenance=args.provenance,
            b0_threshold=args.b0_threshold,
            dry_run=args.dry_run,
            in_place=args.force,
        )
    except Exception as exc:  # native bindings raise on bad input / IO
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _emit_recovery(json.loads(summary), args.dry_run)
    return 0


def _emit(
    text: str,
    report_path: str | None,
    figures_dir: str | None,
    glyphs: dict[str, Any] | None = None,
) -> int:
    report = Report.from_json(text)
    _print_summary(report)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"report: {report_path}")
    if figures_dir:
        _write_figures(report, figures_dir, glyphs)
    return _report_exit_code(report)


def _report_exit_code(report: Report) -> int:
    if report.status == "WARN":
        # Unrepaired WARN halts the pipeline (exit 3); a WARN whose correction
        # was force-applied to disk continues (exit 0)
        applied = report.repair is not None and bool(report.repair.outputs)
        return _EXIT["PASS"] if applied else _EXIT["WARN"]
    return _EXIT.get(report.status, 1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    glyphs: dict[str, Any] | None = None
    try:
        if args.command == "recompute-bval":
            return _run_recompute(args)
        if args.command == "inspect":
            text = _gradlint.inspect(
                bvec=args.bvec,
                bval=args.bval,
                grad=args.grad,
                tolerance=args.tolerance,
                b0_threshold=args.b0_threshold,
                shell=args.shell,
                strict=args.strict,
                norm_tolerance=args.norm_tolerance,
            )
        elif args.command == "audit":
            if args.bids:
                summary = _gradlint.audit_bids(
                    args.bids,
                    tolerance=args.tolerance,
                    b0_threshold=args.b0_threshold,
                    shell=args.shell,
                    step=args.step,
                    strict=args.strict,
                    norm_tolerance=args.norm_tolerance,
                )
                return _emit_bids(summary)
            if args.profile:
                if not args.dwi:
                    print("error: --profile requires --dwi", file=sys.stderr)
                    return 1
                audit_profiled = (
                    _gradlint.audit_profiled_with_glyphs
                    if args.figures
                    else _gradlint.audit_profiled
                )
                result = audit_profiled(
                    args.dwi,
                    bvec=args.bvec,
                    bval=args.bval,
                    grad=args.grad,
                    mask=args.mask,
                    tolerance=args.tolerance,
                    b0_threshold=args.b0_threshold,
                    shell=args.shell,
                    step=args.step,
                    strict=args.strict,
                    norm_tolerance=args.norm_tolerance,
                )
                if args.figures:
                    text, profile, glyphs = result
                else:
                    text, profile = result
                rc = _emit(text, args.report, args.figures, glyphs)
                _print_profile(profile)
                return rc
            if args.figures and args.dwi:
                text, glyphs = _gradlint.audit_with_glyphs(
                    args.dwi,
                    bvec=args.bvec,
                    bval=args.bval,
                    grad=args.grad,
                    mask=args.mask,
                    tolerance=args.tolerance,
                    b0_threshold=args.b0_threshold,
                    shell=args.shell,
                    step=args.step,
                    strict=args.strict,
                    norm_tolerance=args.norm_tolerance,
                )
            else:
                text = _gradlint.audit(
                    bvec=args.bvec,
                    bval=args.bval,
                    grad=args.grad,
                    dwi=args.dwi,
                    mask=args.mask,
                    tolerance=args.tolerance,
                    b0_threshold=args.b0_threshold,
                    shell=args.shell,
                    step=args.step,
                    strict=args.strict,
                    norm_tolerance=args.norm_tolerance,
                )
        elif args.command == "detect-flip":
            detect_flip = (
                _gradlint.detect_flip_with_glyphs
                if args.figures
                else _gradlint.detect_flip
            )
            result = detect_flip(
                args.dwi,
                bvec=args.bvec,
                bval=args.bval,
                grad=args.grad,
                mask=args.mask,
                tolerance=args.tolerance,
                b0_threshold=args.b0_threshold,
                shell=args.shell,
                step=args.step,
            )
            if args.figures:
                text, glyphs = result
            else:
                text = result
        else:  # repair
            repair = _gradlint.repair_with_glyphs if args.figures else _gradlint.repair
            result = repair(
                args.dwi,
                args.out_bvec,
                args.out_bval,
                bvec=args.bvec,
                bval=args.bval,
                grad=args.grad,
                mask=args.mask,
                out_grad=args.out_grad,
                provenance=args.provenance,
                tolerance=args.tolerance,
                b0_threshold=args.b0_threshold,
                shell=args.shell,
                step=args.step,
                dry_run=args.dry_run,
                in_place=args.force,
                strict=args.strict,
                force_repair=args.force_repair,
                norm_tolerance=args.norm_tolerance,
            )
            if args.figures:
                text, glyphs = result
            else:
                text = result
    except Exception as exc:  # native bindings raise on bad input / IO
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return _emit(text, args.report, args.figures, glyphs)


if __name__ == "__main__":
    sys.exit(main())
