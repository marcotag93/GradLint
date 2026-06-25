"""Typed Python API over the native bindings.

Each function runs a Rust core pipeline and returns a parsed :class:`Report`.
"""

from __future__ import annotations

import json
from typing import Any

from . import _gradlint
from .report.model import Report


def inspect(
    bvec: str | None = None,
    bval: str | None = None,
    *,
    grad: str | None = None,
    tolerance: float = 0.05,
    b0_threshold: float = 50.0,
    shell: float | None = None,
    strict: bool = False,
    norm_tolerance: float = 0.05,
) -> Report:
    """Scheme-only QC (shells + angular metrics); no image required."""
    return Report.from_json(
        _gradlint.inspect(
            bvec=bvec,
            bval=bval,
            grad=grad,
            tolerance=tolerance,
            b0_threshold=b0_threshold,
            shell=shell,
            strict=strict,
            norm_tolerance=norm_tolerance,
        )
    )


def audit(
    bvec: str | None = None,
    bval: str | None = None,
    *,
    grad: str | None = None,
    dwi: str | None = None,
    mask: str | None = None,
    tolerance: float = 0.05,
    b0_threshold: float = 50.0,
    shell: float | None = None,
    step: float | None = None,
    strict: bool = False,
    norm_tolerance: float = 0.05,
) -> Report:
    """Full audit: scheme QC plus flip detection when ``dwi`` is given."""
    return Report.from_json(
        _gradlint.audit(
            bvec=bvec,
            bval=bval,
            grad=grad,
            dwi=dwi,
            mask=mask,
            tolerance=tolerance,
            b0_threshold=b0_threshold,
            shell=shell,
            step=step,
            strict=strict,
            norm_tolerance=norm_tolerance,
        )
    )


def detect_flip(
    dwi: str,
    bvec: str | None = None,
    bval: str | None = None,
    *,
    grad: str | None = None,
    mask: str | None = None,
    tolerance: float = 0.05,
    b0_threshold: float = 50.0,
    shell: float | None = None,
    step: float | None = None,
) -> Report:
    """Flip detection only (requires an image): a lean scheme + flip report."""
    return Report.from_json(
        _gradlint.detect_flip(
            dwi,
            bvec=bvec,
            bval=bval,
            grad=grad,
            mask=mask,
            tolerance=tolerance,
            b0_threshold=b0_threshold,
            shell=shell,
            step=step,
        )
    )


def repair(
    dwi: str,
    out_bvec: str,
    out_bval: str,
    bvec: str | None = None,
    bval: str | None = None,
    *,
    grad: str | None = None,
    mask: str | None = None,
    out_grad: str | None = None,
    provenance: str | None = None,
    tolerance: float = 0.05,
    b0_threshold: float = 50.0,
    shell: float | None = None,
    step: float | None = None,
    dry_run: bool = False,
    in_place: bool = False,
    strict: bool = False,
    force_repair: bool = False,
    norm_tolerance: float = 0.05,
) -> Report:
    """Audit and write a corrected table when a flip is flagged."""
    return Report.from_json(
        _gradlint.repair(
            dwi,
            out_bvec,
            out_bval,
            bvec=bvec,
            bval=bval,
            grad=grad,
            mask=mask,
            out_grad=out_grad,
            provenance=provenance,
            tolerance=tolerance,
            b0_threshold=b0_threshold,
            shell=shell,
            step=step,
            dry_run=dry_run,
            in_place=in_place,
            strict=strict,
            force_repair=force_repair,
            norm_tolerance=norm_tolerance,
        )
    )


def recompute_bval(
    out_bvec: str,
    out_bval: str,
    bvec: str | None = None,
    bval: str | None = None,
    *,
    grad: str | None = None,
    out_grad: str | None = None,
    provenance: str | None = None,
    b0_threshold: float = 50.0,
    dry_run: bool = False,
    in_place: bool = False,
) -> dict[str, Any]:
    """Opt-in b-value recovery from amplitude-encoded bvecs (never run from repair)."""
    return json.loads(
        _gradlint.recompute_bval(
            out_bvec,
            out_bval,
            bvec=bvec,
            bval=bval,
            grad=grad,
            out_grad=out_grad,
            provenance=provenance,
            b0_threshold=b0_threshold,
            dry_run=dry_run,
            in_place=in_place,
        )
    )
