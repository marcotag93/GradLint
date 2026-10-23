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
    threads: int | None = None,
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
            threads=threads,
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
    margin_threshold: float = 0.02,
    strict: bool = False,
    norm_tolerance: float = 0.05,
    threads: int | None = None,
    apply_repair: bool = False,
    force_repair: bool = False,
    dry_run: bool = False,
    in_place: bool = False,
    out_bvec: str | None = None,
    out_bval: str | None = None,
    out_grad: str | None = None,
    provenance: str | None = None,
) -> Report:
    """Audit, withholding nonidentity corrections unless explicitly authorized."""
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
            margin_threshold=margin_threshold,
            strict=strict,
            norm_tolerance=norm_tolerance,
            threads=threads,
            apply_repair=apply_repair,
            force_repair=force_repair,
            dry_run=dry_run,
            in_place=in_place,
            out_bvec=out_bvec,
            out_bval=out_bval,
            out_grad=out_grad,
            provenance=provenance,
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
    margin_threshold: float = 0.02,
    threads: int | None = None,
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
            margin_threshold=margin_threshold,
            threads=threads,
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
    threads: int | None = None,
) -> dict[str, Any]:
    """Opt-in b-value recovery from amplitude-encoded bvecs (never run from audit)."""
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
            threads=threads,
        )
    )
