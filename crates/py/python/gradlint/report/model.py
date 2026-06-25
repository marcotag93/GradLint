"""Typed, parse-only view of the canonical ``report.json`` emitted by the core."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Matrix = list[list[float]]
Vec3 = list[float]


@dataclass(frozen=True)
class InputFile:
    path: str
    sha256: str
    bytes: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InputFile:
        return cls(path=d["path"], sha256=d["sha256"], bytes=int(d["bytes"]))


@dataclass(frozen=True)
class SchemeTable:
    directions: list[Vec3]
    bvals: list[float]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SchemeTable:
        return cls(
            directions=[[float(c) for c in v] for v in d["directions"]],
            bvals=[float(b) for b in d["bvals"]],
        )


@dataclass(frozen=True)
class Shell:
    nominal_b: float
    mean_b: float
    min_b: float
    max_b: float
    count: int
    indices: list[int]
    is_b0: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Shell:
        return cls(
            nominal_b=float(d["nominal_b"]),
            mean_b=float(d["mean_b"]),
            min_b=float(d["min_b"]),
            max_b=float(d["max_b"]),
            count=int(d["count"]),
            indices=[int(i) for i in d["indices"]],
            is_b0=bool(d["is_b0"]),
        )


@dataclass(frozen=True)
class B0Summary:
    count: int
    indices: list[int]
    spacings: list[int]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> B0Summary:
        return cls(
            count=int(d["count"]),
            indices=[int(i) for i in d["indices"]],
            spacings=[int(s) for s in d["spacings"]],
        )


@dataclass(frozen=True)
class ShellSummary:
    b0_threshold: float
    tolerance: float
    shells: list[Shell]
    b0: B0Summary
    non_integer_bvals: list[int]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShellSummary:
        return cls(
            b0_threshold=float(d["b0_threshold"]),
            tolerance=float(d["tolerance"]),
            shells=[Shell.from_dict(s) for s in d["shells"]],
            b0=B0Summary.from_dict(d["b0"]),
            non_integer_bvals=[int(i) for i in d["non_integer_bvals"]],
        )


@dataclass(frozen=True)
class B0Drift:
    indices: list[int]
    mean_signal: list[float]
    slope: float
    relative_drift: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> B0Drift:
        return cls(
            indices=[int(i) for i in d["indices"]],
            mean_signal=[float(s) for s in d["mean_signal"]],
            slope=float(d["slope"]),
            relative_drift=float(d["relative_drift"]),
        )


@dataclass(frozen=True)
class DuplicatePair:
    i: int
    j: int
    angle_deg: float
    antipodal: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DuplicatePair:
        return cls(
            i=int(d["i"]),
            j=int(d["j"]),
            angle_deg=float(d["angle_deg"]),
            antipodal=bool(d["antipodal"]),
        )


@dataclass(frozen=True)
class ShellAngular:
    nominal_b: float
    count: int
    electrostatic_energy: float
    condition_number: float | None
    meets_dti_minimum: bool
    meets_dti_recommended: bool
    meets_csd_minimum: bool
    duplicates: list[DuplicatePair]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShellAngular:
        kappa = d.get("condition_number")
        return cls(
            nominal_b=float(d["nominal_b"]),
            count=int(d["count"]),
            electrostatic_energy=float(d["electrostatic_energy"]),
            condition_number=None if kappa is None else float(kappa),
            meets_dti_minimum=bool(d["meets_dti_minimum"]),
            meets_dti_recommended=bool(d["meets_dti_recommended"]),
            meets_csd_minimum=bool(d["meets_csd_minimum"]),
            duplicates=[DuplicatePair.from_dict(p) for p in d["duplicates"]],
        )


@dataclass(frozen=True)
class AngularSummary:
    duplicate_angle_deg: float
    shells: list[ShellAngular]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AngularSummary:
        return cls(
            duplicate_angle_deg=float(d["duplicate_angle_deg"]),
            shells=[ShellAngular.from_dict(s) for s in d["shells"]],
        )


@dataclass(frozen=True)
class CandidateScore:
    label: str
    matrix: Matrix
    is_identity: bool
    coherence: float
    n_samples: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateScore:
        return cls(
            label=d["label"],
            matrix=[[float(c) for c in row] for row in d["matrix"]],
            is_identity=bool(d["is_identity"]),
            coherence=float(d["coherence"]),
            n_samples=int(d["n_samples"]),
        )


@dataclass(frozen=True)
class FlipResult:
    working_b: float
    n_wm_voxels: int
    ranking: list[CandidateScore]
    best: CandidateScore
    runner_up: CandidateScore
    identity_coherence: float
    margin: float
    relative_margin: float
    decision: str
    recommended_transform: Matrix | None
    recommended_label: str | None
    mask_mean_fa: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FlipResult:
        transform = d.get("recommended_transform")
        return cls(
            working_b=float(d["working_b"]),
            n_wm_voxels=int(d["n_wm_voxels"]),
            mask_mean_fa=float(d.get("mask_mean_fa", 0.0)),
            ranking=[CandidateScore.from_dict(c) for c in d["ranking"]],
            best=CandidateScore.from_dict(d["best"]),
            runner_up=CandidateScore.from_dict(d["runner_up"]),
            identity_coherence=float(d["identity_coherence"]),
            margin=float(d["margin"]),
            relative_margin=float(d["relative_margin"]),
            decision=d["decision"],
            recommended_transform=(
                None
                if transform is None
                else [[float(c) for c in row] for row in transform]
            ),
            recommended_label=d.get("recommended_label"),
        )


@dataclass(frozen=True)
class RepairInfo:
    matrix: Matrix
    label: str
    outputs: list[str]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RepairInfo:
        return cls(
            matrix=[[float(c) for c in row] for row in d["matrix"]],
            label=d["label"],
            outputs=list(d["outputs"]),
        )


@dataclass(frozen=True)
class NormStats:
    non_b0_count: int
    non_unit_count: int
    non_unit_fraction: float
    norm_min: float
    norm_max: float
    norm_mean: float
    tolerance: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormStats:
        return cls(
            non_b0_count=int(d["non_b0_count"]),
            non_unit_count=int(d["non_unit_count"]),
            non_unit_fraction=float(d["non_unit_fraction"]),
            norm_min=float(d["norm_min"]),
            norm_max=float(d["norm_max"]),
            norm_mean=float(d["norm_mean"]),
            tolerance=float(d["tolerance"]),
        )


@dataclass(frozen=True)
class Report:
    schema_version: int
    tool: str
    tool_version: str
    timestamp: str
    status: str
    scheme: SchemeTable
    inputs: list[InputFile] = field(default_factory=list)
    shells: ShellSummary | None = None
    b0_drift: B0Drift | None = None
    angular: AngularSummary | None = None
    flip: FlipResult | None = None
    repair: RepairInfo | None = None
    norm_stats: NormStats | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Report:
        return cls(
            schema_version=int(d["schema_version"]),
            tool=d["tool"],
            tool_version=d["tool_version"],
            timestamp=d["timestamp"],
            status=d["status"],
            scheme=SchemeTable.from_dict(d["scheme"]),
            inputs=[InputFile.from_dict(i) for i in d.get("inputs", [])],
            shells=_opt(d.get("shells"), ShellSummary.from_dict),
            b0_drift=_opt(d.get("b0_drift"), B0Drift.from_dict),
            angular=_opt(d.get("angular"), AngularSummary.from_dict),
            flip=_opt(d.get("flip"), FlipResult.from_dict),
            repair=_opt(d.get("repair"), RepairInfo.from_dict),
            norm_stats=_opt(d.get("norm_stats"), NormStats.from_dict),
            notes=list(d.get("notes", [])),
        )

    @classmethod
    def from_json(cls, text: str) -> Report:
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class CorrectionPresentation:
    kind: str
    glyph_title: str
    summary: str


def correction_presentation(report: Report) -> CorrectionPresentation | None:
    if report.flip is None:
        return None

    decision = report.flip.decision
    repair = report.repair
    if decision == "PASS":
        summary = "PASS — no correction applied."
        if report.status == "WARN":
            summary = (
                "WARN — no gradient correction needed; warning comes from scheme QC."
            )
        return CorrectionPresentation(
            kind="unchanged",
            glyph_title="CURRENT TABLE — no correction needed",
            summary=summary,
        )
    if repair is not None and not repair.outputs:
        return CorrectionPresentation(
            kind="dry_run",
            glyph_title="PROPOSED CORRECTION — dry run, not written",
            summary=(
                f"{decision} — proposed correction computed in dry-run mode; "
                "no files written."
            ),
        )
    if repair is not None and decision == "WARN":
        return CorrectionPresentation(
            kind="force_applied",
            glyph_title="CORRECTED TABLE — force-applied",
            summary=(
                "WARN — correction applied explicitly with --force-repair; "
                "verdict remains WARN."
            ),
        )
    if repair is not None:
        return CorrectionPresentation(
            kind="applied",
            glyph_title="CORRECTED TABLE — applied",
            summary="FLAG — correction applied.",
        )
    if decision == "WARN":
        return CorrectionPresentation(
            kind="withheld",
            glyph_title="BEST CANDIDATE PREVIEW — not applied",
            summary=(
                "WARN — correction withheld because confidence is below the threshold."
            ),
        )
    return CorrectionPresentation(
        kind="recommended",
        glyph_title="BEST CANDIDATE PREVIEW — not applied",
        summary="FLAG — correction recommended but not applied.",
    )


def load_report(path: str | Path) -> Report:
    """Load and parse a canonical ``report.json`` from disk."""
    return Report.from_json(Path(path).read_text(encoding="utf-8"))


def _opt(value: dict[str, Any] | None, parse: Any) -> Any:
    return None if value is None else parse(value)
