"""Render a canonical report to a Markdown summary."""

from __future__ import annotations

from .model import Report

_STATUS_BADGE = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FLAG": "❌ FLAG"}


def render_markdown(report: Report) -> str:
    """Render a report as a Markdown summary string."""
    lines: list[str] = [
        f"# gradlint report — {_STATUS_BADGE.get(report.status, report.status)}",
        "",
        f"- **Tool:** {report.tool} {report.tool_version}",
        f"- **Generated:** {report.timestamp}",
    ]
    lines += _flip_section(report)
    lines += _repair_section(report)
    lines += _shell_section(report)
    lines += _angular_section(report)
    lines += _drift_section(report)
    lines += _inputs_section(report)
    lines += _notes_section(report)
    return "\n".join(lines) + "\n"


def _flip_section(report: Report) -> list[str]:
    flip = report.flip
    if flip is None:
        return []
    return [
        "",
        "## Flip / permutation detection",
        "",
        f"- **Decision:** {flip.decision}",
        f"- **Working shell:** b≈{flip.working_b:g} ({flip.n_wm_voxels} WM voxels)",
        f"- **Best candidate:** `{flip.best.label}` "
        f"(coherence {flip.best.coherence:.4f})",
        f"- **Runner-up:** `{flip.runner_up.label}` "
        f"(coherence {flip.runner_up.coherence:.4f})",
        f"- **Margin:** {flip.margin:.4f} "
        f"(relative {flip.relative_margin * 100:.2f}%)",
    ]


def _repair_section(report: Report) -> list[str]:
    repair = report.repair
    if repair is None:
        return []
    outputs = ", ".join(f"`{o}`" for o in repair.outputs) or "—"
    return [
        "",
        "## Applied repair",
        "",
        f"- **Transform:** `{repair.label}`",
        f"- **Outputs:** {outputs}",
    ]


def _shell_section(report: Report) -> list[str]:
    if report.shells is None:
        return []
    summary = report.shells
    lines = ["", "## b-value / shell structure", "", "| Shell | b-value | Volumes |"]
    lines.append("| --- | --- | --- |")
    for shell in summary.shells:
        name = "b0" if shell.is_b0 else f"b≈{shell.nominal_b:g}"
        rng = f"{shell.min_b:g}–{shell.max_b:g}"
        lines.append(f"| {name} | {rng} | {shell.count} |")
    lines.append("")
    lines.append(f"- **b0 volumes:** {summary.b0.count} at {summary.b0.indices}")
    if summary.non_integer_bvals:
        lines.append(f"- **Non-integer b-values:** volumes {summary.non_integer_bvals}")
    return lines


def _angular_section(report: Report) -> list[str]:
    if report.angular is None or not report.angular.shells:
        return []
    lines = [
        "",
        "## Angular scheme quality",
        "",
        "| b-value | Dirs | Energy | Cond. № | DTI | CSD | Duplicates |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in report.angular.shells:
        kappa = "—" if s.condition_number is None else f"{s.condition_number:.1f}"
        if s.meets_dti_recommended:
            dti = "✓"
        elif s.meets_dti_minimum:
            dti = "min"
        else:
            dti = "✗"
        csd = "✓" if s.meets_csd_minimum else "✗"
        lines.append(
            f"| b≈{s.nominal_b:g} | {s.count} | {s.electrostatic_energy:.1f} "
            f"| {kappa} | {dti} | {csd} | {len(s.duplicates)} |"
        )
    return lines


def _drift_section(report: Report) -> list[str]:
    drift = report.b0_drift
    if drift is None:
        return []
    return [
        "",
        "## b0 signal drift",
        "",
        f"- **Slope:** {drift.slope:.4g} per volume",
        f"- **Relative drift:** {drift.relative_drift * 100:.2f}%",
    ]


def _inputs_section(report: Report) -> list[str]:
    if not report.inputs:
        return []
    lines = ["", "## Inputs", ""]
    for f in report.inputs:
        lines.append(f"- `{f.path}` — {f.bytes} bytes, sha256 `{f.sha256[:12]}…`")
    return lines


def _notes_section(report: Report) -> list[str]:
    if not report.notes:
        return []
    return ["", "## Notes", "", *[f"- {note}" for note in report.notes]]
