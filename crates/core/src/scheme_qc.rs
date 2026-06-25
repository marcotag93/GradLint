//! Opt-in scheme-quality gating.
//!
//! Scheme metrics always emit advisory **notes**; with `strict` set, a *severe*
//! issue promotes PASS to WARN only. It never downgrades a more severe verdict
//! and never touches the flip decision.

use crate::angular::{ShellAngular, DTI_MIN_DIRECTIONS, DTI_RECOMMENDED_DIRECTIONS};
use crate::error::{Error, Result};
use crate::report::{Report, Status};

/// DTI design-matrix condition number above which a shell is noted as elevated.
const CONDITION_NOTE: f64 = 10.0;
/// Condition number above which conditioning is treated as severe.
const CONDITION_SEVERE: f64 = 50.0;
/// Relative b0 signal drift (peak-to-peak / mean) above which drift is noted.
const DRIFT_NOTE: f64 = 0.05;
/// Relative b0 signal drift treated as severe.
const DRIFT_SEVERE: f64 = 0.10;
/// Fraction of non-b0 directions deviating from unit norm above which the scheme
/// is flagged as likely amplitude-encoded (majority).
const AMPLITUDE_ENCODING_FRACTION: f64 = 0.5;
/// Default unit-norm tolerance for the amplitude-encoding check.
pub const DEFAULT_NORM_TOLERANCE: f64 = 0.05;

struct Finding {
    note: String,
    severe: bool,
}

/// Append scheme-quality notes and, under `strict`, promote PASS → WARN on a
/// severe issue. The amplitude-encoding check is stricter: it raises WARN by
/// default, becomes a hard error under `strict`, and otherwise leaves the report
/// unchanged when no metric breaches a threshold.
pub fn apply(report: Report, strict: bool) -> Result<Report> {
    let findings = collect(&report);
    let report = if findings.is_empty() {
        report
    } else {
        let severe = findings.iter().any(|f| f.severe);
        let mut report = findings
            .into_iter()
            .fold(report, |r, f| r.with_note(f.note));
        if strict && severe && report.status == Status::Pass {
            report = report
                .with_note("strict: severe scheme-quality issue(s) promoted status PASS → WARN")
                .with_status(Status::Warn);
        }
        report
    };
    amplitude_gate(report, strict)
}

/// Amplitude-encoded bvecs silently corrupt downstream tensors, so this check
/// gates by default (PASS → WARN) and hard-errors under `strict`.
fn amplitude_gate(report: Report, strict: bool) -> Result<Report> {
    let Some(ns) = report.norm_stats else {
        return Ok(report);
    };
    if ns.non_b0_count == 0 || ns.non_unit_fraction <= AMPLITUDE_ENCODING_FRACTION {
        return Ok(report);
    }
    let detail = format!(
        "{:.0}% of non-b0 directions are not unit-length (norm {:.3}–{:.3}, mean {:.3}; \
         tol {:.0}%) — likely amplitude-encoded bvecs: the b-values cannot be trusted and \
         gradlint will not auto-correct them; recover explicitly with `gradlint recompute-bval`.",
        ns.non_unit_fraction * 100.0,
        ns.norm_min,
        ns.norm_max,
        ns.norm_mean,
        ns.tolerance * 100.0
    );
    if strict {
        return Err(Error::AmplitudeEncoded(detail));
    }
    let report = report.with_note(detail);
    Ok(if report.status == Status::Pass {
        report.with_status(Status::Warn)
    } else {
        report
    })
}

fn collect(report: &Report) -> Vec<Finding> {
    let mut findings = Vec::new();
    if let Some(angular) = &report.angular {
        for shell in &angular.shells {
            angular_findings(shell, angular.duplicate_angle_deg, &mut findings);
        }
    }
    if let Some(shells) = &report.shells {
        if !shells.non_integer_bvals.is_empty() {
            findings.push(note(format!(
                "{} volume(s) have non-integer b-values (ramp sampling or rounding)",
                shells.non_integer_bvals.len()
            )));
        }
    }
    if let Some(drift) = &report.b0_drift {
        drift_finding(drift.relative_drift, &mut findings);
    }
    findings
}

fn angular_findings(shell: &ShellAngular, duplicate_angle_deg: f64, out: &mut Vec<Finding>) {
    if !shell.meets_dti_minimum {
        out.push(severe(format!(
            "shell b={:.0}: {} directions, below the DTI minimum of {} — the tensor fit is underdetermined",
            shell.nominal_b, shell.count, DTI_MIN_DIRECTIONS
        )));
    } else if !shell.meets_dti_recommended {
        out.push(note(format!(
            "shell b={:.0}: {} directions, below the recommended {} for a robust DTI fit",
            shell.nominal_b, shell.count, DTI_RECOMMENDED_DIRECTIONS
        )));
    }
    match shell.condition_number {
        None if shell.meets_dti_minimum => out.push(severe(format!(
            "shell b={:.0}: DTI design matrix is rank-deficient (degenerate gradient directions)",
            shell.nominal_b
        ))),
        Some(kappa) if kappa >= CONDITION_SEVERE => out.push(severe(format!(
            "shell b={:.0}: DTI design-matrix condition number {:.1} is very high (ill-conditioned scheme)",
            shell.nominal_b, kappa
        ))),
        Some(kappa) if kappa >= CONDITION_NOTE => out.push(note(format!(
            "shell b={:.0}: DTI design-matrix condition number {:.1} is elevated",
            shell.nominal_b, kappa
        ))),
        _ => {}
    }
    if !shell.duplicates.is_empty() {
        let antipodal = shell.duplicates.iter().filter(|d| d.antipodal).count();
        let near = shell.duplicates.len() - antipodal;
        out.push(note(format!(
            "shell b={:.0}: {antipodal} antipodal + {near} near-duplicate direction pair(s) within {duplicate_angle_deg:.1}°",
            shell.nominal_b
        )));
    }
}

fn drift_finding(relative_drift: f64, out: &mut Vec<Finding>) {
    if relative_drift >= DRIFT_SEVERE {
        out.push(severe(format!(
            "b0 signal drift is large ({:.1}% peak-to-peak) — check scanner stability",
            relative_drift * 100.0
        )));
    } else if relative_drift >= DRIFT_NOTE {
        out.push(note(format!(
            "b0 signal drift is {:.1}% peak-to-peak",
            relative_drift * 100.0
        )));
    }
}

fn note(message: String) -> Finding {
    Finding {
        note: message,
        severe: false,
    }
}

fn severe(message: String) -> Finding {
    Finding {
        note: message,
        severe: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::angular::AngularSummary;
    use crate::gradient::GradientTable;
    use crate::report::SchemeTable;
    use crate::shell::{B0Drift, B0Summary, ShellSummary};

    fn base() -> Report {
        let table = GradientTable::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0.0, 1000.0, 1000.0],
        )
        .unwrap();
        Report::new(SchemeTable::from_table(&table))
    }

    fn shell_angular(count: usize, condition_number: Option<f64>) -> ShellAngular {
        ShellAngular {
            nominal_b: 1000.0,
            count,
            electrostatic_energy: 1.0,
            condition_number,
            meets_dti_minimum: count >= DTI_MIN_DIRECTIONS,
            meets_dti_recommended: count >= DTI_RECOMMENDED_DIRECTIONS,
            meets_csd_minimum: count >= 45,
            duplicates: Vec::new(),
        }
    }

    fn with_angular(report: Report, shell: ShellAngular) -> Report {
        report.with_angular(AngularSummary {
            duplicate_angle_deg: 5.0,
            shells: vec![shell],
        })
    }

    fn with_drift(report: Report, relative_drift: f64) -> Report {
        report.with_b0_drift(B0Drift {
            indices: vec![0, 1],
            mean_signal: vec![100.0, 100.0],
            slope: 0.0,
            relative_drift,
        })
    }

    fn with_non_integer(report: Report) -> Report {
        report.with_shells(ShellSummary {
            b0_threshold: 50.0,
            tolerance: 0.05,
            shells: Vec::new(),
            b0: B0Summary {
                count: 0,
                indices: Vec::new(),
                spacings: Vec::new(),
            },
            non_integer_bvals: vec![2],
        })
    }

    fn with_norms(report: Report, non_b0_count: usize, non_unit_count: usize) -> Report {
        report.with_norm_stats(crate::gradient::NormStats {
            non_b0_count,
            non_unit_count,
            non_unit_fraction: if non_b0_count == 0 {
                0.0
            } else {
                non_unit_count as f64 / non_b0_count as f64
            },
            norm_min: 0.5,
            norm_max: 1.0,
            norm_mean: 0.7,
            tolerance: 0.05,
        })
    }

    #[test]
    fn clean_scheme_adds_no_notes() {
        let report = apply(with_angular(base(), shell_angular(64, Some(1.5))), true).unwrap();
        assert!(report.notes.is_empty());
        assert_eq!(report.status, Status::Pass);
    }

    #[test]
    fn under_recommended_is_note_only_and_never_promotes() {
        let report = apply(with_angular(base(), shell_angular(20, Some(2.0))), true).unwrap();
        assert!(report.notes.iter().any(|n| n.contains("recommended")));
        assert_eq!(report.status, Status::Pass);
    }

    #[test]
    fn below_dti_minimum_is_severe() {
        let lenient = apply(with_angular(base(), shell_angular(4, None)), false).unwrap();
        assert!(lenient.notes.iter().any(|n| n.contains("DTI minimum")));
        assert_eq!(lenient.status, Status::Pass);

        let strict = apply(with_angular(base(), shell_angular(4, None)), true).unwrap();
        assert_eq!(strict.status, Status::Warn);
        assert!(strict.notes.iter().any(|n| n.contains("PASS → WARN")));
    }

    #[test]
    fn rank_deficient_design_is_severe() {
        let strict = apply(with_angular(base(), shell_angular(8, None)), true).unwrap();
        assert!(strict.notes.iter().any(|n| n.contains("rank-deficient")));
        assert_eq!(strict.status, Status::Warn);
    }

    #[test]
    fn high_condition_number_is_severe() {
        let strict = apply(with_angular(base(), shell_angular(30, Some(80.0))), true).unwrap();
        assert!(strict.notes.iter().any(|n| n.contains("very high")));
        assert_eq!(strict.status, Status::Warn);
    }

    #[test]
    fn large_drift_is_severe() {
        let strict = apply(with_drift(base(), 0.15), true).unwrap();
        assert!(strict.notes.iter().any(|n| n.contains("drift is large")));
        assert_eq!(strict.status, Status::Warn);
    }

    #[test]
    fn non_integer_bvals_is_note_only() {
        let strict = apply(with_non_integer(base()), true).unwrap();
        assert!(strict.notes.iter().any(|n| n.contains("non-integer")));
        assert_eq!(strict.status, Status::Pass);
    }

    #[test]
    fn strict_never_lowers_a_flag() {
        let report = with_angular(base(), shell_angular(4, None)).with_status(Status::Flag);
        let strict = apply(report, true).unwrap();
        assert_eq!(strict.status, Status::Flag);
    }

    #[test]
    fn amplitude_encoding_warns_by_default() {
        let report = apply(with_norms(base(), 30, 30), false).unwrap();
        assert_eq!(report.status, Status::Warn);
        assert!(report.notes.iter().any(|n| n.contains("amplitude-encoded")));
    }

    #[test]
    fn amplitude_encoding_hard_errors_under_strict() {
        let err = apply(with_norms(base(), 30, 30), true).unwrap_err();
        assert!(matches!(err, Error::AmplitudeEncoded(_)));
    }

    #[test]
    fn unit_length_directions_do_not_warn() {
        let report = apply(with_norms(base(), 30, 0), true).unwrap();
        assert_eq!(report.status, Status::Pass);
        assert!(!report.notes.iter().any(|n| n.contains("amplitude-encoded")));
    }

    #[test]
    fn all_b0_does_not_warn() {
        let report = apply(with_norms(base(), 0, 0), true).unwrap();
        assert_eq!(report.status, Status::Pass);
    }
}
