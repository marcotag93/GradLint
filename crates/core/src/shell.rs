//! b-value structure QC: shell clustering, b0 detection, and signal drift.

use serde::{Deserialize, Serialize};

/// Tolerance for treating a b-value as integer-valued.
pub const NON_INTEGER_TOL: f64 = 1e-3;

/// Tuning for shell detection and b0 classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellConfig {
    /// b-values at or below this are treated as b0.
    pub b0_threshold: f64,
    /// Relative tolerance for grouping b-values into one shell.
    pub tolerance: f64,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            b0_threshold: 50.0,
            tolerance: 0.05,
        }
    }
}

/// One detected shell: a cluster of volumes sharing a b-value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shell {
    /// Representative b-value (rounded cluster mean).
    pub nominal_b: f64,
    pub mean_b: f64,
    pub min_b: f64,
    pub max_b: f64,
    pub count: usize,
    pub indices: Vec<usize>,
    pub is_b0: bool,
}

/// b0 layout across the series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct B0Summary {
    pub count: usize,
    pub indices: Vec<usize>,
    /// Volume-index gaps between consecutive b0s.
    pub spacings: Vec<usize>,
}

/// Per-shell summary emitted to JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellSummary {
    pub b0_threshold: f64,
    pub tolerance: f64,
    pub shells: Vec<Shell>,
    pub b0: B0Summary,
    /// Volumes whose b-value is not integer-valued (ramp sampling / rounding).
    pub non_integer_bvals: Vec<usize>,
}

/// b0 signal drift across the series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct B0Drift {
    pub indices: Vec<usize>,
    pub mean_signal: Vec<f64>,
    /// Least-squares slope of mean signal vs volume index.
    pub slope: f64,
    /// `(max - min) / mean` of the b0 mean signals.
    pub relative_drift: f64,
}

/// Cluster b-values into shells, with b0 as its own shell.
///
/// Each cluster is anchored on its first (lowest) member and spans up to
/// `anchor·(1 + tolerance)`; the next b-value beyond that opens a new shell. This
/// is bounded and correct for discrete shells, but a long, slowly rising ramp
/// could split where a running-mean anchor would not.
#[must_use]
pub fn detect_shells(bvals: &[f64], config: ShellConfig) -> Vec<Shell> {
    let mut shells = Vec::new();

    let b0: Vec<usize> = indices_where(bvals, |b| b <= config.b0_threshold);
    if !b0.is_empty() {
        shells.push(make_shell(bvals, b0, true));
    }

    let mut dwi = indices_where(bvals, |b| b > config.b0_threshold);
    dwi.sort_by(|&a, &b| {
        bvals[a]
            .partial_cmp(&bvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut current: Vec<usize> = Vec::new();
    let mut anchor = 0.0;
    for idx in dwi {
        let b = bvals[idx];
        if current.is_empty() {
            anchor = b;
        } else if b > anchor * (1.0 + config.tolerance) {
            shells.push(make_shell(bvals, std::mem::take(&mut current), false));
            anchor = b;
        }
        current.push(idx);
    }
    if !current.is_empty() {
        shells.push(make_shell(bvals, current, false));
    }
    shells
}

/// Locate b0 volumes and the spacing between them.
#[must_use]
pub fn b0_summary(bvals: &[f64], b0_threshold: f64) -> B0Summary {
    let indices = indices_where(bvals, |b| b <= b0_threshold);
    let spacings = indices.windows(2).map(|w| w[1] - w[0]).collect();
    B0Summary {
        count: indices.len(),
        indices,
        spacings,
    }
}

/// Volumes whose b-value deviates from an integer by more than `tol`.
#[must_use]
pub fn non_integer_bvals(bvals: &[f64], tol: f64) -> Vec<usize> {
    indices_where(bvals, |b| (b - b.round()).abs() > tol)
}

/// Build the full per-shell summary.
#[must_use]
pub fn summarize(bvals: &[f64], config: ShellConfig) -> ShellSummary {
    ShellSummary {
        b0_threshold: config.b0_threshold,
        tolerance: config.tolerance,
        shells: detect_shells(bvals, config),
        b0: b0_summary(bvals, config.b0_threshold),
        non_integer_bvals: non_integer_bvals(bvals, NON_INTEGER_TOL),
    }
}

/// Fit signal drift across the b0 volumes.
#[must_use]
pub fn b0_drift(indices: &[usize], mean_signal: &[f64]) -> B0Drift {
    let n = mean_signal.len();
    let (slope, relative_drift) = if n < 2 {
        (0.0, 0.0)
    } else {
        let mean_x = indices.iter().map(|&i| i as f64).sum::<f64>() / n as f64;
        let mean_y = mean_signal.iter().sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for (&i, &y) in indices.iter().zip(mean_signal) {
            let dx = i as f64 - mean_x;
            num += dx * (y - mean_y);
            den += dx * dx;
        }
        let slope = if den > 0.0 { num / den } else { 0.0 };
        let min = mean_signal.iter().copied().fold(f64::INFINITY, f64::min);
        let max = mean_signal
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let rel = if mean_y != 0.0 {
            (max - min) / mean_y
        } else {
            0.0
        };
        (slope, rel)
    };
    B0Drift {
        indices: indices.to_vec(),
        mean_signal: mean_signal.to_vec(),
        slope,
        relative_drift,
    }
}

fn indices_where(bvals: &[f64], pred: impl Fn(f64) -> bool) -> Vec<usize> {
    bvals
        .iter()
        .enumerate()
        .filter(|&(_, &b)| pred(b))
        .map(|(i, _)| i)
        .collect()
}

fn make_shell(bvals: &[f64], mut indices: Vec<usize>, is_b0: bool) -> Shell {
    indices.sort_unstable();
    let count = indices.len();
    let mut sum = 0.0;
    let mut min_b = f64::INFINITY;
    let mut max_b = f64::NEG_INFINITY;
    for &i in &indices {
        let b = bvals[i];
        sum += b;
        min_b = min_b.min(b);
        max_b = max_b.max(b);
    }
    let mean_b = sum / count as f64;
    Shell {
        nominal_b: mean_b.round(),
        mean_b,
        min_b,
        max_b,
        count,
        indices,
        is_b0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusters_single_shell_with_b0() {
        let bvals = [0.0, 1000.0, 1000.0, 0.0, 1000.0];
        let shells = detect_shells(&bvals, ShellConfig::default());
        assert_eq!(shells.len(), 2);
        assert!(shells[0].is_b0);
        assert_eq!(shells[0].count, 2);
        assert_eq!(shells[1].count, 3);
        assert_eq!(shells[1].nominal_b, 1000.0);
    }

    #[test]
    fn separates_multishell() {
        let bvals = [0.0, 1000.0, 2000.0, 3000.0, 1000.0, 2000.0];
        let shells = detect_shells(&bvals, ShellConfig::default());
        let nominals: Vec<f64> = shells.iter().map(|s| s.nominal_b).collect();
        assert_eq!(nominals, vec![0.0, 1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn ramp_sampling_collapses_into_one_shell() {
        let bvals = [995.0, 1000.0, 1005.0, 1010.0];
        let shells = detect_shells(&bvals, ShellConfig::default());
        assert_eq!(shells.len(), 1);
        assert_eq!(shells[0].count, 4);
        assert_eq!(shells[0].nominal_b, 1003.0);
        assert_eq!(shells[0].min_b, 995.0);
        assert_eq!(shells[0].max_b, 1010.0);
    }

    #[test]
    fn b0_spacing_is_volume_gap() {
        let bvals = [0.0, 1000.0, 1000.0, 0.0, 1000.0, 1000.0, 0.0];
        let s = b0_summary(&bvals, 50.0);
        assert_eq!(s.count, 3);
        assert_eq!(s.indices, vec![0, 3, 6]);
        assert_eq!(s.spacings, vec![3, 3]);
    }

    #[test]
    fn flags_non_integer_bvals() {
        let bvals = [0.0, 1000.0, 1000.5, 2000.0];
        assert_eq!(non_integer_bvals(&bvals, NON_INTEGER_TOL), vec![2]);
    }

    #[test]
    fn drift_slope_positive_for_rising_signal() {
        let d = b0_drift(&[0, 10, 20], &[100.0, 110.0, 120.0]);
        assert!((d.slope - 1.0).abs() < 1e-9);
        assert!((d.relative_drift - 20.0 / 110.0).abs() < 1e-9);
    }

    #[test]
    fn drift_handles_single_b0() {
        let d = b0_drift(&[0], &[100.0]);
        assert_eq!(d.slope, 0.0);
        assert_eq!(d.relative_drift, 0.0);
    }

    #[test]
    fn summary_serializes_to_json() {
        let bvals = [0.0, 1000.0, 2000.0];
        let summary = summarize(&bvals, ShellConfig::default());
        let json = serde_json::to_string(&summary).unwrap();
        let back: ShellSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(summary, back);
    }
}
