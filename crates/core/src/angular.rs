//! Angular scheme QC: per-shell uniformity, conditioning, and duplicates.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::gradient::{norm, GradientTable};
use crate::shell::{detect_shells, Shell, ShellConfig};

/// Mathematical minimum directions for a DTI fit (tensor has 6 DOF).
pub const DTI_MIN_DIRECTIONS: usize = 6;
/// Commonly recommended minimum for a robust DTI fit.
pub const DTI_RECOMMENDED_DIRECTIONS: usize = 30;
/// Commonly cited minimum for CSD (lmax 8).
pub const CSD_MIN_DIRECTIONS: usize = 45;

/// Smallest separation / singular value treated as nonzero.
const MIN_SEPARATION: f64 = 1e-9;

/// Tuning for angular QC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngularConfig {
    /// Acute angle below which two directions are flagged as duplicates.
    pub duplicate_angle_deg: f64,
}

impl Default for AngularConfig {
    fn default() -> Self {
        Self {
            duplicate_angle_deg: 5.0,
        }
    }
}

/// A pair of near-collinear directions within one shell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DuplicatePair {
    pub i: usize,
    pub j: usize,
    pub angle_deg: f64,
    /// `true` when the raw vectors point opposite ways (antipodal redundancy).
    pub antipodal: bool,
}

/// Angular metrics for a single non-b0 shell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellAngular {
    pub nominal_b: f64,
    pub count: usize,
    /// Electrostatic repulsion energy (lower is more uniform).
    pub electrostatic_energy: f64,
    /// Condition number of the DTI design matrix; `None` if rank-deficient.
    pub condition_number: Option<f64>,
    pub meets_dti_minimum: bool,
    pub meets_dti_recommended: bool,
    pub meets_csd_minimum: bool,
    pub duplicates: Vec<DuplicatePair>,
}

/// Angular QC across all non-b0 shells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AngularSummary {
    pub duplicate_angle_deg: f64,
    pub shells: Vec<ShellAngular>,
}

/// Run angular QC per shell over the unit-normalized directions.
#[must_use]
pub fn analyze(
    table: &GradientTable,
    shell_config: ShellConfig,
    config: AngularConfig,
) -> AngularSummary {
    let unit = table.normalized(shell_config.b0_threshold);
    let shells = detect_shells(&table.bvals, shell_config);
    let shell_metrics = shells
        .iter()
        .filter(|s| !s.is_b0)
        .map(|s| analyze_shell(&unit.directions, s, config))
        .collect();
    AngularSummary {
        duplicate_angle_deg: config.duplicate_angle_deg,
        shells: shell_metrics,
    }
}

fn analyze_shell(unit_dirs: &[[f64; 3]], shell: &Shell, config: AngularConfig) -> ShellAngular {
    let dirs: Vec<[f64; 3]> = shell.indices.iter().map(|&i| unit_dirs[i]).collect();
    let count = dirs.len();
    ShellAngular {
        nominal_b: shell.nominal_b,
        count,
        electrostatic_energy: electrostatic_energy(&dirs),
        condition_number: design_condition_number(&dirs),
        meets_dti_minimum: count >= DTI_MIN_DIRECTIONS,
        meets_dti_recommended: count >= DTI_RECOMMENDED_DIRECTIONS,
        meets_csd_minimum: count >= CSD_MIN_DIRECTIONS,
        duplicates: duplicate_pairs(&shell.indices, &dirs, config.duplicate_angle_deg),
    }
}

/// Electrostatic energy `Σ 1/|qᵢ−qⱼ| + 1/|qᵢ+qⱼ|` over unit directions.
#[must_use]
pub fn electrostatic_energy(dirs: &[[f64; 3]]) -> f64 {
    let mut energy = 0.0;
    for i in 0..dirs.len() {
        for j in (i + 1)..dirs.len() {
            let minus = norm(sub(dirs[i], dirs[j])).max(MIN_SEPARATION);
            let plus = norm(add(dirs[i], dirs[j])).max(MIN_SEPARATION);
            energy += 1.0 / minus + 1.0 / plus;
        }
    }
    energy
}

/// Condition number of the 6-column DTI design matrix; `None` if rank-deficient.
#[must_use]
pub fn design_condition_number(dirs: &[[f64; 3]]) -> Option<f64> {
    if dirs.len() < DTI_MIN_DIRECTIONS {
        return None;
    }
    let mut rows = Vec::with_capacity(dirs.len() * 6);
    for &[x, y, z] in dirs {
        rows.extend_from_slice(&[x * x, y * y, z * z, 2.0 * x * y, 2.0 * x * z, 2.0 * y * z]);
    }
    let matrix = DMatrix::from_row_slice(dirs.len(), 6, &rows);
    let sv = matrix.singular_values();
    let s_max = sv[0];
    let s_min = sv[sv.len() - 1];
    if s_min <= MIN_SEPARATION {
        None
    } else {
        Some(s_max / s_min)
    }
}

fn duplicate_pairs(indices: &[usize], dirs: &[[f64; 3]], threshold_deg: f64) -> Vec<DuplicatePair> {
    let cos_threshold = threshold_deg.to_radians().cos();
    let mut pairs = Vec::new();
    for i in 0..dirs.len() {
        for j in (i + 1)..dirs.len() {
            let d = dot(dirs[i], dirs[j]);
            let abs = d.abs().min(1.0);
            if abs > cos_threshold {
                pairs.push(DuplicatePair {
                    i: indices[i],
                    j: indices[j],
                    angle_deg: abs.acos().to_degrees(),
                    antipodal: d < 0.0,
                });
            }
        }
    }
    pairs
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

    fn dti6() -> Vec<[f64; 3]> {
        vec![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [SQRT_HALF, SQRT_HALF, 0.0],
            [SQRT_HALF, 0.0, SQRT_HALF],
            [0.0, SQRT_HALF, SQRT_HALF],
        ]
    }

    #[test]
    fn clustered_directions_have_higher_energy() {
        let spread = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let clustered = [[1.0, 0.0, 0.0], [0.999, 0.044, 0.0], [0.999, 0.0, 0.044]];
        assert!(electrostatic_energy(&clustered) > electrostatic_energy(&spread));
    }

    #[test]
    fn condition_number_finite_for_well_posed_set() {
        let kappa = design_condition_number(&dti6()).unwrap();
        assert!(kappa.is_finite() && kappa > 0.0);
    }

    #[test]
    fn condition_number_none_when_underdetermined() {
        let dirs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(design_condition_number(&dirs).is_none());
    }

    #[test]
    fn condition_number_none_for_coplanar_set() {
        let dirs = vec![[1.0, 0.0, 0.0]; 6]
            .into_iter()
            .enumerate()
            .map(|(k, _)| {
                let a = (k as f64) * 0.3;
                [a.cos(), a.sin(), 0.0]
            })
            .collect::<Vec<_>>();
        assert!(design_condition_number(&dirs).is_none());
    }

    #[test]
    fn detects_near_and_antipodal_duplicates() {
        let dirs = [
            [1.0, 0.0, 0.0],
            [-0.9997, -0.0262, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9997, 0.0262],
        ];
        let indices = [3, 5, 7, 9];
        let pairs = duplicate_pairs(&indices, &dirs, 5.0);
        assert_eq!(pairs.len(), 2);
        let anti = pairs.iter().find(|p| p.antipodal).unwrap();
        assert_eq!((anti.i, anti.j), (3, 5));
        let near = pairs.iter().find(|p| !p.antipodal).unwrap();
        assert_eq!((near.i, near.j), (7, 9));
    }

    #[test]
    fn analyze_excludes_b0_and_flags_counts() {
        let mut directions = vec![[0.0, 0.0, 0.0]];
        directions.extend(dti6());
        let mut bvals = vec![0.0];
        bvals.extend([1000.0; 6]);
        let table = GradientTable::new(directions, bvals).unwrap();
        let summary = analyze(&table, ShellConfig::default(), AngularConfig::default());
        assert_eq!(summary.shells.len(), 1);
        let shell = &summary.shells[0];
        assert_eq!(shell.count, 6);
        assert!(shell.meets_dti_minimum);
        assert!(!shell.meets_dti_recommended);
        assert!(!shell.meets_csd_minimum);
    }

    #[test]
    fn summary_serializes_to_json() {
        let mut directions = vec![[0.0, 0.0, 0.0]];
        directions.extend(dti6());
        let mut bvals = vec![0.0];
        bvals.extend([1000.0; 6]);
        let table = GradientTable::new(directions, bvals).unwrap();
        let summary = analyze(&table, ShellConfig::default(), AngularConfig::default());
        let json = serde_json::to_string(&summary).unwrap();
        let back: AngularSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(summary, back);
    }
}
