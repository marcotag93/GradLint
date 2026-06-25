//! Apply a chosen candidate transform to a gradient table, with write safety.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::flip::FlipResult;
use crate::gradient::GradientTable;

/// Apply a signed-permutation matrix `C` to every gradient direction
/// (`g_repaired = C · g`); b-values are left unchanged.
#[must_use]
pub fn apply_transform(table: &GradientTable, matrix: &[[f64; 3]; 3]) -> GradientTable {
    let directions = table
        .directions
        .iter()
        .map(|&g| {
            [
                pos_zero(matrix[0][0] * g[0] + matrix[0][1] * g[1] + matrix[0][2] * g[2]),
                pos_zero(matrix[1][0] * g[0] + matrix[1][1] * g[1] + matrix[1][2] * g[2]),
                pos_zero(matrix[2][0] * g[0] + matrix[2][1] * g[1] + matrix[2][2] * g[2]),
            ]
        })
        .collect();
    GradientTable {
        directions,
        bvals: table.bvals.clone(),
    }
}

/// A concrete repair: the corrected table and the transform that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct Repair {
    pub table: GradientTable,
    pub matrix: [[f64; 3]; 3],
    pub label: String,
}

impl Repair {
    /// Build the repair recommended by a flip result, or `None` when the
    /// decision was not `Flag` — WARN and PASS never auto-repair.
    #[must_use]
    pub fn from_flip(table: &GradientTable, flip: &FlipResult) -> Option<Self> {
        let matrix = flip.recommended_transform?;
        let label = flip.recommended_label.clone()?;
        Some(Self {
            table: apply_transform(table, &matrix),
            matrix,
            label,
        })
    }

    /// Force the best-candidate convention even on a non-`Flag` decision (used
    /// only by `--force-repair`). `None` when the best is the current (identity
    /// or antipodal) convention — there is nothing to repair.
    #[must_use]
    pub fn force_from_best(table: &GradientTable, flip: &FlipResult) -> Option<Self> {
        let best = &flip.best;
        if best.is_identity || is_antipodal_identity(&best.matrix) {
            return None;
        }
        Some(Self {
            table: apply_transform(table, &best.matrix),
            matrix: best.matrix,
            label: best.label.clone(),
        })
    }
}

fn is_antipodal_identity(matrix: &[[f64; 3]; 3]) -> bool {
    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    (0..3).all(|i| (0..3).all(|j| matrix[i][j] == -IDENTITY[i][j]))
}

/// Guard an output path before writing.
///
/// Returns `Ok(None)` when the path is free. When it already exists, the write
/// is refused unless `in_place` is set, in which case the original is moved to
/// a sibling `.bak` file (returned) so it is never silently lost.
pub fn prepare_output(path: impl AsRef<Path>, in_place: bool) -> Result<Option<PathBuf>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }
    if !in_place {
        return Err(Error::OutputExists {
            path: path.display().to_string(),
        });
    }
    let backup = next_backup_path(path);
    fs::rename(path, &backup)?;
    Ok(Some(backup))
}

fn next_backup_path(path: &Path) -> PathBuf {
    let mut candidate = with_suffix(path, "bak");
    let mut n = 1u32;
    while candidate.exists() {
        candidate = with_suffix(path, &format!("bak.{n}"));
        n += 1;
    }
    candidate
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name: OsString = path.as_os_str().to_owned();
    name.push(".");
    name.push(suffix);
    PathBuf::from(name)
}

#[inline]
fn pos_zero(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flip::{CandidateScore, Decision};
    use tempfile::tempdir;

    fn table() -> GradientTable {
        GradientTable::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.7, 0.7]],
            vec![0.0, 1000.0, 1000.0],
        )
        .unwrap()
    }

    fn score(label: &str, matrix: [[f64; 3]; 3], is_identity: bool) -> CandidateScore {
        CandidateScore {
            label: label.to_string(),
            matrix,
            is_identity,
            coherence: 0.9,
            n_samples: 100,
        }
    }

    fn flip(decision: Decision, matrix: [[f64; 3]; 3], label: &str) -> FlipResult {
        let (recommended_transform, recommended_label) = match decision {
            Decision::Flag => (Some(matrix), Some(label.to_string())),
            _ => (None, None),
        };
        FlipResult {
            working_b: 1000.0,
            n_wm_voxels: 10,
            mask_mean_fa: 0.4,
            ranking: vec![score(label, matrix, decision == Decision::Pass)],
            best: score(label, matrix, decision == Decision::Pass),
            runner_up: score("dummy", matrix, false),
            identity_coherence: 0.5,
            margin: 0.1,
            relative_margin: 0.1,
            decision,
            recommended_transform,
            recommended_label,
        }
    }

    #[test]
    fn identity_transform_is_a_noop() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(apply_transform(&table(), &id), table());
    }

    #[test]
    fn flip_x_negates_first_axis_without_signed_zero() {
        let flip_x = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let out = apply_transform(&table(), &flip_x);
        assert_eq!(out.directions[1], [-1.0, 0.0, 0.0]);
        assert!(out.directions[0][0].is_sign_positive());
        assert_eq!(out.bvals, table().bvals);
    }

    #[test]
    fn repair_only_built_on_flag() {
        let m = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(Repair::from_flip(&table(), &flip(Decision::Pass, m, "+x+y+z")).is_none());
        assert!(Repair::from_flip(&table(), &flip(Decision::Warn, m, "-x+y+z")).is_none());
        let repair = Repair::from_flip(&table(), &flip(Decision::Flag, m, "-x+y+z")).unwrap();
        assert_eq!(repair.label, "-x+y+z");
        assert_eq!(repair.table.directions[1], [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn force_from_best_repairs_warn_but_not_identity() {
        let flip_x = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let repair = Repair::force_from_best(&table(), &flip(Decision::Warn, flip_x, "-x+y+z"))
            .expect("force-repair applies the best convention on WARN");
        assert_eq!(repair.label, "-x+y+z");
        assert_eq!(repair.table.directions[1], [-1.0, 0.0, 0.0]);

        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(
            Repair::force_from_best(&table(), &flip(Decision::Pass, identity, "+x+y+z")).is_none()
        );
        let antipode = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        assert!(
            Repair::force_from_best(&table(), &flip(Decision::Warn, antipode, "-x-y-z")).is_none()
        );
    }

    #[test]
    fn refuses_to_overwrite_without_in_place() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("dwi.bvec");
        fs::write(&p, "orig").unwrap();
        assert!(matches!(
            prepare_output(&p, false),
            Err(Error::OutputExists { .. })
        ));
        assert_eq!(fs::read_to_string(&p).unwrap(), "orig");
    }

    #[test]
    fn in_place_backs_up_then_frees_path() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("dwi.bvec");
        fs::write(&p, "orig").unwrap();
        let backup = prepare_output(&p, true).unwrap().unwrap();
        assert!(!p.exists());
        assert_eq!(fs::read_to_string(&backup).unwrap(), "orig");
        assert_eq!(backup.file_name().unwrap(), "dwi.bvec.bak");
    }

    #[test]
    fn fresh_path_needs_no_backup() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("new.bvec");
        assert!(prepare_output(&p, false).unwrap().is_none());
    }
}
