//! Gradient table representation and helpers.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A diffusion gradient table: one direction and one b-value per volume.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientTable {
    /// Gradient directions, one `[x, y, z]` per volume.
    pub directions: Vec<[f64; 3]>,
    /// b-values, one per volume.
    pub bvals: Vec<f64>,
}

/// Unit-norm statistics over the non-b0 directions, used to flag
/// amplitude-encoded bvecs (vector length encoding the b-value).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormStats {
    pub non_b0_count: usize,
    pub non_unit_count: usize,
    pub non_unit_fraction: f64,
    pub norm_min: f64,
    pub norm_max: f64,
    pub norm_mean: f64,
    pub tolerance: f64,
}

impl GradientTable {
    /// Build a table, checking that directions and b-values agree in count.
    pub fn new(directions: Vec<[f64; 3]>, bvals: Vec<f64>) -> Result<Self> {
        if directions.len() != bvals.len() {
            return Err(Error::GradientCountMismatch {
                directions: directions.len(),
                bvals: bvals.len(),
            });
        }
        Ok(Self { directions, bvals })
    }

    /// Number of volumes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bvals.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bvals.is_empty()
    }

    /// Indices whose b-value is at or below `b0_threshold`.
    #[must_use]
    pub fn b0_indices(&self, b0_threshold: f64) -> Vec<usize> {
        self.bvals
            .iter()
            .enumerate()
            .filter(|(_, &b)| b <= b0_threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Return a copy with non-b0 directions scaled to unit length.
    ///
    /// b0 and (near-)zero-norm directions are left as `[0, 0, 0]`.
    #[must_use]
    pub fn normalized(&self, b0_threshold: f64) -> Self {
        let directions = self
            .directions
            .iter()
            .zip(&self.bvals)
            .map(|(&d, &b)| {
                let n = norm(d);
                if b <= b0_threshold || n < f64::EPSILON {
                    [0.0, 0.0, 0.0]
                } else {
                    [d[0] / n, d[1] / n, d[2] / n]
                }
            })
            .collect();
        Self {
            directions,
            bvals: self.bvals.clone(),
        }
    }

    /// Unit-norm statistics over the non-b0 directions, reusing [`norm`].
    ///
    /// Amplitude-encoded bvecs have lengths that encode the b-value, so their
    /// non-b0 norms deviate from 1; a high `non_unit_fraction` flags this.
    #[must_use]
    pub fn norm_stats(&self, b0_threshold: f64, tolerance: f64) -> NormStats {
        let norms: Vec<f64> = self
            .directions
            .iter()
            .zip(&self.bvals)
            .filter(|(_, &b)| b > b0_threshold)
            .map(|(&d, _)| norm(d))
            .collect();
        let non_b0_count = norms.len();
        if non_b0_count == 0 {
            return NormStats {
                non_b0_count: 0,
                non_unit_count: 0,
                non_unit_fraction: 0.0,
                norm_min: 0.0,
                norm_max: 0.0,
                norm_mean: 0.0,
                tolerance,
            };
        }
        let non_unit_count = norms
            .iter()
            .filter(|&&n| (n - 1.0).abs() > tolerance)
            .count();
        let norm_min = norms.iter().copied().fold(f64::INFINITY, f64::min);
        let norm_max = norms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let norm_mean = norms.iter().sum::<f64>() / non_b0_count as f64;
        NormStats {
            non_b0_count,
            non_unit_count,
            non_unit_fraction: non_unit_count as f64 / non_b0_count as f64,
            norm_min,
            norm_max,
            norm_mean,
            tolerance,
        }
    }

    /// Recover per-volume b-values from amplitude-encoded norms.
    ///
    /// With `b ∝ |g|²`, the true weighting is `b_i = b_nominal·(|g_i|/max)²`,
    /// where `b_nominal` is the largest non-b0 b-value and `max` the largest
    /// non-b0 norm. Returns the corrected table (unit-length directions, rounded
    /// integer b-values) plus `(b_nominal, max_norm)`.
    pub fn recover_bvals(&self, b0_threshold: f64) -> Result<(Self, f64, f64)> {
        let mut max_norm = 0.0_f64;
        let mut b_nominal = 0.0_f64;
        for (&d, &b) in self.directions.iter().zip(&self.bvals) {
            if b > b0_threshold {
                max_norm = max_norm.max(norm(d));
                b_nominal = b_nominal.max(b);
            }
        }
        if max_norm < f64::EPSILON {
            return Err(Error::NoUsableShell);
        }
        let bvals = self
            .directions
            .iter()
            .zip(&self.bvals)
            .map(|(&d, &b)| {
                if b <= b0_threshold {
                    0.0
                } else {
                    let ratio = norm(d) / max_norm;
                    (b_nominal * ratio * ratio).round()
                }
            })
            .collect();
        let table = Self {
            directions: self.normalized(b0_threshold).directions,
            bvals,
        };
        Ok((table, b_nominal, max_norm))
    }
}

#[must_use]
pub(crate) fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_count_mismatch() {
        assert!(GradientTable::new(vec![[0.0, 0.0, 0.0]], vec![0.0, 1.0]).is_err());
    }

    #[test]
    fn finds_b0() {
        let t =
            GradientTable::new(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], vec![0.0, 1000.0]).unwrap();
        assert_eq!(t.b0_indices(50.0), vec![0]);
    }

    #[test]
    fn normalizes_non_b0() {
        let t =
            GradientTable::new(vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], vec![0.0, 1000.0]).unwrap();
        let n = t.normalized(50.0);
        assert_eq!(n.directions[0], [0.0, 0.0, 0.0]);
        assert!((n.directions[1][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn norm_stats_clean_unit_table() {
        let t = GradientTable::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0.0, 1000.0, 1000.0],
        )
        .unwrap();
        let s = t.norm_stats(50.0, 0.05);
        assert_eq!(s.non_b0_count, 2);
        assert_eq!(s.non_unit_count, 0);
        assert_eq!(s.non_unit_fraction, 0.0);
        assert!((s.norm_mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn norm_stats_flags_amplitude_encoded() {
        // norms ∝ √b: |g| = sqrt(b / b_nominal), b_nominal = 3000.
        let t = GradientTable::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],                    // |g| = 1   -> b 3000
                [(2.0_f64 / 3.0).sqrt(), 0.0, 0.0], // |g|²=2/3  -> b 2000
                [0.0, (1.0_f64 / 3.0).sqrt(), 0.0], // |g|²=1/3  -> b 1000
            ],
            vec![0.0, 3000.0, 3000.0, 3000.0],
        )
        .unwrap();
        let s = t.norm_stats(50.0, 0.05);
        assert_eq!(s.non_b0_count, 3);
        assert_eq!(s.non_unit_count, 2);
        assert!(s.non_unit_fraction > 0.5);
        assert!(s.norm_max <= 1.0 + 1e-12 && s.norm_min < 0.95);
    }

    #[test]
    fn norm_stats_all_b0_is_empty() {
        let t = GradientTable::new(vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], vec![0.0, 0.0]).unwrap();
        let s = t.norm_stats(50.0, 0.05);
        assert_eq!(s.non_b0_count, 0);
        assert_eq!(s.non_unit_fraction, 0.0);
    }

    #[test]
    fn recover_bvals_reconstructs_multishell() {
        let t = GradientTable::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [(2.0_f64 / 3.0).sqrt(), 0.0, 0.0],
                [0.0, (1.0_f64 / 3.0).sqrt(), 0.0],
            ],
            vec![0.0, 3000.0, 3000.0, 3000.0],
        )
        .unwrap();
        let (rec, b_nominal, _) = t.recover_bvals(50.0).unwrap();
        assert_eq!(b_nominal, 3000.0);
        assert_eq!(rec.bvals, vec![0.0, 3000.0, 2000.0, 1000.0]);
        assert!((norm(rec.directions[1]) - 1.0).abs() < 1e-12);
        assert_eq!(rec.directions[0], [0.0, 0.0, 0.0]);
        let s = rec.norm_stats(50.0, 0.05);
        assert_eq!(s.non_unit_count, 0);
    }

    #[test]
    fn recover_bvals_rejects_all_b0() {
        let t = GradientTable::new(vec![[0.0, 0.0, 0.0]], vec![0.0]).unwrap();
        assert!(t.recover_bvals(50.0).is_err());
    }
}
