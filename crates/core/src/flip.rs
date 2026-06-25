//! Flip/permutation detection: shell selection, candidate ranking, decision.

use std::time::{Duration, Instant};

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use crate::candidate::all_candidates;
use crate::coherence::{coherence_index, CoherenceConfig};
use crate::error::{Error, Result};
use crate::frame::IDENTITY;
use crate::gradient::{norm, GradientTable};
use crate::shell::{detect_shells, Shell, ShellConfig};
use crate::tensor::{fit_dti, FitConfig, TensorField};

/// b-value the working shell is preferred to sit nearest to.
const PREFERRED_B: f64 = 1000.0;
/// Smallest vector norm treated as a usable direction.
const MIN_NORM: f64 = 1e-9;

/// Tuning for the full flip-detection pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlipConfig {
    pub shell: ShellConfig,
    pub fit: FitConfig,
    pub coherence: CoherenceConfig,
    /// Relative coherence margin (best over runner-up) required to FLAG a repair.
    pub margin_threshold: f64,
    /// Gradient-frame → voxel-index map applied to `v1` before the coherence
    /// index (see [`crate::frame`]); identity leaves orientation untouched.
    pub frame_map: [[f64; 3]; 3],
}

impl Default for FlipConfig {
    fn default() -> Self {
        Self {
            shell: ShellConfig::default(),
            fit: FitConfig::default(),
            coherence: CoherenceConfig::default(),
            margin_threshold: 0.02,
            frame_map: IDENTITY,
        }
    }
}

/// Outcome of flip detection, mapped onto the report status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Decision {
    /// Identity is the best candidate: the table is consistent.
    Pass,
    /// A non-identity candidate wins but the margin is too small to trust.
    Warn,
    /// A non-identity candidate wins clearly: recommend the repair.
    Flag,
}

/// Coherence score for one candidate transform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateScore {
    pub label: String,
    pub matrix: [[f64; 3]; 3],
    pub is_identity: bool,
    pub coherence: f64,
    pub n_samples: usize,
}

/// Full flip-detection result, emitted to JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlipResult {
    pub working_b: f64,
    pub n_wm_voxels: usize,
    /// Mean FA over the coherence mask; low values flag a whole-brain-like mask
    /// that weakens flip-detection sensitivity. Defaults to `0` for older reports.
    #[serde(default)]
    pub mask_mean_fa: f64,
    /// All 48 candidates, sorted by descending coherence.
    pub ranking: Vec<CandidateScore>,
    pub best: CandidateScore,
    pub runner_up: CandidateScore,
    pub identity_coherence: f64,
    /// Absolute coherence margin, best over runner-up.
    pub margin: f64,
    /// Relative coherence margin, best over runner-up.
    pub relative_margin: f64,
    pub decision: Decision,
    /// Recommended repair matrix and label, present only on `Flag`.
    pub recommended_transform: Option<[[f64; 3]; 3]>,
    pub recommended_label: Option<String>,
}

/// Pick the working shell: nearest to `b ≈ 1000`, ties broken toward lower b.
#[must_use]
pub fn select_working_shell(table: &GradientTable, config: ShellConfig) -> Option<Shell> {
    detect_shells(&table.bvals, config)
        .into_iter()
        .filter(|s| !s.is_b0)
        .min_by(|a, b| {
            (a.nominal_b - PREFERRED_B)
                .abs()
                .total_cmp(&(b.nominal_b - PREFERRED_B).abs())
                .then(a.nominal_b.total_cmp(&b.nominal_b))
        })
}

/// Score all 48 candidates by fiber coherence, sorted by descending coherence.
///
/// `frame_map` is applied to the candidate orientation before scoring, so the
/// reported candidate matrix stays in the user's gradient frame while coherence
/// is measured in the image voxel-index frame.
#[must_use]
pub fn rank_candidates(
    field: &TensorField,
    mask: &[bool],
    config: CoherenceConfig,
    frame_map: &[[f64; 3]; 3],
) -> Vec<CandidateScore> {
    let mut scores: Vec<CandidateScore> = all_candidates()
        .iter()
        .map(|c| {
            let combined = compose(frame_map, &c.matrix);
            let transformed = transform_field(&field.v1, &combined);
            let (coherence, n_samples) = coherence_index(field.shape, &transformed, mask, config);
            CandidateScore {
                label: c.label.clone(),
                matrix: c.matrix,
                is_identity: c.is_identity,
                coherence,
                n_samples,
            }
        })
        .collect();
    scores.sort_by(|a, b| b.coherence.total_cmp(&a.coherence));
    scores
}

/// Classify a ranking into PASS / WARN / FLAG and the margins behind it.
///
/// A candidate `C` and its antipode `-C` are the same diffusion convention
/// (`g ≡ -g`), so they tie under the absolute-dot coherence index. The runner-up
/// therefore skips the best candidate's antipodal twin, and a best of `±I` is
/// treated as the identity convention (PASS).
#[must_use]
pub fn decide(ranking: &[CandidateScore], margin_threshold: f64) -> (Decision, f64, f64) {
    let Some(best) = ranking.first() else {
        return (Decision::Warn, 0.0, 0.0);
    };
    // With no runner-up (a lone candidate) the margin is undefined: treat it as 0,
    // which can only ever yield PASS (identity) or WARN, never a FLAG.
    let runner_coherence = runner_up(ranking).map_or(best.coherence, |r| r.coherence);
    let margin = best.coherence - runner_coherence;
    let relative = if best.coherence > 0.0 {
        margin / best.coherence
    } else {
        0.0
    };
    let decision = if represents_identity(best) {
        Decision::Pass
    } else if relative >= margin_threshold {
        Decision::Flag
    } else {
        Decision::Warn
    };
    (decision, margin, relative)
}

/// The highest-ranked candidate that is neither the best one nor its antipodal
/// twin, or `None` when the ranking holds fewer than two distinct conventions.
#[must_use]
pub fn runner_up(ranking: &[CandidateScore]) -> Option<&CandidateScore> {
    let best = ranking.first()?;
    let second = ranking.get(1)?;
    Some(
        ranking[1..]
            .iter()
            .find(|c| !antipodal_or_equal(&c.matrix, &best.matrix))
            .unwrap_or(second),
    )
}

fn represents_identity(candidate: &CandidateScore) -> bool {
    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    candidate.is_identity || antipodal_or_equal(&candidate.matrix, &IDENTITY)
}

fn antipodal_or_equal(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> bool {
    let equal = a == b;
    let antipodal = (0..3).all(|i| (0..3).all(|j| a[i][j] == -b[i][j]));
    equal || antipodal
}

/// Run the full pipeline: fit once, transform per candidate, rank, decide.
///
/// `data` is a 4-D `[nx, ny, nz, nvol]` DWI array. `mask`, when given, is a
/// flattened WM/brain mask in the same row-major voxel order as the fit grid;
/// otherwise an FA-threshold WM proxy is derived from the identity-table fit.
pub fn detect_flip(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    config: FlipConfig,
) -> Result<FlipResult> {
    let shell = select_working_shell(table, config.shell).ok_or(Error::NoUsableShell)?;
    detect_flip_on_shell(data, table, mask, &shell, config)
}

/// Run flip detection on an explicit working shell (skips shell auto-selection).
pub fn detect_flip_on_shell(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    shell: &Shell,
    config: FlipConfig,
) -> Result<FlipResult> {
    detect_flip_on_shell_with_glyphs(data, table, mask, shell, config).map(|(result, _)| result)
}

/// Wall-clock split of flip detection: tensor fit vs candidate coherence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetectTimings {
    /// Single log-linear DTI fit plus the WM-mask resolve.
    pub fit: Duration,
    /// Ranking all 48 candidates by the fiber-coherence index.
    pub coherence: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlyphData {
    pub field: TensorField,
    pub mask: Vec<bool>,
    pub frame_map: [[f64; 3]; 3],
}

/// [`detect_flip`] instrumented with a fit-vs-coherence timing split.
pub fn detect_flip_timed(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    config: FlipConfig,
) -> Result<(FlipResult, DetectTimings)> {
    let shell = select_working_shell(table, config.shell).ok_or(Error::NoUsableShell)?;
    detect_flip_on_shell_timed(data, table, mask, &shell, config)
}

pub fn detect_flip_with_glyphs(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    config: FlipConfig,
) -> Result<(FlipResult, GlyphData)> {
    let shell = select_working_shell(table, config.shell).ok_or(Error::NoUsableShell)?;
    detect_flip_on_shell_with_glyphs(data, table, mask, &shell, config)
}

pub fn detect_flip_on_shell_with_glyphs(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    shell: &Shell,
    config: FlipConfig,
) -> Result<(FlipResult, GlyphData)> {
    let (field, wm, n_wm, mask_mean_fa) = fit_and_mask(data, table, mask, shell, config)?;
    let ranking = rank_candidates(&field, &wm, config.coherence, &config.frame_map);
    let result = assemble_result(shell, n_wm, mask_mean_fa, ranking, config.margin_threshold);
    Ok((
        result,
        GlyphData {
            field,
            mask: wm,
            frame_map: config.frame_map,
        },
    ))
}

pub fn detect_flip_timed_with_glyphs(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    config: FlipConfig,
) -> Result<(FlipResult, DetectTimings, GlyphData)> {
    let shell = select_working_shell(table, config.shell).ok_or(Error::NoUsableShell)?;
    detect_flip_on_shell_timed_with_glyphs(data, table, mask, &shell, config)
}

pub fn detect_flip_on_shell_timed_with_glyphs(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    shell: &Shell,
    config: FlipConfig,
) -> Result<(FlipResult, DetectTimings, GlyphData)> {
    let t0 = Instant::now();
    let (field, wm, n_wm, mask_mean_fa) = fit_and_mask(data, table, mask, shell, config)?;
    let fit = t0.elapsed();
    let t1 = Instant::now();
    let ranking = rank_candidates(&field, &wm, config.coherence, &config.frame_map);
    let coherence = t1.elapsed();
    let result = assemble_result(shell, n_wm, mask_mean_fa, ranking, config.margin_threshold);
    Ok((
        result,
        DetectTimings { fit, coherence },
        GlyphData {
            field,
            mask: wm,
            frame_map: config.frame_map,
        },
    ))
}

/// [`detect_flip_on_shell`] instrumented with a fit-vs-coherence timing split.
pub fn detect_flip_on_shell_timed(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    shell: &Shell,
    config: FlipConfig,
) -> Result<(FlipResult, DetectTimings)> {
    detect_flip_on_shell_timed_with_glyphs(data, table, mask, shell, config)
        .map(|(result, timings, _)| (result, timings))
}

/// Fit the tensor once and resolve the WM mask (user-supplied or FA proxy).
///
/// With no mask, the foreground-gated [`TensorField::wm_proxy`] is used, falling
/// back to the plain FA proxy if the foreground cut leaves it empty. Also returns
/// the mean FA over the chosen mask, used to flag a whole-brain-like mask.
fn fit_and_mask(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    shell: &Shell,
    config: FlipConfig,
) -> Result<(TensorField, Vec<bool>, usize, f64)> {
    let b0_indices = table.b0_indices(config.shell.b0_threshold);
    let field = fit_dti(data, table, &b0_indices, &shell.indices, config.fit)?;
    let wm = match mask {
        Some(m) => m.to_vec(),
        None => {
            let proxy = field.wm_proxy(config.fit.fa_threshold);
            if proxy.iter().any(|&keep| keep) {
                proxy
            } else {
                field.wm_mask(config.fit.fa_threshold)
            }
        }
    };
    let n_wm = wm.iter().filter(|&&keep| keep).count();
    if n_wm == 0 {
        return Err(Error::Fit("no WM voxels above the FA threshold".into()));
    }
    let mask_mean_fa = mean_in_mask_fa(&field, &wm);
    Ok((field, wm, n_wm, mask_mean_fa))
}

/// Mean FA over the valid voxels of a mask (`0` when none are valid).
fn mean_in_mask_fa(field: &TensorField, mask: &[bool]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for ((&keep, &fa), &ok) in mask.iter().zip(&field.fa).zip(&field.valid) {
        if keep && ok {
            sum += fa;
            n += 1;
        }
    }
    if n > 0 {
        sum / n as f64
    } else {
        0.0
    }
}

/// Build the final result from a candidate ranking and the working shell.
fn assemble_result(
    shell: &Shell,
    n_wm: usize,
    mask_mean_fa: f64,
    ranking: Vec<CandidateScore>,
    margin_threshold: f64,
) -> FlipResult {
    let (decision, margin, relative_margin) = decide(&ranking, margin_threshold);
    let identity_coherence = ranking
        .iter()
        .find(|c| c.is_identity)
        .map_or(0.0, |c| c.coherence);
    let best = ranking[0].clone();
    let runner_up = runner_up(&ranking).cloned().unwrap_or_else(|| best.clone());
    let (recommended_transform, recommended_label) = match decision {
        Decision::Flag => (Some(best.matrix), Some(best.label.clone())),
        _ => (None, None),
    };

    FlipResult {
        working_b: shell.nominal_b,
        n_wm_voxels: n_wm,
        mask_mean_fa,
        ranking,
        best,
        runner_up,
        identity_coherence,
        margin,
        relative_margin,
        decision,
        recommended_transform,
        recommended_label,
    }
}

fn transform_field(v1: &[[f64; 3]], matrix: &[[f64; 3]; 3]) -> Vec<[f64; 3]> {
    v1.iter()
        .map(|&v| {
            if norm(v) < MIN_NORM {
                v
            } else {
                apply_matrix(matrix, v)
            }
        })
        .collect()
}

fn apply_matrix(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn compose(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};
    use ndarray::{Array, ArrayD, IxDyn};

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = norm(v);
        [v[0] / n, v[1] / n, v[2] / n]
    }

    fn dwi_scheme() -> GradientTable {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let t = 1.0 / 3.0_f64.sqrt();
        let directions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [s, s, 0.0],
            [s, 0.0, s],
            [0.0, s, s],
            [t, t, t],
        ];
        let mut bvals = vec![0.0];
        bvals.extend([1000.0; 7]);
        GradientTable::new(directions, bvals).unwrap()
    }

    // 4-D DWI of a single straight fiber tube along `a` in a 3-D box; background
    // is empty. A full 3-D extent keeps the coherence index well-posed on every
    // axis (a single slice ties z-permutations with identity).
    fn tube_dwi(n: usize, a: [f64; 3], radius: f64, table: &GradientTable) -> ArrayD<f32> {
        let c = (n as f64 - 1.0) / 2.0;
        let an = unit(a);
        let mut d = Matrix3::identity() * 0.3e-3;
        for r in 0..3 {
            for col in 0..3 {
                d[(r, col)] += (1.7e-3 - 0.3e-3) * an[r] * an[col];
            }
        }
        let nt = table.len();
        Array::from_shape_fn(IxDyn(&[n, n, n, nt]), |idx| {
            let p = [idx[0] as f64 - c, idx[1] as f64 - c, idx[2] as f64 - c];
            let proj = p[0] * an[0] + p[1] * an[1] + p[2] * an[2];
            let perp = [
                p[0] - proj * an[0],
                p[1] - proj * an[1],
                p[2] - proj * an[2],
            ];
            if norm(perp) > radius {
                return 0.0;
            }
            let g = table.directions[idx[3]];
            let gv = Vector3::new(g[0], g[1], g[2]);
            let q = (gv.transpose() * d * gv)[(0, 0)];
            (1000.0 * (-table.bvals[idx[3]] * q).exp()) as f32
        })
        .into_dyn()
    }

    fn tensor_for(an: [f64; 3]) -> Matrix3<f64> {
        let mut d = Matrix3::identity() * 0.3e-3;
        for r in 0..3 {
            for col in 0..3 {
                d[(r, col)] += (1.7e-3 - 0.3e-3) * an[r] * an[col];
            }
        }
        d
    }

    // Crossing-fiber phantom: non-parallel tubes whose mix of orientations breaks
    // the permutation symmetry a single straight tube leaves ambiguous, so the
    // true convention is uniquely best (not a numerical tie).
    fn crossing(n: usize, axes: &[[f64; 3]], radius: f64, table: &GradientTable) -> ArrayD<f32> {
        let c = (n as f64 - 1.0) / 2.0;
        let units: Vec<[f64; 3]> = axes.iter().map(|a| unit(*a)).collect();
        let tensors: Vec<Matrix3<f64>> = units.iter().map(|&a| tensor_for(a)).collect();
        let nt = table.len();
        Array::from_shape_fn(IxDyn(&[n, n, n, nt]), |idx| {
            let p = [idx[0] as f64 - c, idx[1] as f64 - c, idx[2] as f64 - c];
            let g = table.directions[idx[3]];
            let gv = Vector3::new(g[0], g[1], g[2]);
            let b = table.bvals[idx[3]];
            let mut sum = 0.0;
            let mut count = 0;
            for (an, d) in units.iter().zip(&tensors) {
                let proj = p[0] * an[0] + p[1] * an[1] + p[2] * an[2];
                let perp = [
                    p[0] - proj * an[0],
                    p[1] - proj * an[1],
                    p[2] - proj * an[2],
                ];
                if norm(perp) <= radius {
                    sum += (-b * (gv.transpose() * d * gv)[(0, 0)]).exp();
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                (1000.0 * sum / count as f64) as f32
            }
        })
        .into_dyn()
    }

    #[test]
    fn detect_flip_passes_on_consistent_table() {
        let table = dwi_scheme();
        let data = crossing(21, &[[3.0, 1.0, 2.0], [1.0, -2.0, 3.0]], 1.5, &table);
        let config = FlipConfig {
            coherence: CoherenceConfig { step: 2.0 },
            ..FlipConfig::default()
        };
        let result = detect_flip(&data, &table, None, config).unwrap();
        assert_eq!(result.working_b, 1000.0);
        assert!(result.n_wm_voxels > 0);
        assert!(result.mask_mean_fa > 0.0);
        assert!(result.best.is_identity, "best={}", result.best.label);
        assert_eq!(result.decision, Decision::Pass);
        assert!(result.recommended_transform.is_none());
        assert_eq!(result.ranking.len(), crate::candidate::N_CANDIDATES);
    }

    // A correct FSL table on a positive-determinant image: the true voxel-frame
    // gradient is `F·g` (x-flip). With the frame map applied, identity wins.
    #[test]
    fn correct_table_passes_on_positive_determinant_frame() {
        let table = dwi_scheme();
        let xflip = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let voxel_table = crate::repair::apply_transform(&table, &xflip);
        let data = tube_dwi(25, [1.0, 2.0, 0.0], 0.9, &voxel_table);
        let config = FlipConfig {
            coherence: CoherenceConfig { step: 2.0 },
            frame_map: xflip,
            ..FlipConfig::default()
        };
        let result = detect_flip(&data, &table, None, config).unwrap();
        assert!(result.best.is_identity, "best={}", result.best.label);
        assert_eq!(result.decision, Decision::Pass);
    }

    // Same acquisition without the frame map: the good table looks x-flipped —
    // the false positive the frame correction exists to prevent.
    #[test]
    fn missing_frame_map_misreads_positive_determinant_table() {
        let table = dwi_scheme();
        let xflip = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let voxel_table = crate::repair::apply_transform(&table, &xflip);
        let data = tube_dwi(25, [1.0, 2.0, 0.0], 0.9, &voxel_table);
        let config = FlipConfig {
            coherence: CoherenceConfig { step: 2.0 },
            ..FlipConfig::default()
        };
        let result = detect_flip(&data, &table, None, config).unwrap();
        assert!(!result.best.is_identity);
    }

    #[test]
    fn timed_detection_matches_untimed() {
        let table = dwi_scheme();
        let data = tube_dwi(25, [1.0, 2.0, 0.0], 0.9, &table);
        let config = FlipConfig {
            coherence: CoherenceConfig { step: 2.0 },
            ..FlipConfig::default()
        };
        let plain = detect_flip(&data, &table, None, config).unwrap();
        let (timed, _) = detect_flip_timed(&data, &table, None, config).unwrap();
        assert_eq!(plain, timed);
    }

    #[test]
    fn glyph_capture_preserves_detection_result() {
        let table = dwi_scheme();
        let data = tube_dwi(17, [1.0, 2.0, 1.0], 1.2, &table);
        let config = FlipConfig {
            coherence: CoherenceConfig { step: 2.0 },
            ..FlipConfig::default()
        };
        let expected = detect_flip(&data, &table, None, config).unwrap();
        let (actual, glyphs) = detect_flip_with_glyphs(&data, &table, None, config).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(glyphs.field.shape, [17, 17, 17]);
        assert_eq!(glyphs.mask.len(), 17 * 17 * 17);
        assert_eq!(glyphs.frame_map, config.frame_map);
    }

    #[test]
    fn flip_result_serializes_to_json() {
        let table = dwi_scheme();
        let data = tube_dwi(21, [1.0, 2.0, 0.0], 0.9, &table);
        let config = FlipConfig {
            coherence: CoherenceConfig { step: 2.0 },
            ..FlipConfig::default()
        };
        let result = detect_flip(&data, &table, None, config).unwrap();
        let json = serde_json::to_string(&result).unwrap();
        let back: FlipResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }

    fn score(label: &str, is_identity: bool, coherence: f64) -> CandidateScore {
        CandidateScore {
            label: label.to_string(),
            matrix: [[0.0; 3]; 3],
            is_identity,
            coherence,
            n_samples: 100,
        }
    }

    fn score_m(matrix: [[f64; 3]; 3], coherence: f64) -> CandidateScore {
        CandidateScore {
            label: String::new(),
            matrix,
            is_identity: matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            coherence,
            n_samples: 100,
        }
    }

    #[test]
    fn selects_shell_nearest_to_thousand() {
        let mut bvals = vec![0.0];
        bvals.extend([700.0; 6]);
        bvals.extend([2000.0; 6]);
        let dirs = vec![[1.0, 0.0, 0.0]; bvals.len()];
        let table = GradientTable::new(dirs, bvals).unwrap();
        let shell = select_working_shell(&table, ShellConfig::default()).unwrap();
        assert_eq!(shell.nominal_b, 700.0);
    }

    #[test]
    fn no_shell_when_all_b0() {
        let table = GradientTable::new(vec![[0.0, 0.0, 0.0]; 3], vec![0.0; 3]).unwrap();
        assert!(select_working_shell(&table, ShellConfig::default()).is_none());
    }

    #[test]
    fn identity_winner_is_pass() {
        let ranking = vec![score("+x+y+z", true, 0.90), score("-x+y+z", false, 0.70)];
        let (decision, margin, rel) = decide(&ranking, 0.02);
        assert_eq!(decision, Decision::Pass);
        assert!((margin - 0.20).abs() < 1e-12);
        assert!(rel > 0.02);
    }

    #[test]
    fn clear_non_identity_winner_is_flag() {
        let ranking = vec![score("-x+y+z", false, 0.90), score("+x+y+z", true, 0.60)];
        let (decision, _, _) = decide(&ranking, 0.02);
        assert_eq!(decision, Decision::Flag);
    }

    #[test]
    fn ambiguous_non_identity_winner_is_warn() {
        let ranking = vec![score("-x+y+z", false, 0.901), score("+x+y+z", true, 0.900)];
        let (decision, _, rel) = decide(&ranking, 0.02);
        assert_eq!(decision, Decision::Warn);
        assert!(rel < 0.02);
    }

    #[test]
    fn runner_up_skips_antipodal_twin() {
        let flip_x = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let flip_x_twin = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let swap_xy = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let ranking = vec![
            score_m(flip_x, 0.90),
            score_m(flip_x_twin, 0.90),
            score_m(swap_xy, 0.60),
        ];
        let (decision, margin, _) = decide(&ranking, 0.02);
        assert_eq!(decision, Decision::Flag);
        assert!((margin - 0.30).abs() < 1e-12);
    }

    #[test]
    fn decide_handles_empty_ranking() {
        let (decision, margin, rel) = decide(&[], 0.02);
        assert_eq!(decision, Decision::Warn);
        assert_eq!(margin, 0.0);
        assert_eq!(rel, 0.0);
    }

    #[test]
    fn decide_handles_single_candidate() {
        assert!(runner_up(&[]).is_none());
        let identity = vec![score("+x+y+z", true, 0.9)];
        assert!(runner_up(&identity).is_none());
        assert_eq!(decide(&identity, 0.02), (Decision::Pass, 0.0, 0.0));

        let flipped = vec![score("-x+y+z", false, 0.9)];
        assert_eq!(decide(&flipped, 0.02), (Decision::Warn, 0.0, 0.0));
    }

    #[test]
    fn negated_identity_is_pass() {
        let neg_identity = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let flip_x = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let ranking = vec![score_m(neg_identity, 0.90), score_m(flip_x, 0.60)];
        let (decision, _, _) = decide(&ranking, 0.02);
        assert_eq!(decision, Decision::Pass);
    }
}
