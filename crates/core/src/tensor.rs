//! Log-linear single-shell DTI fit, eigen-decomposition, and a WM mask proxy.
//!
//! The tensor is fit **once** with the given table; flip candidates then reuse
//! the principal eigenvector field via the analytic `v1_cand = C · v1` identity
//! (see `crate::flip`). This is exact only for the log-linear single-shell fit.

use nalgebra::{DMatrix, DVector, Matrix3, SymmetricEigen};
use ndarray::ArrayD;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::gradient::{norm, GradientTable};

/// Smallest positive signal used as a floor before taking logs.
const MIN_POS: f64 = 1e-6;
/// Singular-value floor for the design-matrix pseudoinverse.
const PINV_EPS: f64 = 1e-12;
/// Histogram bins for the Otsu foreground threshold.
const OTSU_BINS: usize = 256;

/// Tuning for the DTI fit and the FA-threshold WM proxy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitConfig {
    /// Voxels whose mean b0 signal is below this are treated as background.
    pub min_signal: f64,
    /// FA cutoff for the white-matter proxy mask.
    pub fa_threshold: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            min_signal: 1.0,
            fa_threshold: 0.2,
        }
    }
}

/// Per-voxel principal-eigenvector and FA field on a 3-D grid.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorField {
    /// Grid shape `[nx, ny, nz]`.
    pub shape: [usize; 3],
    /// Principal eigenvector per voxel (`[0, 0, 0]` where invalid), row-major.
    pub v1: Vec<[f64; 3]>,
    /// Fractional anisotropy per voxel (`0` where invalid).
    pub fa: Vec<f64>,
    /// Mean b0 signal per voxel (`0` where invalid), used for the foreground cut.
    pub s0: Vec<f64>,
    /// Whether each voxel produced a usable fit.
    pub valid: Vec<bool>,
}

impl TensorField {
    /// Row-major linear index of a voxel.
    #[must_use]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (x * self.shape[1] + y) * self.shape[2] + z
    }

    /// White-matter proxy mask: valid voxels with FA at or above the threshold.
    #[must_use]
    pub fn wm_mask(&self, fa_threshold: f64) -> Vec<bool> {
        self.fa
            .iter()
            .zip(&self.valid)
            .map(|(&fa, &ok)| ok && fa >= fa_threshold)
            .collect()
    }

    /// White-matter proxy used when no mask is supplied: valid, anisotropic
    /// (FA ≥ threshold) voxels that also clear an Otsu foreground cut on the b0
    /// signal. The foreground cut drops background/air, whose spurious anisotropy
    /// would otherwise dilute the coherence index and shrink the flip margin.
    #[must_use]
    pub fn wm_proxy(&self, fa_threshold: f64) -> Vec<bool> {
        let floor = self.foreground_threshold();
        self.fa
            .iter()
            .zip(&self.valid)
            .zip(&self.s0)
            .map(|((&fa, &ok), &s0)| ok && fa >= fa_threshold && s0 >= floor)
            .collect()
    }

    /// Otsu foreground threshold over the whole b0 field. Using every voxel
    /// (background included) keeps the background/tissue split bimodal, so a
    /// tissue-only field is never split — the cut only removes true background.
    fn foreground_threshold(&self) -> f64 {
        otsu_threshold(&self.s0)
    }
}

/// Fit the log-linear DTI tensor per voxel over `b0_indices ∪ shell_indices`.
///
/// `data` is a 4-D `[nx, ny, nz, nvol]` DWI array. The fit is parallelized
/// across voxels with `rayon`.
pub fn fit_dti(
    data: &ArrayD<f32>,
    table: &GradientTable,
    b0_indices: &[usize],
    shell_indices: &[usize],
    config: FitConfig,
) -> Result<TensorField> {
    if data.ndim() != 4 {
        return Err(Error::Fit(format!(
            "expected 4-D DWI, got {}-D",
            data.ndim()
        )));
    }
    let shape = [data.shape()[0], data.shape()[1], data.shape()[2]];
    if shell_indices.is_empty() {
        return Err(Error::Fit("no DWI volumes in the working shell".into()));
    }

    let pinv = pseudoinverse(table, b0_indices, shell_indices)?;
    let (ny, nz) = (shape[1], shape[2]);
    let nvox = shape[0] * ny * nz;

    let fits: Vec<VoxelFit> = (0..nvox)
        .into_par_iter()
        .map(|lin| {
            let z = lin % nz;
            let y = (lin / nz) % ny;
            let x = lin / (nz * ny);
            fit_voxel(data, b0_indices, shell_indices, &pinv, config, x, y, z)
        })
        .collect();

    let mut field = TensorField {
        shape,
        v1: Vec::with_capacity(nvox),
        fa: Vec::with_capacity(nvox),
        s0: Vec::with_capacity(nvox),
        valid: Vec::with_capacity(nvox),
    };
    for f in fits {
        field.v1.push(f.v1);
        field.fa.push(f.fa);
        field.s0.push(f.s0);
        field.valid.push(f.valid);
    }
    Ok(field)
}

struct VoxelFit {
    v1: [f64; 3],
    fa: f64,
    s0: f64,
    valid: bool,
}

const INVALID: VoxelFit = VoxelFit {
    v1: [0.0, 0.0, 0.0],
    fa: 0.0,
    s0: 0.0,
    valid: false,
};

#[allow(clippy::too_many_arguments)]
fn fit_voxel(
    data: &ArrayD<f32>,
    b0_indices: &[usize],
    shell_indices: &[usize],
    pinv: &DMatrix<f64>,
    config: FitConfig,
    x: usize,
    y: usize,
    z: usize,
) -> VoxelFit {
    let s0 = if b0_indices.is_empty() {
        shell_indices
            .iter()
            .map(|&i| f64::from(data[[x, y, z, i]]))
            .sum::<f64>()
            / shell_indices.len() as f64
    } else {
        b0_indices
            .iter()
            .map(|&i| f64::from(data[[x, y, z, i]]))
            .sum::<f64>()
            / b0_indices.len() as f64
    };
    if s0 < config.min_signal {
        return VoxelFit { s0, ..INVALID };
    }

    let m = b0_indices.len() + shell_indices.len();
    let mut yv = DVector::zeros(m);
    for (k, &i) in b0_indices.iter().chain(shell_indices).enumerate() {
        yv[k] = f64::from(data[[x, y, z, i]]).max(MIN_POS).ln();
    }
    let p = pinv * yv;
    // p = [ln S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz].
    match principal_eigen([p[1], p[2], p[3], p[4], p[5], p[6]]) {
        Some((v1, fa)) => VoxelFit {
            v1,
            fa,
            s0,
            valid: true,
        },
        None => VoxelFit { s0, ..INVALID },
    }
}

fn pseudoinverse(
    table: &GradientTable,
    b0_indices: &[usize],
    shell_indices: &[usize],
) -> Result<DMatrix<f64>> {
    let m = b0_indices.len() + shell_indices.len();
    let mut rows = Vec::with_capacity(m * 7);
    for _ in b0_indices {
        rows.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    for &i in shell_indices {
        let d = table.directions[i];
        let n = norm(d);
        let (gx, gy, gz) = if n > MIN_POS {
            (d[0] / n, d[1] / n, d[2] / n)
        } else {
            (0.0, 0.0, 0.0)
        };
        let b = table.bvals[i];
        rows.extend_from_slice(&[
            1.0,
            -b * gx * gx,
            -b * gy * gy,
            -b * gz * gz,
            -2.0 * b * gx * gy,
            -2.0 * b * gx * gz,
            -2.0 * b * gy * gz,
        ]);
    }
    DMatrix::from_row_slice(m, 7, &rows)
        .pseudo_inverse(PINV_EPS)
        .map_err(|e| Error::Fit(format!("singular design matrix: {e}")))
}

/// Principal eigenvector and FA of a symmetric tensor `[Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]`.
fn principal_eigen(d: [f64; 6]) -> Option<([f64; 3], f64)> {
    let m = Matrix3::new(d[0], d[3], d[4], d[3], d[1], d[5], d[4], d[5], d[2]);
    let eig = SymmetricEigen::new(m);
    let lambda = [
        eig.eigenvalues[0].max(0.0),
        eig.eigenvalues[1].max(0.0),
        eig.eigenvalues[2].max(0.0),
    ];
    let imax = (0..3).max_by(|&a, &b| lambda[a].total_cmp(&lambda[b]))?;
    let fa = fractional_anisotropy(lambda);
    if !fa.is_finite() || lambda[imax] <= 0.0 {
        return None;
    }
    let col = eig.eigenvectors.column(imax);
    Some(([col[0], col[1], col[2]], fa))
}

fn fractional_anisotropy(l: [f64; 3]) -> f64 {
    let mean = (l[0] + l[1] + l[2]) / 3.0;
    let num = (l[0] - mean).powi(2) + (l[1] - mean).powi(2) + (l[2] - mean).powi(2);
    let den = l[0] * l[0] + l[1] * l[1] + l[2] * l[2];
    if den <= 0.0 {
        0.0
    } else {
        (1.5 * num / den).sqrt()
    }
}

/// Otsu's between-class-variance threshold over a set of positive values.
fn otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if hi <= lo {
        return lo;
    }

    let scale = OTSU_BINS as f64 / (hi - lo);
    let mut hist = [0u64; OTSU_BINS];
    for &v in values {
        let bin = (((v - lo) * scale) as usize).min(OTSU_BINS - 1);
        hist[bin] += 1;
    }

    let total = values.len() as f64;
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let mut weight_bg = 0.0;
    let mut sum_bg = 0.0;
    let mut best_var = -1.0;
    let mut best_bin = 0usize;
    for (i, &c) in hist.iter().enumerate() {
        weight_bg += c as f64;
        if weight_bg == 0.0 {
            continue;
        }
        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }
        sum_bg += i as f64 * c as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum - sum_bg) / weight_fg;
        let var = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if var > best_var {
            best_var = var;
            best_bin = i;
        }
    }
    lo + (best_bin as f64 + 0.5) / scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = norm(v);
        [v[0] / n, v[1] / n, v[2] / n]
    }

    // D = lperp*I + (lpar - lperp) * u uᵀ, principal axis u.
    fn tensor_along(u: [f64; 3], lpar: f64, lperp: f64) -> Matrix3<f64> {
        let mut m = Matrix3::identity() * lperp;
        for r in 0..3 {
            for c in 0..3 {
                m[(r, c)] += (lpar - lperp) * u[r] * u[c];
            }
        }
        m
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

    fn synth_voxel(table: &GradientTable, d: &Matrix3<f64>, s0: f64) -> ArrayD<f32> {
        let nt = table.len();
        Array::from_shape_fn(IxDyn(&[1, 1, 1, nt]), |idx| {
            let t = idx[3];
            let g = table.directions[t];
            let gv = nalgebra::Vector3::new(g[0], g[1], g[2]);
            let q = (gv.transpose() * d * gv)[(0, 0)];
            (s0 * (-table.bvals[t] * q).exp()) as f32
        })
        .into_dyn()
    }

    #[test]
    fn recovers_principal_direction() {
        let table = dwi_scheme();
        let axis = unit([1.0, 2.0, 3.0]);
        let d = tensor_along(axis, 1.7e-3, 0.3e-3);
        let data = synth_voxel(&table, &d, 1000.0);
        let field = fit_dti(
            &data,
            &table,
            &[0],
            &[1, 2, 3, 4, 5, 6, 7],
            FitConfig::default(),
        )
        .unwrap();
        assert!(field.valid[0]);
        let v1 = field.v1[0];
        let dot = (v1[0] * axis[0] + v1[1] * axis[1] + v1[2] * axis[2]).abs();
        assert!(dot > 0.999, "v1={v1:?} dot={dot}");
        assert!(field.fa[0] > 0.6);
    }

    #[test]
    fn background_voxel_is_invalid() {
        let table = dwi_scheme();
        let d = tensor_along([1.0, 0.0, 0.0], 1.7e-3, 0.3e-3);
        let data = synth_voxel(&table, &d, 0.1);
        let field = fit_dti(
            &data,
            &table,
            &[0],
            &[1, 2, 3, 4, 5, 6, 7],
            FitConfig::default(),
        )
        .unwrap();
        assert!(!field.valid[0]);
        assert!(!field.wm_mask(0.2)[0]);
    }

    // 4.5 validation: refitting the same signals with the table transformed by C
    // yields v1 rotated by C — the fit-once-transform identity.
    #[test]
    fn fit_once_transform_matches_refit() {
        let table = dwi_scheme();
        let axis = unit([1.0, 2.0, 3.0]);
        let d = tensor_along(axis, 1.7e-3, 0.3e-3);
        let data = synth_voxel(&table, &d, 1000.0);
        let shell = [1, 2, 3, 4, 5, 6, 7];
        let v1 = fit_dti(&data, &table, &[0], &shell, FitConfig::default())
            .unwrap()
            .v1[0];

        // C: swap x and y axes.
        let directions = table
            .directions
            .iter()
            .map(|&[a, b, c]| [b, a, c])
            .collect();
        let swapped = GradientTable::new(directions, table.bvals.clone()).unwrap();
        let v1_refit = fit_dti(&data, &swapped, &[0], &shell, FitConfig::default())
            .unwrap()
            .v1[0];

        let expected = [v1[1], v1[0], v1[2]];
        let dot =
            (v1_refit[0] * expected[0] + v1_refit[1] * expected[1] + v1_refit[2] * expected[2])
                .abs();
        assert!(dot > 0.999, "refit={v1_refit:?} expected={expected:?}");
    }

    #[test]
    fn otsu_separates_a_bimodal_signal() {
        let mut values = vec![1.0; 40];
        values.extend(vec![100.0; 60]);
        let t = otsu_threshold(&values);
        assert!(t > 1.0 && t < 100.0, "t={t}");
    }

    #[test]
    fn otsu_handles_degenerate_inputs() {
        assert_eq!(otsu_threshold(&[]), 0.0);
        assert_eq!(otsu_threshold(&[5.0, 5.0, 5.0]), 5.0);
    }

    #[test]
    fn wm_proxy_drops_background_and_low_fa() {
        let field = TensorField {
            shape: [2, 2, 1],
            v1: vec![[1.0, 0.0, 0.0]; 4],
            fa: vec![0.5, 0.5, 0.05, 0.5],
            s0: vec![100.0, 1.0, 100.0, 100.0],
            valid: vec![true; 4],
        };
        // bright+anisotropic kept; dark dropped; bright+isotropic dropped.
        assert_eq!(field.wm_proxy(0.2), vec![true, false, false, true]);
    }

    #[test]
    fn isotropic_tensor_has_low_fa() {
        let table = dwi_scheme();
        let d = Matrix3::identity() * 1.0e-3;
        let data = synth_voxel(&table, &d, 1000.0);
        let field = fit_dti(
            &data,
            &table,
            &[0],
            &[1, 2, 3, 4, 5, 6, 7],
            FitConfig::default(),
        )
        .unwrap();
        assert!(field.fa[0] < 0.1);
    }
}
