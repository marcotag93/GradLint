//! Fiber-coherence index over a principal-eigenvector field.
//!
//! For each white-matter voxel, step `±step·v1` and trilinearly sample the `v1`
//! field at the landing point, accumulating `|v1(x) · v1(neighbor)|`. Stepping
//! anchors orientation to fixed image-space anatomy so global wrong flips lose
//! coherence, while sign-ambiguous eigenvectors are aligned before interpolation
//! and compared with absolute dot products.

use rayon::prelude::*;

use crate::gradient::norm;

/// Smallest vector norm treated as a usable direction.
const MIN_NORM: f64 = 1e-9;

/// Tuning for the fiber-coherence index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoherenceConfig {
    /// Step length along `v1`, in voxels.
    pub step: f64,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self { step: 1.0 }
    }
}

/// Fiber-coherence index and the count of valid neighbor samples.
///
/// For every usable masked voxel both step directions are evaluated; a step
/// that leaves the mask contributes zero. The index normalizes the accumulated
/// `|v1(x) · v1(neighbor)|` by the fixed pair count `2 × usable voxels`, so it
/// lies in `[0, 1]` and the same denominator is shared by all 48 candidates —
/// a wrong orientation that steps off the anatomy is penalized, not hidden.
#[must_use]
pub fn coherence_index(
    shape: [usize; 3],
    v1: &[[f64; 3]],
    mask: &[bool],
    config: CoherenceConfig,
) -> (f64, usize) {
    let (ny, nz) = (shape[1], shape[2]);
    let nvox = shape[0] * ny * nz;

    let (sum, valid, pairs) = (0..nvox)
        .into_par_iter()
        .filter(|&lin| mask[lin])
        .map(|lin| {
            let v = v1[lin];
            if norm(v) < MIN_NORM {
                return (0.0, 0usize, 0usize);
            }
            let z = lin % nz;
            let y = (lin / nz) % ny;
            let x = lin / (nz * ny);
            let mut s = 0.0;
            let mut c = 0usize;
            for sign in [config.step, -config.step] {
                let pos = [
                    x as f64 + sign * v[0],
                    y as f64 + sign * v[1],
                    z as f64 + sign * v[2],
                ];
                if let Some(n) = sample(shape, v1, mask, pos, v) {
                    s += dot(v, n).abs();
                    c += 1;
                }
            }
            (s, c, 2usize)
        })
        .reduce(|| (0.0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    if pairs == 0 {
        (0.0, 0)
    } else {
        (sum / pairs as f64, valid)
    }
}

/// Trilinearly sample the `v1` field at a continuous position, aligning each
/// in-mask corner to `reference` before blending. `None` if no valid corner.
fn sample(
    shape: [usize; 3],
    v1: &[[f64; 3]],
    mask: &[bool],
    pos: [f64; 3],
    reference: [f64; 3],
) -> Option<[f64; 3]> {
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let base = [pos[0].floor(), pos[1].floor(), pos[2].floor()];
    let frac = [pos[0] - base[0], pos[1] - base[1], pos[2] - base[2]];

    let mut acc = [0.0; 3];
    let mut wsum = 0.0;
    for (dx, wx) in [(0isize, 1.0 - frac[0]), (1, frac[0])] {
        for (dy, wy) in [(0isize, 1.0 - frac[1]), (1, frac[1])] {
            for (dz, wz) in [(0isize, 1.0 - frac[2]), (1, frac[2])] {
                let w = wx * wy * wz;
                if w <= 0.0 {
                    continue;
                }
                let xi = base[0] as isize + dx;
                let yi = base[1] as isize + dy;
                let zi = base[2] as isize + dz;
                if xi < 0
                    || yi < 0
                    || zi < 0
                    || xi >= nx as isize
                    || yi >= ny as isize
                    || zi >= nz as isize
                {
                    continue;
                }
                let lin = ((xi as usize) * ny + yi as usize) * nz + zi as usize;
                if !mask[lin] {
                    continue;
                }
                let c = v1[lin];
                if norm(c) < MIN_NORM {
                    continue;
                }
                let s = if dot(c, reference) < 0.0 { -1.0 } else { 1.0 };
                acc[0] += w * s * c[0];
                acc[1] += w * s * c[1];
                acc[2] += w * s * c[2];
                wsum += w;
            }
        }
    }
    if wsum <= 0.0 {
        return None;
    }
    let mean = [acc[0] / wsum, acc[1] / wsum, acc[2] / wsum];
    let n = norm(mean);
    if n < MIN_NORM {
        None
    } else {
        Some([mean[0] / n, mean[1] / n, mean[2] / n])
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Straight fiber tube along `a`, masked to a thin neighborhood. Stepping
    // along the true direction stays inside; a wrong rotation leaves the mask
    // and loses coherence — the same anchoring the real index relies on.
    fn tube_field(n: usize, a: [f64; 3], radius: f64) -> ([usize; 3], Vec<[f64; 3]>, Vec<bool>) {
        let c = (n as f64 - 1.0) / 2.0;
        let m = norm(a);
        let an = [a[0] / m, a[1] / m, a[2] / m];
        let mut v1 = Vec::with_capacity(n * n);
        let mut mask = Vec::with_capacity(n * n);
        for x in 0..n {
            for y in 0..n {
                let p = [x as f64 - c, y as f64 - c, 0.0];
                let t = p[0] * an[0] + p[1] * an[1] + p[2] * an[2];
                let perp = [p[0] - t * an[0], p[1] - t * an[1], p[2] - t * an[2]];
                if norm(perp) <= radius {
                    v1.push(an);
                    mask.push(true);
                } else {
                    v1.push([0.0, 0.0, 0.0]);
                    mask.push(false);
                }
            }
        }
        ([n, n, 1], v1, mask)
    }

    #[test]
    fn identity_beats_permutation_on_tube() {
        let (shape, v1, mask) = tube_field(25, [1.0, 2.0, 0.0], 0.9);
        let cfg = CoherenceConfig { step: 2.0 };
        let (idn, n_id) = coherence_index(shape, &v1, &mask, cfg);
        let swapped: Vec<[f64; 3]> = v1.iter().map(|&[a, b, c]| [b, a, c]).collect();
        let (perm, _) = coherence_index(shape, &swapped, &mask, cfg);
        assert!(n_id > 0);
        assert!(idn > perm + 0.05, "identity={idn} permuted={perm}");
    }

    #[test]
    fn identity_beats_single_axis_flip() {
        let (shape, v1, mask) = tube_field(25, [1.0, 2.0, 0.0], 0.9);
        let cfg = CoherenceConfig { step: 2.0 };
        let (idn, _) = coherence_index(shape, &v1, &mask, cfg);
        let flipped: Vec<[f64; 3]> = v1.iter().map(|&[a, b, c]| [-a, b, c]).collect();
        let (flip, _) = coherence_index(shape, &flipped, &mask, cfg);
        assert!(idn > flip + 0.05, "identity={idn} flip_x={flip}");
    }

    #[test]
    fn empty_mask_yields_zero() {
        let shape = [4, 4, 4];
        let v1 = vec![[1.0, 0.0, 0.0]; 64];
        let mask = vec![false; 64];
        assert_eq!(
            coherence_index(shape, &v1, &mask, CoherenceConfig::default()),
            (0.0, 0)
        );
    }
}
