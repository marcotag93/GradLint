//! Resolve a gradient table's frame into the image voxel-index frame.
//!
//! The coherence index steps through the voxel grid, so the principal
//! eigenvector must live in that grid's frame. FSL bvecs are image-relative and
//! flip their x-axis when the affine determinant is positive; MRtrix `.b` tables
//! are in world coordinates and rotate by the inverse of the affine rotation.

use crate::volume::VolumeInfo;

/// Identity map: the no-op frame correction.
pub const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Smallest column norm treated as a usable affine axis.
const MIN_NORM: f64 = 1e-9;

/// Coordinate frame a gradient table is expressed in, as read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientFrame {
    /// FSL bvec: image-relative, with the affine-determinant x-flip convention.
    Fsl,
    /// MRtrix `.b`: world / scanner coordinates.
    World,
}

/// Frame a gradient table is in given whether an MRtrix `.b` (`--grad`) was used:
/// `--grad` tables are world-frame, FSL `bvec`/`bval` are image-relative.
#[must_use]
pub fn frame_for(grad_present: bool) -> GradientFrame {
    if grad_present {
        GradientFrame::World
    } else {
        GradientFrame::Fsl
    }
}

/// Linear map taking a gradient direction into the image voxel-index frame.
///
/// Applied to `v1` before the coherence index so stepping is anchored to the
/// fixed image anatomy. Returns [`IDENTITY`] when no correction is needed.
#[must_use]
pub fn voxel_frame_map(frame: GradientFrame, info: &VolumeInfo) -> [[f64; 3]; 3] {
    match frame {
        GradientFrame::Fsl => {
            if info.spatial_determinant() > 0.0 {
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            } else {
                IDENTITY
            }
        }
        GradientFrame::World => world_to_voxel(info),
    }
}

/// Default coherence step in voxels, scaling the physical step (~4 mm) with
/// voxel size so margins stay above threshold on fine-resolution data.
#[must_use]
pub fn default_step(info: &VolumeInfo) -> f64 {
    let mean = info.voxel_sizes.iter().sum::<f64>() / 3.0;
    if mean > 0.0 {
        (4.0 / mean).max(1.0)
    } else {
        1.0
    }
}

/// Transpose of the affine's normalized rotation: world direction → voxel frame.
///
/// Assumes the affine's spatial columns are orthogonal (no shear): each column is
/// normalized independently and the transpose is used as the inverse rotation.
/// This holds for real DWI affines; a sheared affine would be slightly off.
fn world_to_voxel(info: &VolumeInfo) -> [[f64; 3]; 3] {
    let a = &info.affine;
    let mut col_norm = [0.0f64; 3];
    for (j, norm) in col_norm.iter_mut().enumerate() {
        *norm = (a[0][j] * a[0][j] + a[1][j] * a[1][j] + a[2][j] * a[2][j]).sqrt();
    }
    if col_norm.iter().any(|&n| n < MIN_NORM) {
        return IDENTITY;
    }
    // R[i][j] = a[i][j] / |col j| is the voxel→world rotation; return Rᵀ.
    let mut rt = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rt[j][i] = a[i][j] / col_norm[j];
        }
    }
    rt
}

/// Frame maps for re-expressing a corrected gradient table between the FSL
/// (image/voxel) and MRtrix (world) stored frames at emit time.
///
/// Every map is derived from the image affine through [`voxel_frame_map`] — the
/// same primitive detection uses — so ingest and emit share one source of truth.
/// The corrected directions are carried in the input's stored frame; emitting
/// the *other* format applies the affine relabel, while same-format emission
/// collapses to the identity (`A_f·A_f = I`, `R·Rᵀ = I`) and is unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMaps {
    input: GradientFrame,
    /// FSL stored → voxel (`A_f`; a diagonal ±1 involution, so `A_f⁻¹ = A_f`).
    fsl_to_voxel: [[f64; 3]; 3],
    /// World stored → voxel (`Rᵀ`).
    world_to_voxel: [[f64; 3]; 3],
    /// Sign of the affine spatial determinant.
    pub determinant_sign: f64,
    /// FSL and world interpretations of this image diverge (see
    /// [`frames_divergent`]).
    pub divergent: bool,
}

impl FrameMaps {
    /// Resolve the emit maps for an `input` frame from the image geometry.
    #[must_use]
    pub fn resolve(input: GradientFrame, info: &VolumeInfo) -> Self {
        Self {
            input,
            fsl_to_voxel: voxel_frame_map(GradientFrame::Fsl, info),
            world_to_voxel: voxel_frame_map(GradientFrame::World, info),
            determinant_sign: info.spatial_determinant().signum(),
            divergent: frames_divergent(info),
        }
    }

    /// The input stored frame → voxel map (`A_f` for FSL, `Rᵀ` for world).
    fn input_to_voxel(&self) -> [[f64; 3]; 3] {
        match self.input {
            GradientFrame::Fsl => self.fsl_to_voxel,
            GradientFrame::World => self.world_to_voxel,
        }
    }

    /// Map corrected (input-frame) directions into the FSL stored frame.
    /// Identity when the input is already FSL.
    #[must_use]
    pub fn to_fsl(&self) -> [[f64; 3]; 3] {
        matmul3(&self.fsl_to_voxel, &self.input_to_voxel())
    }

    /// Map corrected (input-frame) directions into the MRtrix world frame.
    /// Identity when the input is already MRtrix `.b`.
    #[must_use]
    pub fn to_world(&self) -> [[f64; 3]; 3] {
        matmul3(&self.rotation(), &self.input_to_voxel())
    }

    /// Input frame the corrected directions are carried in.
    #[must_use]
    pub fn input(&self) -> GradientFrame {
        self.input
    }

    /// Voxel → world rotation `R` (transpose of `Rᵀ`).
    #[must_use]
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        transpose3(&self.world_to_voxel)
    }

    /// FSL stored → voxel map `A_f`.
    #[must_use]
    pub fn fsl_voxel_map(&self) -> [[f64; 3]; 3] {
        self.fsl_to_voxel
    }

    /// World stored → voxel map `Rᵀ`.
    #[must_use]
    pub fn world_voxel_map(&self) -> [[f64; 3]; 3] {
        self.world_to_voxel
    }
}

/// Whether the FSL and world interpretations of this image diverge beyond the
/// FSL determinant (x-flip) convention.
///
/// True only for a positive determinant whose world rotation, rounded to a
/// signed permutation, negates two axes — the `diag(-2,-2,2)`-class affine where
/// feeding FSL but consuming world (or vice-versa) silently corrupts the table.
#[must_use]
pub fn frames_divergent(info: &VolumeInfo) -> bool {
    info.spatial_determinant() > 0.0 && negated_axis_count(&world_to_voxel(info)) >= 2
}

/// Count axes negated by a rotation rounded to its nearest signed permutation
/// (per row, the sign of the dominant-magnitude entry).
fn negated_axis_count(m: &[[f64; 3]; 3]) -> usize {
    (0..3)
        .filter(|&i| {
            let dom = (0..3).max_by(|&a, &b| m[i][a].abs().total_cmp(&m[i][b].abs()));
            dom.is_some_and(|j| m[i][j] < 0.0)
        })
        .count()
}

/// Multiply two 3×3 matrices (`a · b`).
fn matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Transpose a 3×3 matrix.
fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            t[j][i] = v;
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(affine: [[f64; 4]; 4], voxel: [f64; 3]) -> VolumeInfo {
        VolumeInfo {
            shape: vec![2, 2, 2, 1],
            voxel_sizes: voxel,
            affine,
            num_volumes: 1,
        }
    }

    fn neurological() -> [[f64; 4]; 4] {
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn radiological() -> [[f64; 4]; 4] {
        [
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn fsl_flips_x_on_positive_determinant() {
        let map = voxel_frame_map(GradientFrame::Fsl, &info(neurological(), [2.0; 3]));
        assert_eq!(map, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn fsl_is_identity_on_negative_determinant() {
        let map = voxel_frame_map(GradientFrame::Fsl, &info(radiological(), [2.0; 3]));
        assert_eq!(map, IDENTITY);
    }

    #[test]
    fn world_axis_aligned_is_identity() {
        let map = voxel_frame_map(GradientFrame::World, &info(neurological(), [2.0; 3]));
        assert_eq!(map, IDENTITY);
    }

    #[test]
    fn world_recovers_axis_swap() {
        let swap = [
            [0.0, 3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let map = voxel_frame_map(GradientFrame::World, &info(swap, [2.0, 3.0, 4.0]));
        // World y-axis is voxel x-axis here, so Rᵀ maps world (0,1,0) → voxel (1,0,0).
        assert_eq!(map, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn default_step_scales_with_voxel_size() {
        assert!((default_step(&info(neurological(), [2.0; 3])) - 2.0).abs() < 1e-12);
        assert!((default_step(&info(neurological(), [1.25; 3])) - 3.2).abs() < 1e-12);
        assert_eq!(default_step(&info(neurological(), [4.0; 3])), 1.0);
        assert_eq!(default_step(&info(neurological(), [8.0; 3])), 1.0);
    }

    const X_FLIP: [[f64; 3]; 3] = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Positive determinant with two negated axes: the diag(-2,-2,2) bug class.
    fn divergent_affine() -> [[f64; 4]; 4] {
        [
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    // 1.37° in-plane x–y tilt on a radiological (det < 0) affine.
    fn oblique_radiological() -> [[f64; 4]; 4] {
        let (c, s) = (1.37_f64.to_radians().cos(), 1.37_f64.to_radians().sin());
        [
            [-2.0 * c, -2.0 * s, 0.0, 0.0],
            [2.0 * s, 2.0 * c, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn frames_divergent_only_on_multi_axis_positive_determinant() {
        assert!(frames_divergent(&info(divergent_affine(), [2.0; 3])));
        assert!(!frames_divergent(&info(neurological(), [2.0; 3])));
        assert!(!frames_divergent(&info(radiological(), [2.0; 3])));
        assert!(!frames_divergent(&info(oblique_radiological(), [2.0; 3])));
    }

    #[test]
    fn same_format_emit_maps_are_identity() {
        for affine in [neurological(), radiological(), divergent_affine()] {
            let i = info(affine, [2.0; 3]);
            assert_eq!(
                FrameMaps::resolve(GradientFrame::Fsl, &i).to_fsl(),
                IDENTITY
            );
            assert_eq!(
                FrameMaps::resolve(GradientFrame::World, &i).to_world(),
                IDENTITY
            );
        }
    }

    #[test]
    fn cross_format_emit_relabels_neurological_by_x_flip() {
        // Neurological axis-aligned (det > 0): the FSL bvec is x-flipped from
        // world, so each cross-format emit applies that single x-flip.
        let i = info(neurological(), [2.0; 3]);
        assert_eq!(
            FrameMaps::resolve(GradientFrame::Fsl, &i).to_world(),
            X_FLIP
        );
        assert_eq!(
            FrameMaps::resolve(GradientFrame::World, &i).to_fsl(),
            X_FLIP
        );
    }
}
