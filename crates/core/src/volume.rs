//! NIfTI header inspection: shape, voxel sizes, affine, and handedness.

use std::path::Path;
use std::time::{Duration, Instant};

use ndarray::{ArrayD, Axis};
use nifti::{InMemNiftiObject, IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use rayon::prelude::*;

use crate::error::Result;

/// Acquisition-frame handedness, derived from the affine determinant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handedness {
    /// Negative spatial determinant.
    Radiological,
    /// Positive spatial determinant.
    Neurological,
}

/// Geometry extracted from a NIfTI header.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeInfo {
    /// Effective volume shape (without the leading dimensionality entry).
    pub shape: Vec<usize>,
    /// Voxel sizes along the first three axes (mm).
    pub voxel_sizes: [f64; 3],
    /// Voxel-to-world affine (4x4, row-major).
    pub affine: [[f64; 4]; 4],
    /// Number of DWI volumes (4th dimension, or 1 if 3-D).
    pub num_volumes: usize,
}

impl VolumeInfo {
    /// Determinant of the 3x3 spatial part of the affine.
    #[must_use]
    pub fn spatial_determinant(&self) -> f64 {
        let a = &self.affine;
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    }

    /// Frame handedness from the sign of the spatial determinant.
    #[must_use]
    pub fn handedness(&self) -> Handedness {
        if self.spatial_determinant() < 0.0 {
            Handedness::Radiological
        } else {
            Handedness::Neurological
        }
    }
}

/// Read geometry from a NIfTI file header. Volume data is not loaded.
pub fn read_header<P: AsRef<Path>>(path: P) -> Result<VolumeInfo> {
    let header = NiftiHeader::from_file(path)?;
    from_header(&header)
}

fn from_header(header: &NiftiHeader) -> Result<VolumeInfo> {
    let dim = header.dim()?;
    let shape: Vec<usize> = dim.iter().map(|&d| d as usize).collect();
    let num_volumes = if shape.len() >= 4 { shape[3] } else { 1 };
    let voxel_sizes = [
        header.pixdim[1] as f64,
        header.pixdim[2] as f64,
        header.pixdim[3] as f64,
    ];
    Ok(VolumeInfo {
        shape,
        voxel_sizes,
        affine: compute_affine(header),
        num_volumes,
    })
}

fn compute_affine(h: &NiftiHeader) -> [[f64; 4]; 4] {
    if h.sform_code > 0 {
        sform_affine(h)
    } else if h.qform_code > 0 {
        qform_affine(h)
    } else {
        scale_affine(h)
    }
}

fn sform_affine(h: &NiftiHeader) -> [[f64; 4]; 4] {
    let row = |r: [f32; 4]| [r[0] as f64, r[1] as f64, r[2] as f64, r[3] as f64];
    [
        row(h.srow_x),
        row(h.srow_y),
        row(h.srow_z),
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn scale_affine(h: &NiftiHeader) -> [[f64; 4]; 4] {
    [
        [h.pixdim[1] as f64, 0.0, 0.0, 0.0],
        [0.0, h.pixdim[2] as f64, 0.0, 0.0],
        [0.0, 0.0, h.pixdim[3] as f64, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// NIfTI-1 qform: rotation from unit quaternion (b, c, d), scaled by voxel
// sizes with the qfac sign on the z column (see nifti1.h).
fn qform_affine(h: &NiftiHeader) -> [[f64; 4]; 4] {
    let (b, c, d) = (h.quatern_b as f64, h.quatern_c as f64, h.quatern_d as f64);
    let a = (1.0 - (b * b + c * c + d * d)).max(0.0).sqrt();
    let r = [
        [
            a * a + b * b - c * c - d * d,
            2.0 * (b * c - a * d),
            2.0 * (b * d + a * c),
        ],
        [
            2.0 * (b * c + a * d),
            a * a + c * c - b * b - d * d,
            2.0 * (c * d - a * b),
        ],
        [
            2.0 * (b * d - a * c),
            2.0 * (c * d + a * b),
            a * a + d * d - b * b - c * c,
        ],
    ];
    let (dx, dy, dz) = (h.pixdim[1] as f64, h.pixdim[2] as f64, h.pixdim[3] as f64);
    let qfac = if h.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
    [
        [
            r[0][0] * dx,
            r[0][1] * dy,
            r[0][2] * dz * qfac,
            h.quatern_x as f64,
        ],
        [
            r[1][0] * dx,
            r[1][1] * dy,
            r[1][2] * dz * qfac,
            h.quatern_y as f64,
        ],
        [
            r[2][0] * dx,
            r[2][1] * dy,
            r[2][2] * dz * qfac,
            h.quatern_z as f64,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Load an in-memory NIfTI object, the single read entry point for all readers.
#[cfg(not(feature = "libdeflate"))]
fn open_object(path: &Path) -> Result<InMemNiftiObject> {
    Ok(ReaderOptions::new().read_file(path)?)
}

/// Load an in-memory NIfTI object, decompressing `.nii.gz` via libdeflate.
///
/// Any deviation from the fast path falls back to the default reader, so the
/// result is byte-identical to [`ReaderOptions::read_file`] in all cases.
#[cfg(feature = "libdeflate")]
fn open_object(path: &Path) -> Result<InMemNiftiObject> {
    if is_gz(path) {
        if let Some(object) = open_object_libdeflate(path) {
            return Ok(object);
        }
    }
    Ok(ReaderOptions::new().read_file(path)?)
}

#[cfg(feature = "libdeflate")]
fn is_gz(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"))
}

#[cfg(feature = "libdeflate")]
fn open_object_libdeflate(path: &Path) -> Option<InMemNiftiObject> {
    // The gzip ISIZE trailer wraps at 4 GB, so size the output from the header.
    let header = NiftiHeader::from_file(path).ok()?;
    let out_len = gz_uncompressed_len(&header).ok()?;
    let compressed = std::fs::read(path).ok()?;
    let mut out = vec![0u8; out_len];
    let actual = libdeflater::Decompressor::new()
        .gzip_decompress(&compressed, &mut out)
        .ok()?;
    drop(compressed);
    InMemNiftiObject::from_reader(&out[..actual]).ok()
}

#[cfg(feature = "libdeflate")]
fn gz_uncompressed_len(header: &NiftiHeader) -> Result<usize> {
    let n_voxels: usize = header.dim()?.iter().map(|&d| d as usize).product();
    let bytes_per_voxel = (header.bitpix as usize).div_ceil(8);
    Ok(header.vox_offset as usize + n_voxels * bytes_per_voxel)
}

/// Read a full NIfTI volume as an `f32` array (3-D or 4-D).
///
/// `f32` storage halves the allocation versus `f64`; integer DWI converts to
/// `f32` exactly, and all downstream arithmetic promotes to `f64` at use.
pub fn read_volume(path: impl AsRef<Path>) -> Result<ArrayD<f32>> {
    Ok(open_object(path.as_ref())?
        .into_volume()
        .into_ndarray::<f32>()?)
}

/// Read a NIfTI volume together with its geometry in a single pass.
pub fn read_volume_with_info(path: impl AsRef<Path>) -> Result<(ArrayD<f32>, VolumeInfo)> {
    let object = open_object(path.as_ref())?;
    let info = from_header(object.header())?;
    let data = object.into_volume().into_ndarray::<f32>()?;
    Ok((data, info))
}

/// Wall-clock split of a NIfTI read: stream decode vs `f32` conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadTimings {
    /// Header read plus gzip-stream decompression into memory.
    pub decompress: Duration,
    /// `f32` allocation and integer/float conversion of every voxel.
    pub convert: Duration,
}

/// [`read_volume_with_info`] instrumented to time decode vs `f32` conversion.
pub fn read_volume_with_info_timed(
    path: impl AsRef<Path>,
) -> Result<(ArrayD<f32>, VolumeInfo, ReadTimings)> {
    let t0 = Instant::now();
    let object = open_object(path.as_ref())?;
    let info = from_header(object.header())?;
    let decompress = t0.elapsed();

    let t1 = Instant::now();
    let data = object.into_volume().into_ndarray::<f32>()?;
    let convert = t1.elapsed();

    Ok((
        data,
        info,
        ReadTimings {
            decompress,
            convert,
        },
    ))
}

/// Read a mask volume as a flattened row-major `[x][y][z]` boolean mask.
///
/// Nonzero voxels are kept, and a 4-D mask uses its first volume. The voxel
/// order matches the DTI fit grid (`(x·ny + y)·nz + z`).
pub fn read_mask(path: impl AsRef<Path>) -> Result<Vec<bool>> {
    Ok(mask_from_volume(&read_volume(path)?))
}

fn mask_from_volume(data: &ArrayD<f32>) -> Vec<bool> {
    let shape = data.shape();
    let nx = shape.first().copied().unwrap_or(0);
    let ny = shape.get(1).copied().unwrap_or(1);
    let nz = shape.get(2).copied().unwrap_or(1);
    let mut mask = Vec::with_capacity(nx * ny * nz);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let v = if data.ndim() >= 4 {
                    data[[x, y, z, 0]]
                } else {
                    data[[x, y, z]]
                };
                mask.push(v != 0.0);
            }
        }
    }
    mask
}

/// Mean signal for a chosen subset of volumes, optionally restricted to a mask.
///
/// Each volume is summed independently in `f64` (parallel over `volumes`), so the
/// result is order-stable. Sub-4-D data returns the single mean for every entry.
#[must_use]
pub fn mean_for_volumes(data: &ArrayD<f32>, volumes: &[usize], mask: Option<&[bool]>) -> Vec<f64> {
    if data.ndim() < 4 {
        let mean = masked_mean(data.iter().copied(), mask);
        return vec![mean; volumes.len()];
    }
    let axis = Axis(data.ndim() - 1);
    volumes
        .par_iter()
        .map(|&t| masked_mean(data.index_axis(axis, t).iter().copied(), mask))
        .collect()
}

fn masked_mean(values: impl Iterator<Item = f32>, mask: Option<&[bool]>) -> f64 {
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    match mask {
        Some(m) => {
            for (v, &keep) in values.zip(m) {
                if keep {
                    sum += f64::from(v);
                    n += 1;
                }
            }
        }
        None => {
            for v in values {
                sum += f64::from(v);
                n += 1;
            }
        }
    }
    if n > 0 {
        sum / n as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn header_4d() -> NiftiHeader {
        NiftiHeader {
            dim: [4, 2, 3, 4, 5, 0, 0, 0],
            pixdim: [1.0, 2.0, 2.0, 2.5, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        }
    }

    #[test]
    fn extracts_shape_and_volume_count() {
        let info = from_header(&header_4d()).unwrap();
        assert_eq!(info.shape, vec![2, 3, 4, 5]);
        assert_eq!(info.num_volumes, 5);
        assert_eq!(info.voxel_sizes, [2.0, 2.0, 2.5]);
    }

    #[test]
    fn sform_radiological_handedness() {
        let h = NiftiHeader {
            sform_code: 1,
            srow_x: [-2.0, 0.0, 0.0, 0.0],
            srow_y: [0.0, 2.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 2.5, 0.0],
            ..header_4d()
        };
        let info = from_header(&h).unwrap();
        assert!(info.spatial_determinant() < 0.0);
        assert_eq!(info.handedness(), Handedness::Radiological);
    }

    #[test]
    fn qform_identity_affine_is_scaled() {
        let h = NiftiHeader {
            sform_code: 0,
            qform_code: 1,
            ..header_4d()
        };
        let info = from_header(&h).unwrap();
        assert!((info.affine[0][0] - 2.0).abs() < 1e-9);
        assert!((info.affine[1][1] - 2.0).abs() < 1e-9);
        assert!((info.affine[2][2] - 2.5).abs() < 1e-9);
        assert_eq!(info.handedness(), Handedness::Neurological);
    }

    #[test]
    fn mean_for_volumes_matches_subset() {
        let data = Array::from_shape_vec(
            ndarray::IxDyn(&[2, 1, 1, 3]),
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        )
        .unwrap();
        assert_eq!(mean_for_volumes(&data, &[0, 2], None), vec![5.5, 16.5]);
    }

    #[test]
    fn mean_for_volumes_respects_mask() {
        let data = Array::from_shape_vec(
            ndarray::IxDyn(&[2, 1, 1, 3]),
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        )
        .unwrap();
        assert_eq!(
            mean_for_volumes(&data, &[1], Some(&[true, false])),
            vec![2.0]
        );
    }

    #[test]
    fn mask_keeps_nonzero_voxels_in_grid_order() {
        let data =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 2, 1]), vec![0.0, 1.0, 0.0, 2.0]).unwrap();
        assert_eq!(mask_from_volume(&data), vec![false, true, false, true]);
    }

    #[test]
    fn mask_from_4d_uses_first_volume() {
        let data =
            Array::from_shape_vec(ndarray::IxDyn(&[1, 2, 1, 2]), vec![0.0, 9.0, 3.0, 0.0]).unwrap();
        assert_eq!(mask_from_volume(&data), vec![false, true]);
    }
}

#[cfg(all(test, feature = "libdeflate"))]
mod libdeflate_tests {
    use super::*;
    use ndarray::Array;
    use nifti::writer::WriterOptions;

    #[test]
    fn libdeflate_read_matches_reference() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vol.nii.gz");
        let data = Array::from_shape_vec(
            ndarray::IxDyn(&[3, 4, 5, 2]),
            (0..120).map(|v| v as f32).collect(),
        )
        .unwrap();
        WriterOptions::new(&path).write_nifti(&data).unwrap();

        let fast = open_object_libdeflate(&path)
            .expect("libdeflate fast path engaged")
            .into_volume()
            .into_ndarray::<f32>()
            .unwrap();
        let reference = ReaderOptions::new()
            .read_file(&path)
            .unwrap()
            .into_volume()
            .into_ndarray::<f32>()
            .unwrap();

        assert_eq!(fast, reference);
    }
}
