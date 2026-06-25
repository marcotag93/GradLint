//! Error types for `gradlint-core`.

use thiserror::Error;

/// Errors returned by `gradlint-core`.
#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("NIfTI error: {0}")]
    Nifti(#[from] nifti::NiftiError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("failed to parse {path}: {reason}")]
    Parse { path: String, reason: String },

    #[error("gradient table mismatch: {directions} directions but {bvals} b-values")]
    GradientCountMismatch { directions: usize, bvals: usize },

    #[error("volume mismatch: {gradients} gradients but {volumes} DWI volumes")]
    VolumeCountMismatch { gradients: usize, volumes: usize },

    #[error("mask grid mismatch: mask has {mask} voxels but the DWI grid has {expected}")]
    MaskShapeMismatch { mask: usize, expected: usize },

    #[error("DTI fit failed: {0}")]
    Fit(String),

    #[error("no usable non-b0 shell for flip detection")]
    NoUsableShell,

    #[error("amplitude-encoded bvecs (--strict): {0}")]
    AmplitudeEncoded(String),

    #[error("refusing to overwrite existing file: {path} (pass --force to allow)")]
    OutputExists { path: String },
}

/// Convenience result type for `gradlint-core`.
pub type Result<T> = std::result::Result<T, Error>;
