//! Core algorithms for gradlint.

pub mod angular;
pub mod bids_batch;
pub mod candidate;
pub mod coherence;
pub mod consistency;
pub mod error;
pub mod flip;
pub mod frame;
pub mod gradient;
pub mod io;
pub mod pipeline;
pub mod repair;
pub mod report;
pub mod scheme_qc;
pub mod shell;
pub mod tensor;
pub mod volume;

pub use angular::{AngularConfig, AngularSummary, DuplicatePair, ShellAngular};
pub use bids_batch::{BatchItem, BatchOutcome};
pub use candidate::{all_candidates, Candidate, N_CANDIDATES};
pub use coherence::{coherence_index, CoherenceConfig};
pub use error::{Error, Result};
pub use flip::{
    decide, detect_flip, detect_flip_on_shell, detect_flip_on_shell_timed,
    detect_flip_on_shell_timed_with_glyphs, detect_flip_on_shell_with_glyphs, detect_flip_timed,
    detect_flip_timed_with_glyphs, detect_flip_with_glyphs, rank_candidates, select_working_shell,
    CandidateScore, Decision, DetectTimings, FlipConfig, FlipResult, GlyphData,
};
pub use frame::{
    default_step, frame_for, frames_divergent, voxel_frame_map, FrameMaps, GradientFrame,
};
pub use gradient::GradientTable;
pub use pipeline::{
    apply_geometry, audit, audit_timed, audit_timed_with_glyphs, audit_with_glyphs, detect,
    detect_with_glyphs, inspect, repair, repair_with_glyphs, AuditOptions, RepairOutput,
    RepairSpec,
};
pub use repair::{apply_transform, prepare_output, Repair};
pub use report::{RepairInfo, Report, SchemeTable, Status};
pub use shell::{B0Drift, B0Summary, Shell, ShellConfig, ShellSummary};
pub use tensor::{fit_dti, FitConfig, TensorField};
pub use volume::{
    read_mask, read_volume, read_volume_with_info, read_volume_with_info_timed, Handedness,
    ReadTimings, VolumeInfo,
};

/// Crate version, taken from `Cargo.toml`.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
