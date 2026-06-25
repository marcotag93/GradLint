//! Schema-versioned provenance log written alongside repaired outputs.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::angular::AngularSummary;
use crate::error::Result;
use crate::flip::FlipResult;
use crate::shell::ShellSummary;

/// Provenance schema version. Bump on any breaking field change.
pub const SCHEMA_VERSION: u32 = 1;

/// A hashed reference to an input file, for tamper-evident provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputFile {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

/// Affine-derived frame facts used to re-express the corrected table between the
/// FSL and MRtrix stored frames, recorded so a cross-format repair is auditable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameProvenance {
    /// `"fsl"` or `"mrtrix"` — the frame the input table was read in.
    pub input_format: String,
    pub affine_determinant_sign: f64,
    /// Voxel → world rotation `R`.
    pub rotation: [[f64; 3]; 3],
    /// FSL stored → voxel map `A_f`.
    pub fsl_voxel_map: [[f64; 3]; 3],
    /// World stored → voxel map `Rᵀ`.
    pub world_voxel_map: [[f64; 3]; 3],
    pub frames_divergent: bool,
    pub emitted_fsl: bool,
    pub emitted_mrtrix: bool,
}

impl InputFile {
    /// Hash a file's contents with SHA-256.
    pub fn of(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = fs::read(path)?;
        let digest = Sha256::digest(&data);
        Ok(Self {
            path: path.display().to_string(),
            sha256: hex(&digest),
            bytes: data.len() as u64,
        })
    }
}

/// Machine-readable record of what gradlint inspected and changed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub schema_version: u32,
    pub tool: String,
    pub tool_version: String,
    /// RFC 3339 UTC timestamp of when the record was created.
    pub timestamp: String,
    pub inputs: Vec<InputFile>,
    /// b-value / shell QC of the detected scheme.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shells: Option<ShellSummary>,
    /// Angular scheme QC (uniformity, conditioning, duplicates).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angular: Option<AngularSummary>,
    /// Flip-detection result: candidate ranking, margins, and decision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flip: Option<FlipResult>,
    /// The signed-permutation matrix applied to repair the table.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_transform: Option<[[f64; 3]; 3]>,
    /// Human-readable label of the applied transform, e.g. `-x+y+z`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_label: Option<String>,
    /// Frame facts used to re-express the corrected table between formats.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame: Option<FrameProvenance>,
    pub outputs: Vec<String>,
    pub notes: Vec<String>,
}

impl Provenance {
    /// Create a record stamped with the current tool version and UTC time.
    #[must_use]
    pub fn new(inputs: Vec<InputFile>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            tool: "gradlint".to_string(),
            tool_version: crate::version().to_string(),
            timestamp: now_rfc3339(),
            inputs,
            shells: None,
            angular: None,
            flip: None,
            applied_transform: None,
            applied_label: None,
            frame: None,
            outputs: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Attach the detected scheme metrics (shell and angular QC).
    #[must_use]
    pub fn with_metrics(mut self, shells: ShellSummary, angular: AngularSummary) -> Self {
        self.shells = Some(shells);
        self.angular = Some(angular);
        self
    }

    /// Attach a flip result and the repair it recommends.
    #[must_use]
    pub fn with_flip(mut self, flip: &FlipResult) -> Self {
        self.applied_transform = flip.recommended_transform;
        self.applied_label = flip.recommended_label.clone();
        self.flip = Some(flip.clone());
        self
    }

    /// Record the files written by the repair.
    #[must_use]
    pub fn with_outputs(mut self, outputs: Vec<String>) -> Self {
        self.outputs = outputs;
        self
    }

    /// Attach the affine-derived frame facts used for cross-format emission.
    #[must_use]
    pub fn with_frame(mut self, frame: FrameProvenance) -> Self {
        self.frame = Some(frame);
        self
    }
}

/// Write a provenance record as pretty-printed JSON.
pub fn write(path: impl AsRef<Path>, provenance: &Provenance) -> Result<()> {
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, provenance)?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn input(name: &str, body: &str) -> InputFile {
        let dir = tempdir().unwrap();
        let p = dir.path().join(name);
        fs::write(&p, body).unwrap();
        InputFile::of(&p).unwrap()
    }

    #[test]
    fn new_is_stamped() {
        let p = Provenance::new(vec![input("dwi.bvec", "0 1 0\n")]);
        assert_eq!(p.schema_version, SCHEMA_VERSION);
        assert_eq!(p.tool, "gradlint");
        assert!(!p.tool_version.is_empty());
        assert!(p.timestamp.contains('T'), "timestamp={}", p.timestamp);
    }

    #[test]
    fn hashes_file_contents() {
        let empty = input("a", "");
        assert_eq!(empty.bytes, 0);
        assert_eq!(
            empty.sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_ne!(input("b", "data").sha256, empty.sha256);
    }

    #[test]
    fn json_roundtrip() {
        let p = Provenance::new(vec![input("a", "x")]);
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Provenance>(&json).unwrap(), p);
    }

    #[test]
    fn with_flip_records_applied_transform() {
        use crate::flip::{CandidateScore, Decision};
        let m = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let s = CandidateScore {
            label: "-x+y+z".to_string(),
            matrix: m,
            is_identity: false,
            coherence: 0.9,
            n_samples: 10,
        };
        let flip = FlipResult {
            working_b: 1000.0,
            n_wm_voxels: 5,
            mask_mean_fa: 0.4,
            ranking: vec![s.clone()],
            best: s.clone(),
            runner_up: s,
            identity_coherence: 0.4,
            margin: 0.2,
            relative_margin: 0.2,
            decision: Decision::Flag,
            recommended_transform: Some(m),
            recommended_label: Some("-x+y+z".to_string()),
        };
        let p = Provenance::new(vec![input("a", "x")])
            .with_flip(&flip)
            .with_outputs(vec!["out.bvec".to_string()]);
        assert_eq!(p.applied_transform, Some(m));
        assert_eq!(p.applied_label.as_deref(), Some("-x+y+z"));
        assert_eq!(p.outputs, vec!["out.bvec".to_string()]);
        let back: Provenance = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back, p);
    }
}
