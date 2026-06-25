//! Canonical QC report aggregating every result for the rendering layer.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use serde::{Deserialize, Serialize};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::angular::AngularSummary;
use crate::error::Result;
use crate::flip::{Decision, FlipResult};
use crate::gradient::{GradientTable, NormStats};
use crate::io::provenance::InputFile;
use crate::shell::{B0Drift, ShellSummary};

/// Report schema version. Bump on any breaking field change.
pub const SCHEMA_VERSION: u32 = 1;

/// Overall QC status, mapped onto the report verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Status {
    /// Scheme is consistent: `0`.
    Pass,
    /// Ambiguous result needing review.
    Warn,
    /// A flip/permutation was detected.
    Flag,
}

impl Status {
    /// Status-only CLI exit code (`0` PASS/FLAG, `3` WARN).
    #[must_use]
    pub fn exit_code(self) -> i32 {
        match self {
            Status::Pass => 0,
            Status::Warn => 3,
            Status::Flag => 0,
        }
    }

    /// Uppercase verdict word (`PASS`/`WARN`/`FLAG`) for terminal and JSON.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Warn => "WARN",
            Status::Flag => "FLAG",
        }
    }
}

impl From<Decision> for Status {
    fn from(decision: Decision) -> Self {
        match decision {
            Decision::Pass => Status::Pass,
            Decision::Warn => Status::Warn,
            Decision::Flag => Status::Flag,
        }
    }
}

/// The gradient scheme as rendered: directions and b-values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemeTable {
    pub directions: Vec<[f64; 3]>,
    pub bvals: Vec<f64>,
}

impl SchemeTable {
    /// Snapshot a gradient table for the report.
    #[must_use]
    pub fn from_table(table: &GradientTable) -> Self {
        Self {
            directions: table.directions.clone(),
            bvals: table.bvals.clone(),
        }
    }
}

/// The repair applied to the gradient table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairInfo {
    pub matrix: [[f64; 3]; 3],
    pub label: String,
    pub outputs: Vec<String>,
}

/// One shell as `(nominal_b, count)` for the recovery summary.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShellCount {
    pub nominal_b: f64,
    pub count: usize,
}

/// Summary of an opt-in `recompute-bval` run (amplitude → b recovery).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BvalRecoverySummary {
    pub b_nominal: f64,
    pub max_norm: f64,
    pub b_min: f64,
    pub b_max: f64,
    pub before: Vec<ShellCount>,
    pub after: Vec<ShellCount>,
    pub outputs: Vec<String>,
}

/// Canonical QC report consumed by the Python rendering layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    pub tool: String,
    pub tool_version: String,
    /// RFC 3339 UTC timestamp of when the report was created.
    pub timestamp: String,
    pub status: Status,
    pub inputs: Vec<InputFile>,
    pub scheme: SchemeTable,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shells: Option<ShellSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b0_drift: Option<B0Drift>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angular: Option<AngularSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flip: Option<FlipResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repair: Option<RepairInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub norm_stats: Option<NormStats>,
    pub notes: Vec<String>,
}

impl Report {
    /// Start a report for a scheme, defaulting to PASS until a result sets it.
    #[must_use]
    pub fn new(scheme: SchemeTable) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            tool: "gradlint".to_string(),
            tool_version: crate::version().to_string(),
            timestamp: now_rfc3339(),
            status: Status::Pass,
            inputs: Vec::new(),
            scheme,
            shells: None,
            b0_drift: None,
            angular: None,
            flip: None,
            repair: None,
            norm_stats: None,
            notes: Vec::new(),
        }
    }

    /// Set the overall status explicitly.
    #[must_use]
    pub fn with_status(mut self, status: Status) -> Self {
        self.status = status;
        self
    }

    /// Record the hashed input files.
    #[must_use]
    pub fn with_inputs(mut self, inputs: Vec<InputFile>) -> Self {
        self.inputs = inputs;
        self
    }

    /// Attach the b-value / shell QC summary.
    #[must_use]
    pub fn with_shells(mut self, shells: ShellSummary) -> Self {
        self.shells = Some(shells);
        self
    }

    /// Attach the b0 signal-drift fit.
    #[must_use]
    pub fn with_b0_drift(mut self, drift: B0Drift) -> Self {
        self.b0_drift = Some(drift);
        self
    }

    /// Attach the angular scheme QC summary.
    #[must_use]
    pub fn with_angular(mut self, angular: AngularSummary) -> Self {
        self.angular = Some(angular);
        self
    }

    /// Attach the flip-detection result, deriving the overall status from it.
    #[must_use]
    pub fn with_flip(mut self, flip: FlipResult) -> Self {
        self.status = Status::from(flip.decision);
        self.flip = Some(flip);
        self
    }

    /// Record the repair applied to the gradient table.
    #[must_use]
    pub fn with_repair(mut self, repair: RepairInfo) -> Self {
        self.repair = Some(repair);
        self
    }

    /// Attach the non-b0 unit-norm statistics (amplitude-encoding check).
    #[must_use]
    pub fn with_norm_stats(mut self, stats: NormStats) -> Self {
        self.norm_stats = Some(stats);
        self
    }

    /// Append a free-form note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Pipeline gate exit code: block only when WARN has no applied repair.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        if self.status == Status::Warn && self.repair.is_none() {
            3
        } else {
            0
        }
    }
}

/// Write a report as pretty-printed JSON.
pub fn write(path: impl AsRef<Path>, report: &Report) -> Result<()> {
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, report)?;
    Ok(())
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flip::CandidateScore;

    fn scheme() -> SchemeTable {
        let table = GradientTable::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0.0, 1000.0, 1000.0],
        )
        .unwrap();
        SchemeTable::from_table(&table)
    }

    fn flip(decision: Decision) -> FlipResult {
        let m = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let best = CandidateScore {
            label: "-x+y+z".to_string(),
            matrix: m,
            is_identity: decision == Decision::Pass,
            coherence: 0.9,
            n_samples: 10,
        };
        let (recommended_transform, recommended_label) = match decision {
            Decision::Flag => (Some(m), Some("-x+y+z".to_string())),
            _ => (None, None),
        };
        FlipResult {
            working_b: 1000.0,
            n_wm_voxels: 5,
            mask_mean_fa: 0.4,
            ranking: vec![best.clone()],
            best: best.clone(),
            runner_up: best,
            identity_coherence: 0.4,
            margin: 0.2,
            relative_margin: 0.2,
            decision,
            recommended_transform,
            recommended_label,
        }
    }

    #[test]
    fn new_is_stamped_and_passes() {
        let r = Report::new(scheme());
        assert_eq!(r.schema_version, SCHEMA_VERSION);
        assert_eq!(r.tool, "gradlint");
        assert!(!r.tool_version.is_empty());
        assert!(r.timestamp.contains('T'));
        assert_eq!(r.status, Status::Pass);
    }

    #[test]
    fn flip_drives_status() {
        assert_eq!(
            Report::new(scheme()).with_flip(flip(Decision::Flag)).status,
            Status::Flag
        );
        assert_eq!(
            Report::new(scheme()).with_flip(flip(Decision::Warn)).status,
            Status::Warn
        );
        assert_eq!(
            Report::new(scheme()).with_flip(flip(Decision::Pass)).status,
            Status::Pass
        );
    }

    #[test]
    fn status_exit_codes_match_contract() {
        assert_eq!(Status::Pass.exit_code(), 0);
        assert_eq!(Status::Warn.exit_code(), 3);
        assert_eq!(Status::Flag.exit_code(), 0);
    }

    #[test]
    fn report_exit_code_blocks_only_unrepaired_warn() {
        assert_eq!(
            Report::new(scheme())
                .with_flip(flip(Decision::Pass))
                .exit_code(),
            0
        );
        assert_eq!(
            Report::new(scheme())
                .with_flip(flip(Decision::Flag))
                .exit_code(),
            0
        );

        let warn = Report::new(scheme()).with_flip(flip(Decision::Warn));
        assert_eq!(warn.exit_code(), 3);

        let repaired_warn = warn.with_repair(RepairInfo {
            matrix: [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            label: "-x+y+z".to_string(),
            outputs: vec!["fixed.bvec".to_string(), "fixed.bval".to_string()],
        });
        assert_eq!(repaired_warn.exit_code(), 0);
    }

    #[test]
    fn status_labels_are_uppercase() {
        assert_eq!(Status::Pass.label(), "PASS");
        assert_eq!(Status::Warn.label(), "WARN");
        assert_eq!(Status::Flag.label(), "FLAG");
    }

    #[test]
    fn status_orders_pass_warn_flag() {
        assert!(Status::Pass < Status::Warn);
        assert!(Status::Warn < Status::Flag);
        assert_eq!(Status::Pass.max(Status::Warn), Status::Warn);
        assert_eq!(Status::Flag.max(Status::Warn), Status::Flag);
    }

    #[test]
    fn json_roundtrips() {
        let r = Report::new(scheme())
            .with_flip(flip(Decision::Flag))
            .with_repair(RepairInfo {
                matrix: [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                label: "-x+y+z".to_string(),
                outputs: vec!["dwi.bvec".to_string()],
            })
            .with_note("auto-repaired");
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<Report>(&json).unwrap(), r);
    }

    #[test]
    fn write_emits_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        let r = Report::new(scheme());
        write(&path, &r).unwrap();
        let back: Report = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(back, r);
    }
}
