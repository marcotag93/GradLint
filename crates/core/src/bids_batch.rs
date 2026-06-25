//! BIDS dataset mode: discover DWIs, audit each, and write derivatives.
//!
//! Lives in core so the Python binding and the Rust binary share one path.
//! Writes per-run `report.json` files plus a dataset summary and returns a
//! [`BatchOutcome`]; printing is left to the caller.

use std::fs;
use std::path::{Path, PathBuf};

use crate::frame::GradientFrame;
use crate::io::provenance::InputFile;
use crate::io::{bids, fsl};
use crate::pipeline::{self, AuditOptions};
use crate::report::{self, Status};
use crate::VolumeInfo;

/// A discovered BIDS diffusion run with its FSL sidecars.
pub struct DwiEntry {
    pub dwi: PathBuf,
    pub bvec: PathBuf,
    pub bval: PathBuf,
    /// Candidate sibling masks in the DWI directory, sorted; the grid-matching
    /// one is selected at audit time (others, e.g. a T1w-space mask, are skipped).
    pub masks: Vec<PathBuf>,
    /// Sibling BIDS `.json` sidecar, when present.
    pub sidecar: Option<PathBuf>,
}

/// One graded run in a batch: its short name, DWI path, and verdict.
pub struct BatchItem {
    pub name: String,
    pub dwi: PathBuf,
    pub status: Status,
}

/// Result of auditing a whole BIDS root.
pub struct BatchOutcome {
    /// Most severe status across all runs (drives the process exit code).
    pub worst: Status,
    /// Path of the written `dataset_summary.json`.
    pub summary_path: PathBuf,
    /// Per-run verdicts, in discovery order.
    pub items: Vec<BatchItem>,
}

/// Audit every DWI under a BIDS root, writing per-run and dataset reports.
///
/// Returns the per-run verdicts and the worst status; it does not print, so the
/// caller chooses how to report progress.
pub fn run(root: &Path, options: AuditOptions, step: Option<f64>) -> Result<BatchOutcome, String> {
    let entries = discover(root).map_err(|e| e.to_string())?;
    if entries.is_empty() {
        return Err(format!(
            "no *_dwi NIfTI with sibling bvec/bval (or bvecs/bvals) found under {}",
            root.display()
        ));
    }

    let deriv = root.join("derivatives").join("gradlint");
    fs::create_dir_all(&deriv).map_err(|e| e.to_string())?;

    let mut items = Vec::new();
    let mut results = Vec::new();
    let mut worst = Status::Pass;
    for entry in &entries {
        let name = run_name(&entry.dwi);
        let status = audit_entry(entry, &deriv, &name, options, step)?;
        worst = worst.max(status);
        results.push(serde_json::json!({
            "name": name,
            "dwi": entry.dwi.display().to_string(),
            "status": status.label(),
        }));
        items.push(BatchItem {
            name,
            dwi: entry.dwi.clone(),
            status,
        });
    }

    let summary = serde_json::json!({
        "tool": "gradlint",
        "bids_root": root.display().to_string(),
        "n_datasets": entries.len(),
        "status": worst.label(),
        "results": results,
    });
    let summary_path = deriv.join("dataset_summary.json");
    let text = serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?;
    fs::write(&summary_path, text).map_err(|e| e.to_string())?;
    Ok(BatchOutcome {
        worst,
        summary_path,
        items,
    })
}

fn audit_entry(
    entry: &DwiEntry,
    deriv: &Path,
    name: &str,
    options: AuditOptions,
    step: Option<f64>,
) -> Result<Status, String> {
    let table = fsl::read(&entry.bvec, &entry.bval).map_err(|e| e.to_string())?;
    let (data, info) = crate::read_volume_with_info(&entry.dwi).map_err(|e| e.to_string())?;
    let mut options = options;
    pipeline::apply_geometry(&mut options, &info, GradientFrame::Fsl, step);
    let selected = select_mask(&entry.masks, &info);
    let mut inputs = vec![
        InputFile::of(&entry.bvec).map_err(|e| e.to_string())?,
        InputFile::of(&entry.bval).map_err(|e| e.to_string())?,
    ];
    let mask = match &selected {
        Some((path, mask)) => {
            inputs.push(InputFile::of(path).map_err(|e| e.to_string())?);
            Some(mask.as_slice())
        }
        None => None,
    };
    let mut report =
        pipeline::audit(&table, Some(&data), mask, inputs, options).map_err(|e| e.to_string())?;
    if let Some(note) = sidecar_note(entry.sidecar.as_deref()) {
        report = report.with_note(note);
    }
    let out = deriv.join(format!("{name}_report.json"));
    report::write(&out, &report).map_err(|e| e.to_string())?;
    Ok(report.status)
}

/// Summarize a BIDS sidecar's present fields as one informational note.
fn sidecar_note(path: Option<&Path>) -> Option<String> {
    let sidecar = bids::read(path?).ok()?;
    let mut parts = Vec::new();
    if let Some(pe) = &sidecar.phase_encoding_direction {
        parts.push(format!("PE={pe}"));
    }
    if let Some(trt) = sidecar.total_readout_time {
        parts.push(format!("TRT={trt:.4}s"));
    }
    if let Some(mb) = sidecar.multiband_acceleration_factor {
        parts.push(format!("MB={mb}"));
    }
    (!parts.is_empty()).then(|| format!("BIDS sidecar: {}", parts.join(", ")))
}

/// Recursively find `*_dwi.nii[.gz]` files that have sibling FSL gradients,
/// tolerating both the `.bvec`/`.bval` and HCP `.bvecs`/`.bvals` spellings, and
/// recording an optional sibling mask and `.json` sidecar.
pub fn discover(root: &Path) -> std::io::Result<Vec<DwiEntry>> {
    let mut files = Vec::new();
    walk(root, &mut files)?;
    files.sort();
    let mut entries = Vec::new();
    for path in files {
        let Some(base) = nifti_base(&path) else {
            continue;
        };
        let (Some(bvec), Some(bval)) = (
            first_sibling(&base, &["bvec", "bvecs"]),
            first_sibling(&base, &["bval", "bvals"]),
        ) else {
            continue;
        };
        let sidecar = first_sibling(&base, &["json"]);
        let masks = path.parent().map(find_masks).unwrap_or_default();
        entries.push(DwiEntry {
            dwi: path,
            bvec,
            bval,
            masks,
            sidecar,
        });
    }
    Ok(entries)
}

/// First existing sibling of `base` carrying one of `exts`, in priority order.
fn first_sibling(base: &Path, exts: &[&str]) -> Option<PathBuf> {
    exts.iter()
        .map(|ext| base.with_extension(ext))
        .find(|p| p.is_file())
}

/// Candidate masks in `dir`: NIfTIs whose name contains "mask" (covers
/// `*brainmask*`), skipping macOS `._` sidecars. Sorted for determinism; the
/// caller selects the one whose grid matches the DWI.
fn find_masks(dir: &Path) -> Vec<PathBuf> {
    let mut masks: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_mask_name(p))
            .collect(),
        Err(_) => Vec::new(),
    };
    masks.sort();
    masks
}

fn is_mask_name(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    !name.starts_with("._")
        && lower.contains("mask")
        && (lower.ends_with(".nii") || lower.ends_with(".nii.gz"))
}

/// Load the first candidate mask whose voxel grid matches the DWI; `None` (use
/// the FA proxy) if none match, so a non-DWI-grid mask never aborts the run.
fn select_mask(candidates: &[PathBuf], info: &VolumeInfo) -> Option<(PathBuf, Vec<bool>)> {
    let expected: usize = info.shape.iter().take(3).product();
    for path in candidates {
        if let Ok(mask) = crate::read_mask(path) {
            if mask.len() == expected {
                return Some((path.clone(), mask));
            }
        }
    }
    None
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some("derivatives") {
                continue;
            }
            walk(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// Strip the NIfTI extension from a `*_dwi.nii[.gz]` path, else `None`.
fn nifti_base(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    if name.starts_with("._") {
        return None;
    }
    let stem = name
        .strip_suffix(".nii.gz")
        .or_else(|| name.strip_suffix(".nii"))?;
    if !stem.ends_with("_dwi") {
        return None;
    }
    Some(path.with_file_name(stem))
}

fn run_name(dwi: &Path) -> String {
    nifti_base(dwi)
        .and_then(|b| b.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "dwi".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn discovers_dwi_with_sidecars_and_skips_derivatives() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("sub-01").join("dwi");
        fs::create_dir_all(&sub).unwrap();
        for ext in ["nii.gz", "bvec", "bval"] {
            fs::write(sub.join(format!("sub-01_dwi.{ext}")), "x").unwrap();
        }
        // A DWI without sidecars is ignored.
        fs::write(sub.join("sub-02_dwi.nii"), "x").unwrap();
        // Anything under derivatives is skipped.
        let deriv = dir.path().join("derivatives");
        fs::create_dir_all(&deriv).unwrap();
        fs::write(deriv.join("old_dwi.nii"), "x").unwrap();

        let entries = discover(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].dwi.ends_with("sub-01_dwi.nii.gz"));
        assert!(entries[0].masks.is_empty());
    }

    #[test]
    fn discovers_hcp_spelling_with_mask_and_sidecar() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("sub-01").join("dwi");
        fs::create_dir_all(&sub).unwrap();
        for name in [
            "sub-01_dwi.nii.gz",
            "sub-01_dwi.bvecs",
            "sub-01_dwi.bvals",
            "sub-01_dwi.json",
            "sub-01_dwi_brainmask.nii.gz",
            "sub-01_T1w_brainmask.nii.gz",
            "._sub-01_dwi.bvecs",
        ] {
            fs::write(sub.join(name), "x").unwrap();
        }

        let entries = discover(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert!(entry.bvec.ends_with("sub-01_dwi.bvecs"));
        assert!(entry.bval.ends_with("sub-01_dwi.bvals"));
        assert!(entry.sidecar.as_ref().unwrap().ends_with("sub-01_dwi.json"));
        // Both mask files are recorded as candidates; the grid-matching one is
        // chosen later, in audit_entry.
        assert_eq!(entry.masks.len(), 2);
    }

    #[test]
    fn sidecar_note_lists_present_fields() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sub-01_dwi.json");
        fs::write(
            &path,
            r#"{"PhaseEncodingDirection":"j-","MultibandAccelerationFactor":3}"#,
        )
        .unwrap();
        let note = sidecar_note(Some(&path)).unwrap();
        assert!(note.contains("PE=j-"));
        assert!(note.contains("MB=3"));
        assert!(sidecar_note(None).is_none());
    }

    #[test]
    fn nifti_base_requires_dwi_suffix() {
        assert!(nifti_base(Path::new("/d/sub-01_dwi.nii.gz")).is_some());
        assert!(nifti_base(Path::new("/d/sub-01_T1w.nii.gz")).is_none());
        assert!(nifti_base(Path::new("/d/notes.txt")).is_none());
    }

    #[test]
    fn worst_status_is_most_severe() {
        assert_eq!(Status::Pass.max(Status::Warn), Status::Warn);
        assert_eq!(Status::Flag.max(Status::Warn), Status::Flag);
    }
}
