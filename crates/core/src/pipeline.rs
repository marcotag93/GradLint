//! End-to-end QC pipelines that assemble metrics into a canonical [`Report`].

use std::path::PathBuf;

use ndarray::ArrayD;

use crate::angular::{analyze, AngularConfig};
use crate::consistency::check_volume_count;
use crate::error::{Error, Result};
use crate::flip::{
    detect_flip, detect_flip_on_shell, detect_flip_on_shell_timed,
    detect_flip_on_shell_timed_with_glyphs, detect_flip_on_shell_with_glyphs, detect_flip_timed,
    detect_flip_timed_with_glyphs, detect_flip_with_glyphs, Decision, DetectTimings, FlipConfig,
    FlipResult, GlyphData,
};
use crate::frame::{default_step, voxel_frame_map, FrameMaps, GradientFrame};
use crate::gradient::GradientTable;
use crate::io::provenance::{self, FrameProvenance, InputFile, Provenance};
use crate::io::{fsl, mrtrix};
use crate::repair::{apply_transform, prepare_output, Repair};
use crate::report::{BvalRecoverySummary, RepairInfo, Report, SchemeTable, ShellCount};
use crate::scheme_qc;
use crate::shell::{b0_drift, detect_shells, summarize, B0Drift, Shell, ShellConfig};
use crate::volume::mean_for_volumes;
use crate::volume::VolumeInfo;

/// Coherence-mask mean FA below this is treated as whole-brain-like: the
/// flip-detection margin shrinks, so the report advises supplying a WM mask.
/// Sits between a measured whole-brain mask (~0.26) and the FA proxy (~0.39).
const LOW_MASK_FA: f64 = 0.30;

/// Tuning shared across the QC pipelines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuditOptions {
    pub shell: ShellConfig,
    pub angular: AngularConfig,
    pub flip: FlipConfig,
    /// Explicit working-shell b-value for flip detection; auto-selected if `None`.
    pub working_shell: Option<f64>,
    /// Promote severe scheme-quality issues to WARN (notes are always emitted).
    pub strict: bool,
    /// Unit-norm tolerance for the amplitude-encoding check.
    pub norm_tolerance: f64,
}

impl Default for AuditOptions {
    fn default() -> Self {
        Self {
            shell: ShellConfig::default(),
            angular: AngularConfig::default(),
            flip: FlipConfig::default(),
            working_shell: None,
            strict: false,
            norm_tolerance: scheme_qc::DEFAULT_NORM_TOLERANCE,
        }
    }
}

/// Resolve the flip-detection frame map and coherence step from image geometry.
pub fn apply_geometry(
    options: &mut AuditOptions,
    info: &VolumeInfo,
    frame: GradientFrame,
    step: Option<f64>,
) {
    options.flip.frame_map = voxel_frame_map(frame, info);
    options.flip.coherence.step = step.unwrap_or_else(|| default_step(info));
}

/// Scheme-only QC: shells + angular metrics, no image required.
pub fn inspect(
    table: &GradientTable,
    inputs: Vec<InputFile>,
    options: AuditOptions,
) -> Result<Report> {
    scheme_qc::apply(scheme_report(table, inputs, options), options.strict)
}

/// Assemble the scheme metrics without applying scheme-quality gating; the
/// public entry points add the [`scheme_qc`] pass once, at the end.
fn scheme_report(table: &GradientTable, inputs: Vec<InputFile>, options: AuditOptions) -> Report {
    Report::new(SchemeTable::from_table(table))
        .with_inputs(inputs)
        .with_shells(summarize(&table.bvals, options.shell))
        .with_angular(analyze(table, options.shell, options.angular))
        .with_norm_stats(table.norm_stats(options.shell.b0_threshold, options.norm_tolerance))
}

/// Full audit: scheme QC plus flip detection and b0 drift when an image is given.
pub fn audit(
    table: &GradientTable,
    data: Option<&ArrayD<f32>>,
    mask: Option<&[bool]>,
    inputs: Vec<InputFile>,
    options: AuditOptions,
) -> Result<Report> {
    let mut report = scheme_report(table, inputs, options);
    if let Some(data) = data {
        check_dwi_inputs(table, data, mask)?;
        report = report.with_b0_drift(compute_drift(data, table, mask, options.shell));
        report = note_low_mask_fa(report.with_flip(run_flip(data, table, mask, options)?));
    }
    scheme_qc::apply(report, options.strict)
}

pub fn audit_with_glyphs(
    table: &GradientTable,
    data: &ArrayD<f32>,
    mask: Option<&[bool]>,
    inputs: Vec<InputFile>,
    options: AuditOptions,
) -> Result<(Report, GlyphData)> {
    check_dwi_inputs(table, data, mask)?;
    let mut report = scheme_report(table, inputs, options);
    report = report.with_b0_drift(compute_drift(data, table, mask, options.shell));
    let (flip, glyphs) = run_flip_with_glyphs(data, table, mask, options)?;
    report = note_low_mask_fa(report.with_flip(flip));
    Ok((scheme_qc::apply(report, options.strict)?, glyphs))
}

/// [`audit`] over an image, returning the flip-stage timing split alongside it.
///
/// One instrumented run: the report is identical to [`audit`]; the extra
/// [`DetectTimings`] separates the single DTI fit from the 48-candidate
/// coherence ranking. Pair with [`crate::volume::read_volume_with_info_timed`]
/// for the read-side decode/convert split.
pub fn audit_timed(
    table: &GradientTable,
    data: &ArrayD<f32>,
    mask: Option<&[bool]>,
    inputs: Vec<InputFile>,
    options: AuditOptions,
) -> Result<(Report, DetectTimings)> {
    check_dwi_inputs(table, data, mask)?;
    let mut report = scheme_report(table, inputs, options);
    report = report.with_b0_drift(compute_drift(data, table, mask, options.shell));
    let (flip, timings) = run_flip_timed(data, table, mask, options)?;
    report = note_low_mask_fa(report.with_flip(flip));
    Ok((scheme_qc::apply(report, options.strict)?, timings))
}

pub fn audit_timed_with_glyphs(
    table: &GradientTable,
    data: &ArrayD<f32>,
    mask: Option<&[bool]>,
    inputs: Vec<InputFile>,
    options: AuditOptions,
) -> Result<(Report, DetectTimings, GlyphData)> {
    check_dwi_inputs(table, data, mask)?;
    let mut report = scheme_report(table, inputs, options);
    report = report.with_b0_drift(compute_drift(data, table, mask, options.shell));
    let (flip, timings, glyphs) = run_flip_timed_with_glyphs(data, table, mask, options)?;
    report = note_low_mask_fa(report.with_flip(flip));
    Ok((scheme_qc::apply(report, options.strict)?, timings, glyphs))
}

/// Flip detection only (requires an image): a lean scheme + flip report.
pub fn detect(
    table: &GradientTable,
    data: &ArrayD<f32>,
    mask: Option<&[bool]>,
    inputs: Vec<InputFile>,
    options: AuditOptions,
) -> Result<Report> {
    check_dwi_inputs(table, data, mask)?;
    Ok(note_low_mask_fa(
        Report::new(SchemeTable::from_table(table))
            .with_inputs(inputs)
            .with_flip(run_flip(data, table, mask, options)?),
    ))
}

pub fn detect_with_glyphs(
    table: &GradientTable,
    data: &ArrayD<f32>,
    mask: Option<&[bool]>,
    inputs: Vec<InputFile>,
    options: AuditOptions,
) -> Result<(Report, GlyphData)> {
    check_dwi_inputs(table, data, mask)?;
    let (flip, glyphs) = run_flip_with_glyphs(data, table, mask, options)?;
    let report = Report::new(SchemeTable::from_table(table))
        .with_inputs(inputs)
        .with_flip(flip);
    Ok((note_low_mask_fa(report), glyphs))
}

/// Where a repair writes its corrected tables and provenance.
#[derive(Debug, Clone)]
pub struct RepairSpec {
    pub bvec: PathBuf,
    pub bval: PathBuf,
    pub mrtrix: Option<PathBuf>,
    pub provenance: Option<PathBuf>,
    pub dry_run: bool,
    pub in_place: bool,
    /// Apply the best-candidate remap on a WARN decision too, overriding the
    /// conservative withhold. Never affects PASS or the detection logic.
    pub force_repair: bool,
    /// Affine-derived maps that re-express the corrected table into each output
    /// format's stored frame. `None` writes the input-frame table verbatim to
    /// every format (correct only when output format = input format).
    pub frame: Option<FrameMaps>,
}

/// Result of a repair run: the report plus the corrected table and any backups.
#[derive(Debug, Clone)]
pub struct RepairOutput {
    pub report: Report,
    pub repaired: Option<GradientTable>,
    pub backups: Vec<PathBuf>,
}

/// Audit, then apply and write the recommended repair when a flip is flagged.
///
/// A repair is built only on `Decision::Flag` (WARN/PASS never auto-repair),
/// unless `spec.force_repair` is set, which also applies the best-candidate
/// remap on a WARN decision (PASS and the detection logic are never affected).
/// With `dry_run`, the corrected table is computed but nothing is written.
pub fn repair(
    table: &GradientTable,
    data: Option<&ArrayD<f32>>,
    mask: Option<&[bool]>,
    inputs: Vec<InputFile>,
    options: AuditOptions,
    spec: &RepairSpec,
) -> Result<RepairOutput> {
    let report = audit(table, data, mask, inputs.clone(), options)?;
    repair_from_report(table, report, inputs, options, spec)
}

pub fn repair_with_glyphs(
    table: &GradientTable,
    data: &ArrayD<f32>,
    mask: Option<&[bool]>,
    inputs: Vec<InputFile>,
    options: AuditOptions,
    spec: &RepairSpec,
) -> Result<(RepairOutput, GlyphData)> {
    let (report, glyphs) = audit_with_glyphs(table, data, mask, inputs.clone(), options)?;
    let output = repair_from_report(table, report, inputs, options, spec)?;
    Ok((output, glyphs))
}

fn repair_from_report(
    table: &GradientTable,
    mut report: Report,
    inputs: Vec<InputFile>,
    options: AuditOptions,
    spec: &RepairSpec,
) -> Result<RepairOutput> {
    if spec.frame.is_some_and(|f| f.divergent) {
        report = note_divergent_frames(report);
    }
    let Some(flip) = report.flip.clone() else {
        return Ok(RepairOutput::passthrough(report));
    };
    let rep = match Repair::from_flip(table, &flip) {
        Some(rep) => rep,
        None if spec.force_repair && flip.decision == Decision::Warn => {
            match Repair::force_from_best(table, &flip) {
                Some(rep) => {
                    report.notes.push(format!(
                        "repair force-applied on a WARN decision (--force-repair); margin \
                         {:.2}% is below the {:.0}% auto-repair threshold — verify the result.",
                        flip.relative_margin * 100.0,
                        options.flip.margin_threshold * 100.0
                    ));
                    rep
                }
                None => return Ok(RepairOutput::passthrough(report)),
            }
        }
        None => return Ok(RepairOutput::passthrough(report)),
    };

    if spec.dry_run {
        let info = RepairInfo {
            matrix: rep.matrix,
            label: rep.label.clone(),
            outputs: Vec::new(),
        };
        return Ok(RepairOutput {
            report: report.with_repair(info),
            repaired: Some(rep.table),
            backups: Vec::new(),
        });
    }

    let mut outputs = Vec::new();
    let mut backups = Vec::new();
    push_backup(&mut backups, prepare_output(&spec.bvec, spec.in_place)?);
    push_backup(&mut backups, prepare_output(&spec.bval, spec.in_place)?);
    fsl::write(
        &spec.bvec,
        &spec.bval,
        &emit_table(&rep.table, spec.frame.as_ref(), FrameMaps::to_fsl),
    )?;
    outputs.push(path_str(&spec.bvec));
    outputs.push(path_str(&spec.bval));
    if let Some(grad) = &spec.mrtrix {
        push_backup(&mut backups, prepare_output(grad, spec.in_place)?);
        mrtrix::write(
            grad,
            &emit_table(&rep.table, spec.frame.as_ref(), FrameMaps::to_world),
        )?;
        outputs.push(path_str(grad));
    }

    let info = RepairInfo {
        matrix: rep.matrix,
        label: rep.label.clone(),
        outputs: outputs.clone(),
    };
    let report = report.with_repair(info);
    if let Some(path) = &spec.provenance {
        write_provenance(path, &report, &flip, inputs, outputs, spec)?;
    }
    Ok(RepairOutput {
        report,
        repaired: Some(rep.table),
        backups,
    })
}

impl RepairOutput {
    fn passthrough(report: Report) -> Self {
        Self {
            report,
            repaired: None,
            backups: Vec::new(),
        }
    }
}

/// Where an opt-in `recompute-bval` run writes its recovered tables.
#[derive(Debug, Clone)]
pub struct RecomputeSpec {
    pub bvec: PathBuf,
    pub bval: PathBuf,
    pub mrtrix: Option<PathBuf>,
    pub provenance: Option<PathBuf>,
    pub dry_run: bool,
    pub in_place: bool,
}

/// Result of a `recompute-bval` run: the summary plus the recovered table.
#[derive(Debug, Clone)]
pub struct RecomputeOutput {
    pub summary: BvalRecoverySummary,
    pub table: GradientTable,
    pub backups: Vec<PathBuf>,
}

/// Recover amplitude-encoded b-values into a corrected bval + unit bvec.
///
/// Opt-in only — never reached from `repair`, which keeps the trust-bval
/// contract. Computes `b_i = b_nominal·(|g_i|/max)²`, writes the corrected
/// tables (honoring `dry_run`/`in_place`) and an optional provenance log, and
/// returns a before/after shell summary.
pub fn recompute_bval(
    table: &GradientTable,
    inputs: Vec<InputFile>,
    shell: ShellConfig,
    spec: &RecomputeSpec,
) -> Result<RecomputeOutput> {
    let (recovered, b_nominal, max_norm) = table.recover_bvals(shell.b0_threshold)?;
    let before = shell_counts(&table.bvals, shell);
    let after = shell_counts(&recovered.bvals, shell);
    let (b_min, b_max) = nonb0_range(&recovered.bvals, shell.b0_threshold);

    if spec.dry_run {
        return Ok(RecomputeOutput {
            summary: BvalRecoverySummary {
                b_nominal,
                max_norm,
                b_min,
                b_max,
                before,
                after,
                outputs: Vec::new(),
            },
            table: recovered,
            backups: Vec::new(),
        });
    }

    let mut outputs = Vec::new();
    let mut backups = Vec::new();
    push_backup(&mut backups, prepare_output(&spec.bvec, spec.in_place)?);
    push_backup(&mut backups, prepare_output(&spec.bval, spec.in_place)?);
    fsl::write(&spec.bvec, &spec.bval, &recovered)?;
    outputs.push(path_str(&spec.bvec));
    outputs.push(path_str(&spec.bval));
    if let Some(grad) = &spec.mrtrix {
        push_backup(&mut backups, prepare_output(grad, spec.in_place)?);
        mrtrix::write(grad, &recovered)?;
        outputs.push(path_str(grad));
    }
    if let Some(path) = &spec.provenance {
        let mut prov = Provenance::new(inputs).with_outputs(outputs.clone());
        prov.notes.push(format!(
            "recompute-bval: recovered b-values from amplitude-encoded bvecs \
             (b_nominal={b_nominal:.0}, max |g|={max_norm:.4}); directions unit-normalized."
        ));
        provenance::write(path, &prov)?;
    }
    Ok(RecomputeOutput {
        summary: BvalRecoverySummary {
            b_nominal,
            max_norm,
            b_min,
            b_max,
            before,
            after,
            outputs,
        },
        table: recovered,
        backups,
    })
}

fn shell_counts(bvals: &[f64], config: ShellConfig) -> Vec<ShellCount> {
    detect_shells(bvals, config)
        .into_iter()
        .map(|s| ShellCount {
            nominal_b: s.nominal_b,
            count: s.count,
        })
        .collect()
}

fn nonb0_range(bvals: &[f64], b0_threshold: f64) -> (f64, f64) {
    let nz: Vec<f64> = bvals
        .iter()
        .copied()
        .filter(|&b| b > b0_threshold)
        .collect();
    if nz.is_empty() {
        (0.0, 0.0)
    } else {
        (
            nz.iter().copied().fold(f64::INFINITY, f64::min),
            nz.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }
}

fn run_flip(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    options: AuditOptions,
) -> Result<FlipResult> {
    match options.working_shell {
        Some(target) => {
            let shell = pick_shell(table, options.shell, target)?;
            detect_flip_on_shell(data, table, mask, &shell, options.flip)
        }
        None => detect_flip(data, table, mask, options.flip),
    }
}

fn run_flip_timed(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    options: AuditOptions,
) -> Result<(FlipResult, DetectTimings)> {
    match options.working_shell {
        Some(target) => {
            let shell = pick_shell(table, options.shell, target)?;
            detect_flip_on_shell_timed(data, table, mask, &shell, options.flip)
        }
        None => detect_flip_timed(data, table, mask, options.flip),
    }
}

fn run_flip_with_glyphs(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    options: AuditOptions,
) -> Result<(FlipResult, GlyphData)> {
    match options.working_shell {
        Some(target) => {
            let shell = pick_shell(table, options.shell, target)?;
            detect_flip_on_shell_with_glyphs(data, table, mask, &shell, options.flip)
        }
        None => detect_flip_with_glyphs(data, table, mask, options.flip),
    }
}

fn run_flip_timed_with_glyphs(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    options: AuditOptions,
) -> Result<(FlipResult, DetectTimings, GlyphData)> {
    match options.working_shell {
        Some(target) => {
            let shell = pick_shell(table, options.shell, target)?;
            detect_flip_on_shell_timed_with_glyphs(data, table, mask, &shell, options.flip)
        }
        None => detect_flip_timed_with_glyphs(data, table, mask, options.flip),
    }
}

fn pick_shell(table: &GradientTable, config: ShellConfig, target_b: f64) -> Result<Shell> {
    detect_shells(&table.bvals, config)
        .into_iter()
        .filter(|s| !s.is_b0)
        .min_by(|a, b| {
            (a.nominal_b - target_b)
                .abs()
                .total_cmp(&(b.nominal_b - target_b).abs())
        })
        .ok_or(Error::NoUsableShell)
}

fn compute_drift(
    data: &ArrayD<f32>,
    table: &GradientTable,
    mask: Option<&[bool]>,
    shell: ShellConfig,
) -> B0Drift {
    let indices = table.b0_indices(shell.b0_threshold);
    let signal = mean_for_volumes(data, &indices, mask);
    b0_drift(&indices, &signal)
}

/// Guard the gradient↔volume and mask↔grid invariants at the IO boundary.
///
/// A typed error here replaces the out-of-bounds panic (or, for an oversized
/// mask, the silently misaligned coherence index) that an unchecked mismatch
/// would otherwise trigger inside the per-voxel fit and the b0-drift mean.
fn check_dwi_inputs(
    table: &GradientTable,
    data: &ArrayD<f32>,
    mask: Option<&[bool]>,
) -> Result<()> {
    check_volume_count(table, Some(num_volumes(data)))?;
    if let Some(m) = mask {
        let expected = spatial_voxels(data);
        if m.len() != expected {
            return Err(Error::MaskShapeMismatch {
                mask: m.len(),
                expected,
            });
        }
    }
    Ok(())
}

/// Volumes available for indexing in a DWI array (`1` for sub-4-D data).
fn num_volumes(data: &ArrayD<f32>) -> usize {
    if data.ndim() >= 4 {
        data.shape()[3]
    } else {
        1
    }
}

/// Spatial voxel count `nx·ny·nz` of a DWI array — the expected mask length.
fn spatial_voxels(data: &ArrayD<f32>) -> usize {
    data.shape().iter().take(3).product()
}

fn write_provenance(
    path: &PathBuf,
    report: &Report,
    flip: &FlipResult,
    inputs: Vec<InputFile>,
    outputs: Vec<String>,
    spec: &RepairSpec,
) -> Result<()> {
    let mut prov = Provenance::new(inputs);
    if let (Some(shells), Some(angular)) = (report.shells.clone(), report.angular.clone()) {
        prov = prov.with_metrics(shells, angular);
    }
    prov = prov.with_flip(flip).with_outputs(outputs);
    if let Some(frame) = &spec.frame {
        prov = prov.with_frame(frame_provenance(frame, spec.mrtrix.is_some()));
    }
    provenance::write(path, &prov)
}

/// Re-express the corrected (input-frame) table into a target stored frame.
/// Verbatim (no conversion) when no frame maps are available.
fn emit_table(
    table: &GradientTable,
    frame: Option<&FrameMaps>,
    map: impl Fn(&FrameMaps) -> [[f64; 3]; 3],
) -> GradientTable {
    match frame {
        Some(frame) => apply_transform(table, &map(frame)),
        None => table.clone(),
    }
}

fn frame_provenance(frame: &FrameMaps, emitted_mrtrix: bool) -> FrameProvenance {
    let input_format = match frame.input() {
        GradientFrame::Fsl => "fsl",
        GradientFrame::World => "mrtrix",
    };
    FrameProvenance {
        input_format: input_format.to_string(),
        affine_determinant_sign: frame.determinant_sign,
        rotation: frame.rotation(),
        fsl_voxel_map: frame.fsl_voxel_map(),
        world_voxel_map: frame.world_voxel_map(),
        frames_divergent: frame.divergent,
        emitted_fsl: true,
        emitted_mrtrix,
    }
}

/// Advisory when the image's FSL and world frames diverge: a cross-format
/// hand-off can silently corrupt the table. The verdict is unchanged.
fn note_divergent_frames(report: Report) -> Report {
    report.with_note(
        "FSL and world frames diverge for this image (positive affine determinant with a \
         multi-axis rotation); confirm the gradient is in the frame you intend before \
         consuming a cross-format output"
            .to_string(),
    )
}

/// Append an advisory note when the coherence mask looks whole-brain (low mean
/// FA), which weakens flip-detection sensitivity. The verdict is unchanged.
fn note_low_mask_fa(report: Report) -> Report {
    match &report.flip {
        Some(flip) if flip.mask_mean_fa > 0.0 && flip.mask_mean_fa < LOW_MASK_FA => {
            let note = format!(
                "coherence mask mean FA {:.2} is low (looks whole-brain) — flip-detection \
                 sensitivity is reduced; supply a WM mask (FA-thresholded) for a wider margin",
                flip.mask_mean_fa
            );
            report.with_note(note)
        }
        _ => report,
    }
}

fn push_backup(backups: &mut Vec<PathBuf>, backup: Option<PathBuf>) {
    if let Some(path) = backup {
        backups.push(path);
    }
}

fn path_str(path: &std::path::Path) -> String {
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::all_candidates;
    use crate::coherence::CoherenceConfig;
    use crate::repair::apply_transform;
    use crate::report::Status;
    use nalgebra::{Matrix3, Vector3};
    use ndarray::{Array, ArrayD, IxDyn};
    use tempfile::tempdir;

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    }

    fn scheme() -> GradientTable {
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

    fn tensor_for(an: [f64; 3]) -> Matrix3<f64> {
        let mut d = Matrix3::identity() * 0.3e-3;
        for r in 0..3 {
            for col in 0..3 {
                d[(r, col)] += (1.7e-3 - 0.3e-3) * an[r] * an[col];
            }
        }
        d
    }

    // Crossing-fiber phantom: multiple non-parallel tubes. The mix of
    // orientations breaks the per-fiber antipodal degeneracy, so the true
    // convention is uniquely identifiable.
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
                if (perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt() <= radius {
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

    fn phantom(table: &GradientTable) -> ArrayD<f32> {
        crossing(21, &[[3.0, 1.0, 2.0], [1.0, -2.0, 3.0]], 1.5, table)
    }

    fn flip_x(table: &GradientTable) -> GradientTable {
        let m = all_candidates()
            .into_iter()
            .find(|c| c.label == "-x+y+z")
            .unwrap()
            .matrix;
        apply_transform(table, &m)
    }

    fn options() -> AuditOptions {
        AuditOptions {
            flip: FlipConfig {
                coherence: CoherenceConfig { step: 2.0 },
                ..FlipConfig::default()
            },
            ..AuditOptions::default()
        }
    }

    fn flip_case() -> (GradientTable, GradientTable, ArrayD<f32>) {
        let truth = scheme();
        let data = phantom(&truth);
        (truth.clone(), flip_x(&truth), data)
    }

    fn flip_with_mask_fa(fa: f64) -> FlipResult {
        use crate::flip::{CandidateScore, Decision};
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let best = CandidateScore {
            label: "+x+y+z".to_string(),
            matrix: id,
            is_identity: true,
            coherence: 0.9,
            n_samples: 10,
        };
        FlipResult {
            working_b: 1000.0,
            n_wm_voxels: 10,
            mask_mean_fa: fa,
            ranking: vec![best.clone()],
            best: best.clone(),
            runner_up: best,
            identity_coherence: 0.9,
            margin: 0.1,
            relative_margin: 0.1,
            decision: Decision::Pass,
            recommended_transform: None,
            recommended_label: None,
        }
    }

    #[test]
    fn low_mask_fa_adds_advisory_note() {
        let low = note_low_mask_fa(
            Report::new(SchemeTable::from_table(&scheme())).with_flip(flip_with_mask_fa(0.25)),
        );
        assert!(low.notes.iter().any(|n| n.contains("WM mask")));

        let high = note_low_mask_fa(
            Report::new(SchemeTable::from_table(&scheme())).with_flip(flip_with_mask_fa(0.40)),
        );
        assert!(high.notes.is_empty());
    }

    #[test]
    fn inspect_reports_scheme_without_image() {
        let report = inspect(&scheme(), Vec::new(), AuditOptions::default()).unwrap();
        assert!(report.shells.is_some());
        assert!(report.angular.is_some());
        assert!(report.flip.is_none());
        assert_eq!(report.status, Status::Pass);
        assert_eq!(report.norm_stats.unwrap().non_unit_count, 0);
    }

    fn amplitude_scheme() -> GradientTable {
        // Constant nominal bval; norms encode the true weighting (|g| = √(b/3000)).
        let mut directions = vec![[0.0, 0.0, 0.0]];
        let mut bvals = vec![0.0];
        for (axis, ratio) in [
            (0, 1.0_f64),   // b 3000
            (1, 2.0 / 3.0), // b 2000
            (2, 1.0 / 3.0), // b 1000
            (0, 2.0 / 3.0),
            (1, 1.0 / 3.0),
            (2, 1.0),
        ] {
            let mut d = [0.0; 3];
            d[axis] = ratio.sqrt();
            directions.push(d);
            bvals.push(3000.0);
        }
        GradientTable::new(directions, bvals).unwrap()
    }

    #[test]
    fn inspect_warns_on_amplitude_encoded() {
        let report = inspect(&amplitude_scheme(), Vec::new(), AuditOptions::default()).unwrap();
        assert_eq!(report.status, Status::Warn);
        assert!(report.norm_stats.unwrap().non_unit_fraction > 0.5);
        assert!(report.notes.iter().any(|n| n.contains("amplitude-encoded")));
    }

    #[test]
    fn inspect_strict_hard_errors_on_amplitude_encoded() {
        let opts = AuditOptions {
            strict: true,
            ..AuditOptions::default()
        };
        let err = inspect(&amplitude_scheme(), Vec::new(), opts).unwrap_err();
        assert!(matches!(err, crate::error::Error::AmplitudeEncoded(_)));
    }

    #[test]
    fn recompute_bval_recovers_and_reaudits_clean() {
        let dir = tempdir().unwrap();
        let bvec = dir.path().join("rec.bvec");
        let bval = dir.path().join("rec.bval");
        let spec = RecomputeSpec {
            bvec: bvec.clone(),
            bval: bval.clone(),
            mrtrix: None,
            provenance: None,
            dry_run: false,
            in_place: false,
        };
        let out = recompute_bval(
            &amplitude_scheme(),
            Vec::new(),
            ShellConfig::default(),
            &spec,
        )
        .unwrap();
        let after: Vec<f64> = out.summary.after.iter().map(|s| s.nominal_b).collect();
        assert_eq!(after, vec![0.0, 1000.0, 2000.0, 3000.0]);

        let report = inspect(&out.table, Vec::new(), AuditOptions::default()).unwrap();
        assert_eq!(report.status, Status::Pass);
        assert_eq!(report.norm_stats.unwrap().non_unit_count, 0);
    }

    #[test]
    fn audit_consistent_table_passes() {
        let table = scheme();
        let data = phantom(&table);
        let report = audit(&table, Some(&data), None, Vec::new(), options()).unwrap();
        assert_eq!(report.status, Status::Pass);
        assert!(report.b0_drift.is_some());
    }

    #[test]
    fn audit_rejects_volume_count_mismatch() {
        let table = scheme();
        let data = phantom(&table);
        let short = GradientTable::new(
            table.directions[..table.len() - 1].to_vec(),
            table.bvals[..table.len() - 1].to_vec(),
        )
        .unwrap();
        let err = audit(&short, Some(&data), None, Vec::new(), options()).unwrap_err();
        assert!(matches!(err, Error::VolumeCountMismatch { .. }));
    }

    #[test]
    fn audit_rejects_mismatched_mask() {
        let table = scheme();
        let data = phantom(&table);
        let mask = vec![true; 10];
        let err = audit(&table, Some(&data), Some(&mask), Vec::new(), options()).unwrap_err();
        assert!(matches!(err, Error::MaskShapeMismatch { .. }));
    }

    #[test]
    fn audit_without_image_skips_flip() {
        let report = audit(&scheme(), None, None, Vec::new(), options()).unwrap();
        assert!(report.flip.is_none());
        assert!(report.b0_drift.is_none());
    }

    #[test]
    fn repair_dry_run_flags_without_writing() {
        let (_truth, corrupted, data) = flip_case();
        let dir = tempdir().unwrap();
        let spec = RepairSpec {
            bvec: dir.path().join("out.bvec"),
            bval: dir.path().join("out.bval"),
            mrtrix: None,
            provenance: None,
            dry_run: true,
            in_place: false,
            force_repair: false,
            frame: None,
        };
        let out = repair(&corrupted, Some(&data), None, Vec::new(), options(), &spec).unwrap();
        assert_eq!(out.report.status, Status::Flag);
        assert!(out.report.repair.is_some());
        assert!(out.repaired.is_some());
        assert!(!spec.bvec.exists());
    }

    #[test]
    fn repair_writes_corrected_table() {
        let (truth, corrupted, data) = flip_case();
        let dir = tempdir().unwrap();
        let spec = RepairSpec {
            bvec: dir.path().join("out.bvec"),
            bval: dir.path().join("out.bval"),
            mrtrix: Some(dir.path().join("out.b")),
            provenance: Some(dir.path().join("prov.json")),
            dry_run: false,
            in_place: false,
            force_repair: false,
            frame: None,
        };
        let out = repair(&corrupted, Some(&data), None, Vec::new(), options(), &spec).unwrap();
        assert!(spec.bvec.exists() && spec.bval.exists());
        assert!(spec.mrtrix.unwrap().exists());
        assert!(spec.provenance.unwrap().exists());
        let info = out.report.repair.unwrap();
        assert_eq!(info.outputs.len(), 3);
        let repaired = out.repaired.unwrap();
        assert_eq!(repaired.directions, truth.directions);
    }

    fn transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut t = [[0.0; 3]; 3];
        for (i, row) in m.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                t[j][i] = v;
            }
        }
        t
    }

    fn same_convention(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> bool {
        let neg = (0..3).all(|i| (0..3).all(|j| a[i][j] == -b[i][j]));
        a == b || neg
    }

    // Frame handedness drives the correction: with the affine's voxel-frame map
    // applied, a correct table PASSes and a genuine flip recovers the right
    // convention on both radiological (F = I) and neurological (F = x-flip)
    // storage. The acquisition's true voxel-frame gradient is `F·g`; the stored
    // table is `C0·g`, so the best candidate must recover `C0⁻¹`.
    #[test]
    fn frame_correction_handles_both_handedness() {
        let truth = scheme();
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let xflip = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let swap_xy = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

        // (frame map F, corruption C0, expected identity best).
        let cases = [
            (id, id, true),        // radiological, correct
            (xflip, id, true),     // neurological, correct
            (id, swap_xy, false),  // radiological, corrupted
            (xflip, xflip, false), // neurological, corrupted
        ];
        for (f, c0, expect_identity) in cases {
            let data = phantom(&apply_transform(&truth, &f));
            let stored = apply_transform(&truth, &c0);
            let config = FlipConfig {
                coherence: CoherenceConfig { step: 2.0 },
                frame_map: f,
                ..FlipConfig::default()
            };
            let result = detect_flip(&data, &stored, None, config).unwrap();
            assert!(
                same_convention(&result.best.matrix, &transpose(&c0)),
                "best={} did not recover the convention",
                result.best.label
            );
            assert_eq!(
                result.best.is_identity, expect_identity,
                "best={}",
                result.best.label
            );
        }
    }

    fn affine(diag: [f64; 3]) -> VolumeInfo {
        VolumeInfo {
            shape: vec![2, 2, 2, 1],
            voxel_sizes: [diag[0].abs(), diag[1].abs(), diag[2].abs()],
            affine: [
                [diag[0], 0.0, 0.0, 0.0],
                [0.0, diag[1], 0.0, 0.0],
                [0.0, 0.0, diag[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            num_volumes: 1,
        }
    }

    // Cross-format emit re-expresses the corrected table; same-format is verbatim.
    #[test]
    fn emit_table_converts_between_frames() {
        let neuro = affine([2.0, 2.0, 2.0]); // det > 0: FSL bvec is x-flipped from world
        let table = scheme();
        let xflip = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let from_fsl = FrameMaps::resolve(GradientFrame::Fsl, &neuro);
        assert_eq!(
            emit_table(&table, Some(&from_fsl), FrameMaps::to_fsl).directions,
            table.directions
        );
        assert_eq!(
            emit_table(&table, Some(&from_fsl), FrameMaps::to_world).directions,
            apply_transform(&table, &xflip).directions
        );

        let from_world = FrameMaps::resolve(GradientFrame::World, &neuro);
        assert_eq!(
            emit_table(&table, Some(&from_world), FrameMaps::to_world).directions,
            table.directions
        );
        assert_eq!(
            emit_table(&table, Some(&from_world), FrameMaps::to_fsl).directions,
            apply_transform(&table, &xflip).directions
        );

        // No frame maps: verbatim (legacy same-format behaviour).
        assert_eq!(
            emit_table(&table, None, FrameMaps::to_world).directions,
            table.directions
        );
    }

    #[test]
    fn repair_notes_divergent_frames() {
        let (_truth, corrupted, data) = flip_case();
        let dir = tempdir().unwrap();
        let spec = RepairSpec {
            bvec: dir.path().join("o.bvec"),
            bval: dir.path().join("o.bval"),
            mrtrix: None,
            provenance: None,
            dry_run: true,
            in_place: false,
            force_repair: false,
            frame: Some(FrameMaps::resolve(
                GradientFrame::Fsl,
                &affine([-2.0, -2.0, 2.0]), // divergent: det > 0, two negated axes
            )),
        };
        let out = repair(&corrupted, Some(&data), None, Vec::new(), options(), &spec).unwrap();
        assert!(out.report.notes.iter().any(|n| n.contains("diverge")));
    }
}
