//! Python bindings for gradlint.
//!
//! Each entry point runs a core pipeline and returns the canonical `report.json`
//! as a string; the Python layer parses it into typed dataclasses.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use gradlint_core::io::provenance::InputFile;
use gradlint_core::io::{fsl, mrtrix};
use gradlint_core::pipeline::{self, AuditOptions, RecomputeSpec, RepairSpec};
use gradlint_core::{bids_batch, GlyphData, GradientTable, Report, ShellConfig, VolumeInfo};

#[pyfunction]
fn version() -> &'static str {
    gradlint_core::version()
}

#[pyfunction]
fn build_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "libdeflate") {
        features.push("libdeflate");
    }
    features
}

#[pyfunction]
#[pyo3(signature = (bvec=None, bval=None, grad=None, tolerance=0.05, b0_threshold=50.0, shell=None, strict=false, norm_tolerance=0.05))]
#[allow(clippy::too_many_arguments)]
fn inspect(
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    strict: bool,
    norm_tolerance: f64,
) -> PyResult<String> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let report = pipeline::inspect(
        &table,
        inputs,
        options(tolerance, b0_threshold, shell, strict, norm_tolerance),
    )
    .map_err(err)?;
    to_json(&report)
}

#[pyfunction]
#[pyo3(signature = (bvec=None, bval=None, grad=None, dwi=None, mask=None, tolerance=0.05, b0_threshold=50.0, shell=None, step=None, strict=false, norm_tolerance=0.05))]
#[allow(clippy::too_many_arguments)]
fn audit(
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    dwi: Option<String>,
    mask: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
    strict: bool,
    norm_tolerance: f64,
) -> PyResult<String> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let mut opts = options(tolerance, b0_threshold, shell, strict, norm_tolerance);
    let data = match dwi.as_deref() {
        Some(path) => {
            let (volume, info) = gradlint_core::read_volume_with_info(path).map_err(err)?;
            pipeline::apply_geometry(
                &mut opts,
                &info,
                gradlint_core::frame_for(grad.is_some()),
                step,
            );
            Some(volume)
        }
        None => None,
    };
    let mask = read_optional_mask(mask.as_deref())?;
    let report =
        pipeline::audit(&table, data.as_ref(), mask.as_deref(), inputs, opts).map_err(err)?;
    to_json(&report)
}

#[pyfunction]
#[pyo3(signature = (dwi, bvec=None, bval=None, grad=None, mask=None, tolerance=0.05, b0_threshold=50.0, shell=None, step=None, strict=false, norm_tolerance=0.05))]
#[allow(clippy::too_many_arguments)]
fn audit_with_glyphs(
    py: Python<'_>,
    dwi: String,
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    mask: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
    strict: bool,
    norm_tolerance: f64,
) -> PyResult<(String, Py<PyDict>)> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let (data, info) = gradlint_core::read_volume_with_info(&dwi).map_err(err)?;
    let mut opts = options(tolerance, b0_threshold, shell, strict, norm_tolerance);
    pipeline::apply_geometry(
        &mut opts,
        &info,
        gradlint_core::frame_for(grad.is_some()),
        step,
    );
    let mask = read_optional_mask(mask.as_deref())?;
    let (report, glyphs) =
        pipeline::audit_with_glyphs(&table, &data, mask.as_deref(), inputs, opts).map_err(err)?;
    Ok((to_json(&report)?, glyph_payload(py, &glyphs, &info)?))
}

/// One instrumented audit: returns the report JSON plus a per-stage wall-clock
/// split (`decompress`/`convert`/`fit`/`coherence`/`other`/`total`, seconds).
/// The report is byte-identical to `audit`; only the timing instrumentation is
/// added, so this drives the Python CLI's `--profile`.
#[pyfunction]
#[pyo3(signature = (dwi, bvec=None, bval=None, grad=None, mask=None, tolerance=0.05, b0_threshold=50.0, shell=None, step=None, strict=false, norm_tolerance=0.05))]
#[allow(clippy::too_many_arguments)]
fn audit_profiled(
    dwi: String,
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    mask: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
    strict: bool,
    norm_tolerance: f64,
) -> PyResult<(String, HashMap<String, f64>)> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let mut opts = options(tolerance, b0_threshold, shell, strict, norm_tolerance);
    let total = Instant::now();
    let (data, info, read) = gradlint_core::read_volume_with_info_timed(&dwi).map_err(err)?;
    pipeline::apply_geometry(
        &mut opts,
        &info,
        gradlint_core::frame_for(grad.is_some()),
        step,
    );
    let mask = read_optional_mask(mask.as_deref())?;
    let (report, detect) =
        pipeline::audit_timed(&table, &data, mask.as_deref(), inputs, opts).map_err(err)?;
    let total = total.elapsed().as_secs_f64();
    let decompress = read.decompress.as_secs_f64();
    let convert = read.convert.as_secs_f64();
    let fit = detect.fit.as_secs_f64();
    let coherence = detect.coherence.as_secs_f64();
    let other = (total - decompress - convert - fit - coherence).max(0.0);
    let profile = HashMap::from([
        ("decompress".to_string(), decompress),
        ("convert".to_string(), convert),
        ("fit".to_string(), fit),
        ("coherence".to_string(), coherence),
        ("other".to_string(), other),
        ("total".to_string(), total),
    ]);
    Ok((to_json(&report)?, profile))
}

#[pyfunction]
#[pyo3(signature = (dwi, bvec=None, bval=None, grad=None, mask=None, tolerance=0.05, b0_threshold=50.0, shell=None, step=None, strict=false, norm_tolerance=0.05))]
#[allow(clippy::too_many_arguments)]
fn audit_profiled_with_glyphs(
    py: Python<'_>,
    dwi: String,
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    mask: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
    strict: bool,
    norm_tolerance: f64,
) -> PyResult<(String, HashMap<String, f64>, Py<PyDict>)> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let mut opts = options(tolerance, b0_threshold, shell, strict, norm_tolerance);
    let total = Instant::now();
    let (data, info, read) = gradlint_core::read_volume_with_info_timed(&dwi).map_err(err)?;
    pipeline::apply_geometry(
        &mut opts,
        &info,
        gradlint_core::frame_for(grad.is_some()),
        step,
    );
    let mask = read_optional_mask(mask.as_deref())?;
    let (report, detect, glyphs) =
        pipeline::audit_timed_with_glyphs(&table, &data, mask.as_deref(), inputs, opts)
            .map_err(err)?;
    let total = total.elapsed().as_secs_f64();
    let decompress = read.decompress.as_secs_f64();
    let convert = read.convert.as_secs_f64();
    let fit = detect.fit.as_secs_f64();
    let coherence = detect.coherence.as_secs_f64();
    let other = (total - decompress - convert - fit - coherence).max(0.0);
    let profile = HashMap::from([
        ("decompress".to_string(), decompress),
        ("convert".to_string(), convert),
        ("fit".to_string(), fit),
        ("coherence".to_string(), coherence),
        ("other".to_string(), other),
        ("total".to_string(), total),
    ]);
    Ok((
        to_json(&report)?,
        profile,
        glyph_payload(py, &glyphs, &info)?,
    ))
}

/// Audit every DWI under a BIDS root (shared `bids_batch` core path). Writes the
/// per-run and dataset reports under `<root>/derivatives/gradlint/` and returns a
/// small JSON summary (`status`, `exit_code`, `summary_path`, `results`) for the
/// caller to print. The exit code blocks only on WARN.
#[pyfunction]
#[pyo3(signature = (root, tolerance=0.05, b0_threshold=50.0, shell=None, step=None, strict=false, norm_tolerance=0.05))]
#[allow(clippy::too_many_arguments)]
fn audit_bids(
    root: String,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
    strict: bool,
    norm_tolerance: f64,
) -> PyResult<String> {
    let opts = options(tolerance, b0_threshold, shell, strict, norm_tolerance);
    let outcome = bids_batch::run(Path::new(&root), opts, step).map_err(err)?;
    let results: Vec<_> = outcome
        .items
        .iter()
        .map(|item| {
            serde_json::json!({
                "name": item.name,
                "dwi": item.dwi.display().to_string(),
                "status": item.status.label(),
            })
        })
        .collect();
    let summary = serde_json::json!({
        "status": outcome.worst.label(),
        "exit_code": outcome.worst.exit_code(),
        "summary_path": outcome.summary_path.display().to_string(),
        "results": results,
    });
    serde_json::to_string(&summary).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (dwi, bvec=None, bval=None, grad=None, mask=None, tolerance=0.05, b0_threshold=50.0, shell=None, step=None))]
#[allow(clippy::too_many_arguments)]
fn detect_flip(
    dwi: String,
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    mask: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
) -> PyResult<String> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let (data, info) = gradlint_core::read_volume_with_info(&dwi).map_err(err)?;
    let mut opts = options(tolerance, b0_threshold, shell, false, 0.05);
    pipeline::apply_geometry(
        &mut opts,
        &info,
        gradlint_core::frame_for(grad.is_some()),
        step,
    );
    let mask = read_optional_mask(mask.as_deref())?;
    let report = pipeline::detect(&table, &data, mask.as_deref(), inputs, opts).map_err(err)?;
    to_json(&report)
}

#[pyfunction]
#[pyo3(signature = (dwi, bvec=None, bval=None, grad=None, mask=None, tolerance=0.05, b0_threshold=50.0, shell=None, step=None))]
#[allow(clippy::too_many_arguments)]
fn detect_flip_with_glyphs(
    py: Python<'_>,
    dwi: String,
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    mask: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
) -> PyResult<(String, Py<PyDict>)> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let (data, info) = gradlint_core::read_volume_with_info(&dwi).map_err(err)?;
    let mut opts = options(tolerance, b0_threshold, shell, false, 0.05);
    pipeline::apply_geometry(
        &mut opts,
        &info,
        gradlint_core::frame_for(grad.is_some()),
        step,
    );
    let mask = read_optional_mask(mask.as_deref())?;
    let (report, glyphs) =
        pipeline::detect_with_glyphs(&table, &data, mask.as_deref(), inputs, opts).map_err(err)?;
    Ok((to_json(&report)?, glyph_payload(py, &glyphs, &info)?))
}

#[pyfunction]
#[pyo3(signature = (dwi, out_bvec, out_bval, bvec=None, bval=None, grad=None, mask=None, out_grad=None, provenance=None, tolerance=0.05, b0_threshold=50.0, shell=None, step=None, dry_run=false, in_place=false, strict=false, force_repair=false, norm_tolerance=0.05))]
#[allow(clippy::too_many_arguments)]
fn repair(
    dwi: String,
    out_bvec: String,
    out_bval: String,
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    mask: Option<String>,
    out_grad: Option<String>,
    provenance: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
    dry_run: bool,
    in_place: bool,
    strict: bool,
    force_repair: bool,
    norm_tolerance: f64,
) -> PyResult<String> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let (data, info) = gradlint_core::read_volume_with_info(&dwi).map_err(err)?;
    let mut opts = options(tolerance, b0_threshold, shell, strict, norm_tolerance);
    let frame = gradlint_core::frame_for(grad.is_some());
    pipeline::apply_geometry(&mut opts, &info, frame, step);
    let mask = read_optional_mask(mask.as_deref())?;
    let spec = RepairSpec {
        bvec: out_bvec.into(),
        bval: out_bval.into(),
        mrtrix: out_grad.map(Into::into),
        provenance: provenance.map(Into::into),
        dry_run,
        in_place,
        force_repair,
        frame: Some(gradlint_core::FrameMaps::resolve(frame, &info)),
    };
    let outcome =
        pipeline::repair(&table, Some(&data), mask.as_deref(), inputs, opts, &spec).map_err(err)?;
    to_json(&outcome.report)
}

#[pyfunction]
#[pyo3(signature = (dwi, out_bvec, out_bval, bvec=None, bval=None, grad=None, mask=None, out_grad=None, provenance=None, tolerance=0.05, b0_threshold=50.0, shell=None, step=None, dry_run=false, in_place=false, strict=false, force_repair=false, norm_tolerance=0.05))]
#[allow(clippy::too_many_arguments)]
fn repair_with_glyphs(
    py: Python<'_>,
    dwi: String,
    out_bvec: String,
    out_bval: String,
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    mask: Option<String>,
    out_grad: Option<String>,
    provenance: Option<String>,
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    step: Option<f64>,
    dry_run: bool,
    in_place: bool,
    strict: bool,
    force_repair: bool,
    norm_tolerance: f64,
) -> PyResult<(String, Py<PyDict>)> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let (data, info) = gradlint_core::read_volume_with_info(&dwi).map_err(err)?;
    let mut opts = options(tolerance, b0_threshold, shell, strict, norm_tolerance);
    let frame = gradlint_core::frame_for(grad.is_some());
    pipeline::apply_geometry(&mut opts, &info, frame, step);
    let mask = read_optional_mask(mask.as_deref())?;
    let spec = RepairSpec {
        bvec: out_bvec.into(),
        bval: out_bval.into(),
        mrtrix: out_grad.map(Into::into),
        provenance: provenance.map(Into::into),
        dry_run,
        in_place,
        force_repair,
        frame: Some(gradlint_core::FrameMaps::resolve(frame, &info)),
    };
    let (outcome, glyphs) =
        pipeline::repair_with_glyphs(&table, &data, mask.as_deref(), inputs, opts, &spec)
            .map_err(err)?;
    Ok((
        to_json(&outcome.report)?,
        glyph_payload(py, &glyphs, &info)?,
    ))
}

/// Opt-in b-value recovery from amplitude-encoded bvecs. Never reached from
/// `repair`; returns a JSON before/after summary.
#[pyfunction]
#[pyo3(signature = (out_bvec, out_bval, bvec=None, bval=None, grad=None, out_grad=None, provenance=None, b0_threshold=50.0, dry_run=false, in_place=false))]
#[allow(clippy::too_many_arguments)]
fn recompute_bval(
    out_bvec: String,
    out_bval: String,
    bvec: Option<String>,
    bval: Option<String>,
    grad: Option<String>,
    out_grad: Option<String>,
    provenance: Option<String>,
    b0_threshold: f64,
    dry_run: bool,
    in_place: bool,
) -> PyResult<String> {
    let (table, inputs) = load_table(bvec.as_deref(), bval.as_deref(), grad.as_deref())?;
    let shell = ShellConfig {
        b0_threshold,
        tolerance: 0.05,
    };
    let spec = RecomputeSpec {
        bvec: out_bvec.into(),
        bval: out_bval.into(),
        mrtrix: out_grad.map(Into::into),
        provenance: provenance.map(Into::into),
        dry_run,
        in_place,
    };
    let out = pipeline::recompute_bval(&table, inputs, shell, &spec).map_err(err)?;
    serde_json::to_string(&out.summary).map_err(err)
}

fn load_table(
    bvec: Option<&str>,
    bval: Option<&str>,
    grad: Option<&str>,
) -> PyResult<(GradientTable, Vec<InputFile>)> {
    if let Some(grad) = grad {
        Ok((
            mrtrix::read(grad).map_err(err)?,
            vec![InputFile::of(grad).map_err(err)?],
        ))
    } else if let (Some(bvec), Some(bval)) = (bvec, bval) {
        Ok((
            fsl::read(bvec, bval).map_err(err)?,
            vec![
                InputFile::of(bvec).map_err(err)?,
                InputFile::of(bval).map_err(err)?,
            ],
        ))
    } else {
        Err(PyValueError::new_err("provide bvec and bval, or grad"))
    }
}

fn options(
    tolerance: f64,
    b0_threshold: f64,
    shell: Option<f64>,
    strict: bool,
    norm_tolerance: f64,
) -> AuditOptions {
    let mut options = AuditOptions::default();
    options.shell.tolerance = tolerance;
    options.shell.b0_threshold = b0_threshold;
    options.flip.shell = options.shell;
    options.working_shell = shell;
    options.strict = strict;
    options.norm_tolerance = norm_tolerance;
    options
}

fn read_optional_mask(path: Option<&str>) -> PyResult<Option<Vec<bool>>> {
    match path {
        Some(p) => Ok(Some(gradlint_core::read_mask(p).map_err(err)?)),
        None => Ok(None),
    }
}

fn glyph_payload(py: Python<'_>, glyphs: &GlyphData, info: &VolumeInfo) -> PyResult<Py<PyDict>> {
    let nvox = glyphs.field.v1.len();
    let mut v1 = Vec::with_capacity(nvox * 3 * size_of::<f32>());
    for vector in &glyphs.field.v1 {
        for &value in vector {
            v1.extend_from_slice(&(value as f32).to_ne_bytes());
        }
    }
    let fa = f32_bytes(glyphs.field.fa.iter().copied());
    let s0 = f32_bytes(glyphs.field.s0.iter().copied());
    let mask: Vec<u8> = glyphs.mask.iter().map(|&keep| u8::from(keep)).collect();
    let payload = PyDict::new(py);
    payload.set_item("shape", glyphs.field.shape)?;
    payload.set_item("affine", info.affine)?;
    payload.set_item("voxel_sizes", info.voxel_sizes)?;
    payload.set_item("frame_map", glyphs.frame_map)?;
    payload.set_item("v1", PyBytes::new(py, &v1))?;
    payload.set_item("fa", PyBytes::new(py, &fa))?;
    payload.set_item("s0", PyBytes::new(py, &s0))?;
    payload.set_item("mask", PyBytes::new(py, &mask))?;
    Ok(payload.unbind())
}

fn f32_bytes(values: impl Iterator<Item = f64>) -> Vec<u8> {
    let (lower, _) = values.size_hint();
    let mut bytes = Vec::with_capacity(lower * size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&(value as f32).to_ne_bytes());
    }
    bytes
}

fn to_json(report: &Report) -> PyResult<String> {
    serde_json::to_string(report).map_err(err)
}

fn err<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn _gradlint(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(build_features, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_function(wrap_pyfunction!(audit, m)?)?;
    m.add_function(wrap_pyfunction!(audit_with_glyphs, m)?)?;
    m.add_function(wrap_pyfunction!(audit_profiled, m)?)?;
    m.add_function(wrap_pyfunction!(audit_profiled_with_glyphs, m)?)?;
    m.add_function(wrap_pyfunction!(audit_bids, m)?)?;
    m.add_function(wrap_pyfunction!(detect_flip, m)?)?;
    m.add_function(wrap_pyfunction!(detect_flip_with_glyphs, m)?)?;
    m.add_function(wrap_pyfunction!(repair, m)?)?;
    m.add_function(wrap_pyfunction!(repair_with_glyphs, m)?)?;
    m.add_function(wrap_pyfunction!(recompute_bval, m)?)?;
    Ok(())
}
