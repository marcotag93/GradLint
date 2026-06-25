//! `gradlint` command-line interface: scheme QC, flip detection, and repair.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};

use gradlint_core::flip::{CandidateScore, Decision, FlipResult};
use gradlint_core::io::provenance::InputFile;
use gradlint_core::io::{fsl, mrtrix};
use gradlint_core::pipeline::{self, AuditOptions, RecomputeSpec, RepairSpec};
use gradlint_core::report::{self, BvalRecoverySummary, RepairInfo, Report, Status};
use gradlint_core::{DetectTimings, GradientTable, ReadTimings, ShellConfig, VolumeInfo};

mod bids;
mod style;

/// Version string shown by `--version`
#[cfg(feature = "libdeflate")]
const LONG_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (libdeflate)");
#[cfg(not(feature = "libdeflate"))]
const LONG_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "gradlint",
    version = LONG_VERSION,
    about = "Gradient-scheme QC and b-vector repair for diffusion MRI"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect the gradient scheme (shells + angular QC); no image required.
    Inspect(InspectArgs),
    /// Audit a DWI: scheme QC plus b-vector flip detection.
    Audit(AuditArgs),
    /// Audit and write a corrected gradient table when a flip is flagged.
    Repair(RepairArgs),
    /// Recover amplitude-encoded b-values into a corrected bval + unit bvec.
    RecomputeBval(RecomputeArgs),
}

#[derive(Args)]
struct GradientInput {
    /// FSL bvec file (3xN or Nx3).
    #[arg(long, value_name = "FILE")]
    bvec: Option<PathBuf>,
    /// FSL bval file.
    #[arg(long, value_name = "FILE")]
    bval: Option<PathBuf>,
    /// MRtrix .b gradient table (world coordinates), instead of bvec/bval.
    #[arg(long, value_name = "FILE")]
    grad: Option<PathBuf>,
}

#[derive(Args)]
struct SchemeOpts {
    /// Relative b-value tolerance for shell clustering.
    #[arg(long, default_value_t = 0.05)]
    tolerance: f64,
    /// b-values at or below this are treated as b0.
    #[arg(long = "b0-threshold", default_value_t = 50.0)]
    b0_threshold: f64,
    /// Working-shell b-value for flip detection (auto-selected if omitted).
    #[arg(long, value_name = "B")]
    shell: Option<f64>,
    /// Coherence step length in voxels (default scales ~4 mm with voxel size).
    #[arg(long, value_name = "VOXELS")]
    step: Option<f64>,
    /// Promote severe scheme-quality issues (conditioning, direction count, b0
    /// drift) to WARN; scheme notes are emitted regardless.
    #[arg(long)]
    strict: bool,
    /// Unit-norm tolerance for amplitude-encoded bvec detection.
    #[arg(long = "norm-tolerance", default_value_t = 0.05)]
    norm_tolerance: f64,
}

#[derive(Args)]
struct InspectArgs {
    #[command(flatten)]
    input: GradientInput,
    #[command(flatten)]
    opts: SchemeOpts,
    /// Write report.json
    #[arg(long, value_name = "FILE")]
    report: Option<PathBuf>,
}

#[derive(Args)]
struct AuditArgs {
    #[command(flatten)]
    input: GradientInput,
    /// DWI NIfTI; required for flip detection (omit for scheme-only QC).
    #[arg(long, value_name = "FILE")]
    dwi: Option<PathBuf>,
    /// White-matter / brain mask NIfTI.
    #[arg(long, value_name = "FILE")]
    mask: Option<PathBuf>,
    #[command(flatten)]
    opts: SchemeOpts,
    /// Write report.json here.
    #[arg(long, value_name = "FILE")]
    report: Option<PathBuf>,
    /// BIDS dataset root: discover and audit every DWI, writing derivatives.
    #[arg(long, value_name = "DIR")]
    bids: Option<PathBuf>,
    /// Print a stage-timing breakdown (decompress/convert/fit/coherence).
    #[arg(long)]
    profile: bool,
}

#[derive(Args)]
struct RepairArgs {
    #[command(flatten)]
    input: GradientInput,
    /// DWI NIfTI (required).
    #[arg(long, value_name = "FILE")]
    dwi: PathBuf,
    /// White-matter / brain mask NIfTI.
    #[arg(long, value_name = "FILE")]
    mask: Option<PathBuf>,
    #[command(flatten)]
    opts: SchemeOpts,
    /// Corrected bvec output (default: alongside the input with a .repaired tag).
    #[arg(long = "out-bvec", value_name = "FILE")]
    out_bvec: Option<PathBuf>,
    /// Corrected bval output.
    #[arg(long = "out-bval", value_name = "FILE")]
    out_bval: Option<PathBuf>,
    /// Also write a corrected MRtrix .b table here.
    #[arg(long = "out-grad", value_name = "FILE")]
    out_grad: Option<PathBuf>,
    /// Write the canonical report.json here.
    #[arg(long, value_name = "FILE")]
    report: Option<PathBuf>,
    /// Write a provenance log here.
    #[arg(long, value_name = "FILE")]
    provenance: Option<PathBuf>,
    /// Compute the repair but write nothing.
    #[arg(long = "dry-run")]
    dry_run: bool,
    /// Overwrite the input files (a .bak backup is kept).
    #[arg(long)]
    force: bool,
    /// Apply the correction on a WARN (thin-margin) decision too, overriding the
    /// conservative withhld. The verdict stays WARN; no effect on PASS.
    #[arg(long = "force-repair")]
    force_repair: bool,
}

#[derive(Args)]
struct RecomputeArgs {
    #[command(flatten)]
    input: GradientInput,
    /// b-values at or below this are treated as b0.
    #[arg(long = "b0-threshold", default_value_t = 50.0)]
    b0_threshold: f64,
    /// Corrected bvec output (default: alongside the input with a .recovered tag).
    #[arg(long = "out-bvec", value_name = "FILE")]
    out_bvec: Option<PathBuf>,
    /// Corrected bval output.
    #[arg(long = "out-bval", value_name = "FILE")]
    out_bval: Option<PathBuf>,
    /// Also write a corrected MRtrix .b table here.
    #[arg(long = "out-grad", value_name = "FILE")]
    out_grad: Option<PathBuf>,
    /// Write a provenance log here.
    #[arg(long, value_name = "FILE")]
    provenance: Option<PathBuf>,
    /// Compute the recovery but write nothing.
    #[arg(long = "dry-run")]
    dry_run: bool,
    /// Overwrite the input files (a .bak backup is kept).
    #[arg(long)]
    force: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<i32, String> {
    match Cli::parse().command {
        Command::Inspect(args) => run_inspect(args),
        Command::Audit(args) => run_audit(args),
        Command::Repair(args) => run_repair(args),
        Command::RecomputeBval(args) => run_recompute(args),
    }
}

fn run_inspect(args: InspectArgs) -> Result<i32, String> {
    let (table, inputs) = load_table(&args.input)?;
    let report =
        pipeline::inspect(&table, inputs, options(&args.opts)).map_err(|e| e.to_string())?;
    finish(&report, args.report.as_deref())
}

fn run_audit(args: AuditArgs) -> Result<i32, String> {
    if let Some(root) = &args.bids {
        if args.profile {
            return Err("--profile is not supported with --bids".to_string());
        }
        return bids::run(root, options(&args.opts), args.opts.step);
    }
    let (table, inputs) = load_table(&args.input)?;
    eprintln!("gradient table: {} volumes", table.len());
    let mut opts = options(&args.opts);
    if args.profile {
        let dwi = args.dwi.as_deref().ok_or("--profile requires --dwi")?;
        return run_audit_profiled(dwi, &table, &args, inputs, opts);
    }
    let data = match args.dwi.as_deref() {
        Some(path) => {
            let (volume, info) = read_dwi_with_progress(path)?;
            pipeline::apply_geometry(
                &mut opts,
                &info,
                gradlint_core::frame_for(args.input.grad.is_some()),
                args.opts.step,
            );
            Some(volume)
        }
        None => None,
    };
    let mask = read_optional_mask(args.mask.as_deref())?;
    let report = match data {
        Some(volume) => {
            announce_mask(args.mask.as_deref());
            let hb = Heartbeat::start("detecting flip: one DTI fit + ranking all 48 conventions");
            let report = pipeline::audit(&table, Some(&volume), mask.as_deref(), inputs, opts)
                .map_err(|e| e.to_string())?;
            hb.done();
            report
        }
        None => pipeline::audit(&table, None, mask.as_deref(), inputs, opts)
            .map_err(|e| e.to_string())?,
    };
    finish(&report, args.report.as_deref())
}

/// One instrumented audit: read with a decode/convert split, then audit with a
/// fit/coherence split, and print the per-stage breakdown.
fn run_audit_profiled(
    dwi: &Path,
    table: &GradientTable,
    args: &AuditArgs,
    inputs: Vec<InputFile>,
    mut opts: AuditOptions,
) -> Result<i32, String> {
    let total = Instant::now();
    eprintln!("DWI: {}", dwi.display());
    let hb = Heartbeat::start("reading DWI volume (instrumented read)");
    let (data, info, read) =
        gradlint_core::read_volume_with_info_timed(dwi).map_err(|e| e.to_string())?;
    hb.done();
    eprintln!("  {}", describe_volume(&info));
    pipeline::apply_geometry(
        &mut opts,
        &info,
        gradlint_core::frame_for(args.input.grad.is_some()),
        args.opts.step,
    );
    announce_mask(args.mask.as_deref());
    let mask = read_optional_mask(args.mask.as_deref())?;
    let hb = Heartbeat::start("detecting flip: one DTI fit + ranking all 48 conventions");
    let (report, detect) = pipeline::audit_timed(table, &data, mask.as_deref(), inputs, opts)
        .map_err(|e| e.to_string())?;
    hb.done();
    let total = total.elapsed();

    print_summary(&report);
    print_profile(&read, &detect, total);
    if let Some(path) = args.report.as_deref() {
        report::write(path, &report).map_err(|e| e.to_string())?;
        println!("report: {}", path.display());
    }
    Ok(report.exit_code())
}

fn run_repair(args: RepairArgs) -> Result<i32, String> {
    let (table, inputs) = load_table(&args.input)?;
    eprintln!("gradient table: {} volumes", table.len());
    let (data, info) = read_dwi_with_progress(&args.dwi)?;
    let mut opts = options(&args.opts);
    let frame = gradlint_core::frame_for(args.input.grad.is_some());
    pipeline::apply_geometry(&mut opts, &info, frame, args.opts.step);
    announce_mask(args.mask.as_deref());
    let mask = read_optional_mask(args.mask.as_deref())?;
    let spec = repair_spec(&args, gradlint_core::FrameMaps::resolve(frame, &info))?;
    let hb = Heartbeat::start("detecting flip + applying repair");
    let outcome = pipeline::repair(&table, Some(&data), mask.as_deref(), inputs, opts, &spec)
        .map_err(|e| e.to_string())?;
    hb.done();

    if args.force_repair && outcome.report.repair.is_none() {
        eprintln!("note: --force-repair had no effect (nothing to repair).");
    }
    for backup in &outcome.backups {
        println!("backed up {} ", backup.display());
    }
    finish(&outcome.report, args.report.as_deref())
}

fn run_recompute(args: RecomputeArgs) -> Result<i32, String> {
    let (table, inputs) = load_table(&args.input)?;
    let shell = ShellConfig {
        b0_threshold: args.b0_threshold,
        tolerance: 0.05,
    };
    let spec = recompute_spec(&args)?;
    let out = pipeline::recompute_bval(&table, inputs, shell, &spec).map_err(|e| e.to_string())?;
    for backup in &out.backups {
        println!("backed up {}", backup.display());
    }
    print_recovery(&out.summary, spec.dry_run);
    Ok(0)
}

fn recompute_spec(args: &RecomputeArgs) -> Result<RecomputeSpec, String> {
    let (bvec, bval) =
        if args.force {
            let bvec = args
                .input
                .bvec
                .clone()
                .ok_or("--force requires FSL --bvec/--bval inputs")?;
            let bval = args
                .input
                .bval
                .clone()
                .ok_or("--force requires FSL --bvec/--bval inputs")?;
            (bvec, bval)
        } else {
            match (&args.out_bvec, &args.out_bval) {
                (Some(bvec), Some(bval)) => (bvec.clone(), bval.clone()),
                _ => {
                    let bvec =
                        args.input.bvec.as_deref().ok_or(
                            "specify --out-bvec/--out-bval (or FSL inputs to derive them)",
                        )?;
                    let bval =
                        args.input.bval.as_deref().ok_or(
                            "specify --out-bvec/--out-bval (or FSL inputs to derive them)",
                        )?;
                    (
                        tagged_sibling(bvec, "recovered"),
                        tagged_sibling(bval, "recovered"),
                    )
                }
            }
        };
    Ok(RecomputeSpec {
        bvec,
        bval,
        mrtrix: args.out_grad.clone(),
        provenance: args.provenance.clone(),
        dry_run: args.dry_run,
        in_place: args.force,
    })
}

fn print_recovery(summary: &BvalRecoverySummary, dry_run: bool) {
    let tag = if dry_run {
        " (dry-run, nothing written)"
    } else {
        ""
    };
    println!(
        "recompute-bval{tag}: b_nominal={:.0}, max |g|={:.4}",
        summary.b_nominal, summary.max_norm
    );
    let fmt = |shells: &[gradlint_core::report::ShellCount]| {
        shells
            .iter()
            .map(|s| format!("b{:.0}×{}", s.nominal_b, s.count))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("  shells before: {}", fmt(&summary.before));
    println!("  shells after:  {}", fmt(&summary.after));
    println!(
        "  recovered non-b0 b-values: {:.0}–{:.0}",
        summary.b_min, summary.b_max
    );
    for path in &summary.outputs {
        println!("  wrote: {path}");
    }
}

fn finish(report: &Report, report_path: Option<&Path>) -> Result<i32, String> {
    print_summary(report);
    if let Some(path) = report_path {
        report::write(path, report).map_err(|e| e.to_string())?;
        println!("report: {}", path.display());
    }
    Ok(report.exit_code())
}

fn options(opts: &SchemeOpts) -> AuditOptions {
    let mut audit = AuditOptions::default();
    audit.shell.tolerance = opts.tolerance;
    audit.shell.b0_threshold = opts.b0_threshold;
    audit.flip.shell = audit.shell;
    audit.working_shell = opts.shell;
    audit.strict = opts.strict;
    audit.norm_tolerance = opts.norm_tolerance;
    audit
}

fn load_table(input: &GradientInput) -> Result<(GradientTable, Vec<InputFile>), String> {
    if let Some(grad) = &input.grad {
        let table = mrtrix::read(grad).map_err(|e| e.to_string())?;
        Ok((table, vec![input_file(grad)?]))
    } else if let (Some(bvec), Some(bval)) = (&input.bvec, &input.bval) {
        let table = fsl::read(bvec, bval).map_err(|e| e.to_string())?;
        Ok((table, vec![input_file(bvec)?, input_file(bval)?]))
    } else {
        Err("provide --bvec and --bval, or --grad".to_string())
    }
}

fn repair_spec(args: &RepairArgs, frame: gradlint_core::FrameMaps) -> Result<RepairSpec, String> {
    let (bvec, bval) =
        if args.force {
            let bvec = args
                .input
                .bvec
                .clone()
                .ok_or("--force requires FSL --bvec/--bval inputs")?;
            let bval = args
                .input
                .bval
                .clone()
                .ok_or("--force requires FSL --bvec/--bval inputs")?;
            (bvec, bval)
        } else {
            match (&args.out_bvec, &args.out_bval) {
                (Some(bvec), Some(bval)) => (bvec.clone(), bval.clone()),
                _ => {
                    let bvec =
                        args.input.bvec.as_deref().ok_or(
                            "specify --out-bvec/--out-bval (or FSL inputs to derive them)",
                        )?;
                    let bval =
                        args.input.bval.as_deref().ok_or(
                            "specify --out-bvec/--out-bval (or FSL inputs to derive them)",
                        )?;
                    (repaired_sibling(bvec), repaired_sibling(bval))
                }
            }
        };
    Ok(RepairSpec {
        bvec,
        bval,
        mrtrix: args.out_grad.clone(),
        provenance: args.provenance.clone(),
        dry_run: args.dry_run,
        in_place: args.force,
        force_repair: args.force_repair,
        frame: Some(frame),
    })
}

/// Stderr "still working" heartbeat for a long stage, so the terminal is never
/// silent.
struct Heartbeat {
    start: Instant,
    done: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Heartbeat {
    fn start(label: &str) -> Self {
        eprintln!("{label} ...");
        let start = Instant::now();
        let done = Arc::new(AtomicBool::new(false));
        let color = style::stderr_color();
        let handle = std::io::stderr().is_terminal().then(|| {
            let flag = Arc::clone(&done);
            thread::spawn(move || {
                let frames = ['|', '/', '-', '\\'];
                let mut i = 0usize;
                while !flag.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(250));
                    eprint!(
                        "\r  {} {:.0}s elapsed ",
                        style::paint(&frames[i % 4].to_string(), style::BEST, color),
                        start.elapsed().as_secs_f64()
                    );
                    let _ = std::io::stderr().flush();
                    i += 1;
                }
            })
        });
        Self {
            start,
            done,
            handle,
        }
    }

    fn done(mut self) {
        self.stop();
        let msg = format!("  done in {:.1}s", self.start.elapsed().as_secs_f64());
        eprintln!("{}", style::paint(&msg, style::DIM, style::stderr_color()));
    }

    fn stop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
        }
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        self.stop();
    }
}

/// One-line geometry summary of a loaded DWI for the progress log.
fn describe_volume(info: &VolumeInfo) -> String {
    let s = &info.shape;
    let v = info.voxel_sizes;
    format!(
        "{}×{}×{} grid, {} volumes, {:.2}×{:.2}×{:.2} mm, {:?}",
        s[0],
        s[1],
        s[2],
        info.num_volumes,
        v[0],
        v[1],
        v[2],
        info.handedness()
    )
}

/// Read a DWI with a live heartbeat, then log its geometry.
fn read_dwi_with_progress(path: &Path) -> Result<(ndarray::ArrayD<f32>, VolumeInfo), String> {
    eprintln!("DWI: {}", path.display());
    let hb = Heartbeat::start("reading DWI volume (gzip decompress is the bottleneck)");
    let (volume, info) = read_volume_with_info(path)?;
    hb.done();
    eprintln!("  {}", describe_volume(&info));
    Ok((volume, info))
}

/// Log where the white-matter mask comes from before flip detection.
fn announce_mask(path: Option<&Path>) {
    match path {
        Some(p) => eprintln!("mask: {}", p.display()),
        None => eprintln!("mask: none supplied — deriving an FA-threshold white-matter proxy"),
    }
}

fn print_summary(report: &Report) {
    let code = match report.status {
        Status::Pass => style::PASS,
        Status::Warn => style::WARN,
        Status::Flag => style::FLAG,
    };
    let status = style::paint(report.status.label(), code, style::stdout_color());
    println!("status: {status}");
    if let Some(shells) = &report.shells {
        let dwi: usize = shells
            .shells
            .iter()
            .filter(|s| !s.is_b0)
            .map(|s| s.count)
            .sum();
        println!(
            "scheme: {} shells, {} b0 + {} DWI volumes",
            shells.shells.len(),
            shells.b0.count,
            dwi
        );
    }
    if let Some(flip) = &report.flip {
        print_flip(flip, report.repair.is_some());
    }
    if let Some(repair) = &report.repair {
        print_repair(repair);
    }
    for note in &report.notes {
        println!("note: {note}");
    }
}

/// Print the candidate ranking, PASS/WARN/FLAG verdict, and `flag` markers for
/// conventions that beat the current table. `repair_followed` suppresses the repair hint when a repair block follows.
fn print_flip(flip: &FlipResult, repair_followed: bool) {
    println!();
    println!(
        "gradient orientation check: {} candidate conventions, working shell b={:.0}, {} WM voxels",
        flip.ranking.len(),
        flip.working_b,
        flip.n_wm_voxels
    );
    println!("ranked by fiber coherence (current table is +x+y+z)");
    println!("  rank  coherence  vs current  convention  flag");
    for (i, c) in flip.ranking.iter().enumerate() {
        let row = format!(
            "  {:>4}  {:>9.6}  {:>+9.2}%  {:<10}  {}",
            i + 1,
            c.coherence,
            relative_to_current(c.coherence, flip.identity_coherence) * 100.0,
            c.label,
            candidate_flag(i, c, flip)
        );
        println!("{}", row.trim_end());
    }
    print_decision(flip, repair_followed);
}

/// One line summarising the verdict, plus the repair recipe on FLAG.
fn print_decision(flip: &FlipResult, repair_followed: bool) {
    let color = style::stdout_color();
    println!();
    match flip.decision {
        Decision::Pass => println!(
            "decision: {} — gradient table is correct ({}); {:.2}% ahead of the next convention ({}).",
            style::paint("PASS", style::PASS, color),
            flip.best.label,
            flip.relative_margin * 100.0,
            flip.runner_up.label
        ),
        Decision::Warn => {
            println!(
                "decision: {} — best convention {} leads the runner-up ({}) by only {:.2}%; too close to call.",
                style::paint("WARN", style::WARN, color),
                flip.best.label,
                flip.runner_up.label,
                flip.relative_margin * 100.0
            );
            if repair_followed {
                println!("  force-repair applied despite the thin margin (--force-repair); verify the result.");
            } else {
                println!("  not auto-repairing; re-check with dwigradcheck or a larger --step.");
            }
        }
        Decision::Flag => {
            println!(
                "decision: {} — gradient table looks wrong: {} beats the current table by {:.2}%.",
                style::paint("FLAG", style::FLAG, color),
                flip.best.label,
                relative_to_current(flip.best.coherence, flip.identity_coherence) * 100.0
            );
            if let Some(label) = &flip.recommended_label {
                println!("  recommended repair: {} (label {})", axis_remap(label), label);
                if !repair_followed {
                    println!("  run `gradlint repair ...` to write a corrected bvec/bval.");
                }
            }
        }
    }
}

/// The repair block for the `repair` subcommand: what was changed and where.
fn print_repair(repair: &RepairInfo) {
    let color = style::stdout_color();
    println!();
    if repair.outputs.is_empty() {
        println!(
            "{}: {} (label {})",
            style::paint("repair (dry-run, nothing written)", style::DIM, color),
            axis_remap(&repair.label),
            repair.label
        );
    } else {
        println!(
            "{}: {} (label {})",
            style::paint("repair applied", style::BEST, color),
            axis_remap(&repair.label),
            repair.label
        );
        println!("  wrote: {}", repair.outputs.join(", "));
    }
}

/// Per-candidate marker: `best`, the current convention, or the WARN/FLAG tag
/// of any convention that outranks the current table.
fn candidate_flag(index: usize, c: &CandidateScore, flip: &FlipResult) -> String {
    let color = style::stdout_color();
    let mut tags: Vec<String> = Vec::new();
    let is_best = index == 0;
    if is_best {
        tags.push(style::paint("best", style::BEST, color));
    }
    if is_identity_convention(c) {
        tags.push(style::paint("current table", style::DIM, color));
    } else if beats_current(c, flip) {
        let code = match flip.decision {
            Decision::Flag => style::FLAG,
            Decision::Warn => style::WARN,
            Decision::Pass => style::DIM,
        };
        tags.push(style::paint(severity_word(flip.decision), code, color));
        if is_best && flip.decision == Decision::Flag {
            tags.push(style::paint("repair", style::BEST, color));
        }
    }
    tags.join(" · ")
}

/// `true` when a candidate scores above the current (identity) convention.
fn beats_current(c: &CandidateScore, flip: &FlipResult) -> bool {
    !is_identity_convention(c) && c.coherence > flip.identity_coherence * (1.0 + 1e-6)
}

fn severity_word(decision: Decision) -> &'static str {
    match decision {
        Decision::Flag => "FLAG",
        Decision::Warn => "WARN",
        Decision::Pass => "",
    }
}

/// Relative coherence gain of a candidate over the current (identity) convention.
fn relative_to_current(coherence: f64, identity_coherence: f64) -> f64 {
    if identity_coherence > 0.0 {
        (coherence - identity_coherence) / identity_coherence
    } else {
        0.0
    }
}

/// Identity or its antipode (`-x-y-z`): both are the current diffusion convention.
fn is_identity_convention(c: &CandidateScore) -> bool {
    const ID: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    c.is_identity || (0..3).all(|i| (0..3).all(|j| c.matrix[i][j] == -ID[i][j]))
}

/// Spell a convention label (`-x+z+y`) as the per-axis remap `x=-x, y=+z, z=+y`.
fn axis_remap(label: &str) -> String {
    let chars: Vec<char> = label.chars().collect();
    let dst = ['x', 'y', 'z'];
    (0..3)
        .map(|k| format!("{}={}{}", dst[k], chars[2 * k], chars[2 * k + 1]))
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_profile(read: &ReadTimings, detect: &DetectTimings, total: Duration) {
    let decompress = read.decompress.as_secs_f64();
    let convert = read.convert.as_secs_f64();
    let fit = detect.fit.as_secs_f64();
    let coherence = detect.coherence.as_secs_f64();
    let total = total.as_secs_f64();
    let other = (total - decompress - convert - fit - coherence).max(0.0);
    println!("profile (seconds):");
    println!("  decompress {decompress:8.3}");
    println!("  convert    {convert:8.3}");
    println!("  fit        {fit:8.3}");
    println!("  coherence  {coherence:8.3}");
    println!("  other      {other:8.3}");
    println!("  total      {total:8.3}");
}

fn read_volume_with_info(path: &Path) -> Result<(ndarray::ArrayD<f32>, VolumeInfo), String> {
    gradlint_core::read_volume_with_info(path).map_err(|e| e.to_string())
}

fn read_optional_mask(path: Option<&Path>) -> Result<Option<Vec<bool>>, String> {
    match path {
        Some(p) => Ok(Some(
            gradlint_core::read_mask(p).map_err(|e| e.to_string())?,
        )),
        None => Ok(None),
    }
}

fn input_file(path: &Path) -> Result<InputFile, String> {
    InputFile::of(path).map_err(|e| e.to_string())
}

fn repaired_sibling(path: &Path) -> PathBuf {
    tagged_sibling(path, "repaired")
}

fn tagged_sibling(path: &Path, tag: &str) -> PathBuf {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("gradlint");
    let name = match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{stem}.{tag}.{ext}"),
        None => format!("{stem}.{tag}"),
    };
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repaired_sibling_inserts_tag() {
        assert_eq!(
            repaired_sibling(Path::new("/d/dwi.bvec")),
            PathBuf::from("/d/dwi.repaired.bvec")
        );
        assert_eq!(
            repaired_sibling(Path::new("grad")),
            PathBuf::from("grad.repaired")
        );
    }

    fn score(label: &str, matrix: [[f64; 3]; 3], coherence: f64) -> CandidateScore {
        CandidateScore {
            label: label.to_string(),
            matrix,
            is_identity: matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            coherence,
            n_samples: 100,
        }
    }

    #[test]
    fn axis_remap_spells_the_label() {
        assert_eq!(axis_remap("+x+y+z"), "x=+x, y=+y, z=+z");
        assert_eq!(axis_remap("-x+z+y"), "x=-x, y=+z, z=+y");
    }

    #[test]
    fn identity_and_its_antipode_are_the_current_convention() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let neg = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        let flip_x = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(is_identity_convention(&score("+x+y+z", id, 0.8)));
        assert!(is_identity_convention(&score("-x-y-z", neg, 0.8)));
        assert!(!is_identity_convention(&score("-x+y+z", flip_x, 0.8)));
    }

    #[test]
    fn flag_marks_a_flip_winner_as_flag_and_repair() {
        let flip_x = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let flip = FlipResult {
            working_b: 1000.0,
            n_wm_voxels: 10,
            mask_mean_fa: 0.4,
            ranking: vec![score("-x+y+z", flip_x, 0.9), score("+x+y+z", id, 0.6)],
            best: score("-x+y+z", flip_x, 0.9),
            runner_up: score("+x+y+z", id, 0.6),
            identity_coherence: 0.6,
            margin: 0.3,
            relative_margin: 0.33,
            decision: Decision::Flag,
            recommended_transform: Some(flip_x),
            recommended_label: Some("-x+y+z".to_string()),
        };
        assert_eq!(
            candidate_flag(0, &flip.ranking[0], &flip),
            "best · FLAG · repair"
        );
        assert_eq!(candidate_flag(1, &flip.ranking[1], &flip), "current table");
    }
}
