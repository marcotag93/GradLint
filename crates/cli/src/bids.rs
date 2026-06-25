//! BIDS dataset mode (reference binary): drive [`gradlint_core::bids_batch`],
//! stream the per-run verdicts and summary path, and gate on WARN-only exits.

use std::path::Path;

use gradlint_core::bids_batch;
use gradlint_core::pipeline::AuditOptions;

/// Audit every DWI under a BIDS root, printing each verdict as it lands.
pub fn run(root: &Path, options: AuditOptions, step: Option<f64>) -> Result<i32, String> {
    let outcome = bids_batch::run(root, options, step)?;
    for item in &outcome.items {
        println!("{}: {}", item.name, item.status.label());
    }
    println!("summary: {}", outcome.summary_path.display());
    Ok(outcome.worst.exit_code())
}
