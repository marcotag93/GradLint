//! Readers and writers for gradient tables, sidecars, and provenance.

pub mod bids;
pub mod fsl;
pub mod mrtrix;
pub mod provenance;

/// Decimal places for gradient direction components in FSL/MRtrix output.
/// Fixed precision keeps tables tidy and tool-agnostic, matching the `%.6f`
/// convention used by FSL/MRtrix and the validation writer.
const DIR_DECIMALS: usize = 6;

/// Format a direction component at fixed precision, normalizing `-0.0` to `0.0`.
fn fmt_component(x: f64) -> String {
    let v = if x == 0.0 { 0.0 } else { x };
    format!("{v:.prec$}", prec = DIR_DECIMALS)
}
