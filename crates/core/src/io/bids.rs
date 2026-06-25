//! BIDS diffusion sidecar (`.json`) reading.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Subset of BIDS sidecar fields relevant to diffusion QC.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct BidsSidecar {
    #[serde(rename = "PhaseEncodingDirection")]
    pub phase_encoding_direction: Option<String>,
    #[serde(rename = "TotalReadoutTime")]
    pub total_readout_time: Option<f64>,
    #[serde(rename = "MultibandAccelerationFactor")]
    pub multiband_acceleration_factor: Option<u32>,
}

/// Read a BIDS sidecar JSON file. Unknown fields are ignored.
pub fn read(path: impl AsRef<Path>) -> Result<BidsSidecar> {
    let reader = BufReader::new(File::open(path)?);
    Ok(serde_json::from_reader(reader)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parses_known_fields_and_ignores_unknown() {
        let json = r#"{"PhaseEncodingDirection":"j-","TotalReadoutTime":0.05,"EchoTime":0.09}"#;
        let sidecar: BidsSidecar = serde_json::from_str(json).unwrap();
        assert_eq!(sidecar.phase_encoding_direction.as_deref(), Some("j-"));
        assert_eq!(sidecar.total_readout_time, Some(0.05));
        assert_eq!(sidecar.multiband_acceleration_factor, None);
    }

    #[test]
    fn reads_from_file() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("dwi.json");
        std::fs::write(&p, r#"{"MultibandAccelerationFactor":3}"#).unwrap();
        assert_eq!(read(&p).unwrap().multiband_acceleration_factor, Some(3));
    }
}
