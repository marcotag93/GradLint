//! Cross-source consistency checks for a gradient table.

use crate::error::{Error, Result};
use crate::gradient::GradientTable;

/// Verify the gradient count matches the DWI volume count, when known.
pub fn check_volume_count(table: &GradientTable, num_volumes: Option<usize>) -> Result<()> {
    match num_volumes {
        Some(volumes) if table.len() != volumes => Err(Error::VolumeCountMismatch {
            gradients: table.len(),
            volumes,
        }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_count_mismatch_detected() {
        let t = GradientTable::new(vec![[0.0, 0.0, 0.0]], vec![0.0]).unwrap();
        assert!(check_volume_count(&t, Some(2)).is_err());
        assert!(check_volume_count(&t, Some(1)).is_ok());
        assert!(check_volume_count(&t, None).is_ok());
    }
}
