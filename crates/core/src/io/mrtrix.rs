//! MRtrix `.b` gradient-table reading and writing (world coordinates).

use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::gradient::GradientTable;

/// Read an MRtrix `.b` table (`x y z b` per row; `#` lines are comments).
pub fn read(path: impl AsRef<Path>) -> Result<GradientTable> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    let mut directions = Vec::new();
    let mut bvals = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let row = parse_row(line, path)?;
        directions.push([row[0], row[1], row[2]]);
        bvals.push(row[3]);
    }
    GradientTable::new(directions, bvals)
}

/// Write an MRtrix `.b` gradient table.
pub fn write(path: impl AsRef<Path>, table: &GradientTable) -> Result<()> {
    let mut out = String::new();
    for (d, b) in table.directions.iter().zip(&table.bvals) {
        out.push_str(&format!(
            "{} {} {} {}\n",
            super::fmt_component(d[0]),
            super::fmt_component(d[1]),
            super::fmt_component(d[2]),
            b
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn parse_row(line: &str, path: &Path) -> Result<[f64; 4]> {
    let nums: Vec<f64> = line
        .split_whitespace()
        .take(4)
        .map(|tok| {
            tok.parse::<f64>().map_err(|_| Error::Parse {
                path: path.display().to_string(),
                reason: format!("invalid number: {tok:?}"),
            })
        })
        .collect::<Result<_>>()?;
    if nums.len() < 4 {
        return Err(Error::Parse {
            path: path.display().to_string(),
            reason: format!("expected 4 values per row, found {}", nums.len()),
        });
    }
    Ok([nums[0], nums[1], nums[2], nums[3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_and_skips_comments() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("grad.b");
        fs::write(&p, "# header\n0 0 0 0\n1 0 0 1000\n\n0 1 0 1000\n").unwrap();
        let table = read(&p).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.bvals, vec![0.0, 1000.0, 1000.0]);

        write(&p, &table).unwrap();
        assert_eq!(read(&p).unwrap(), table);
    }

    #[test]
    fn short_row_is_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("bad.b");
        fs::write(&p, "1 0 0\n").unwrap();
        assert!(read(&p).is_err());
    }
}
