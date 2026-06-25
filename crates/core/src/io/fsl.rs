//! FSL `bvec`/`bval` reading and writing.

use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::gradient::GradientTable;

/// Read an FSL gradient table from separate `bvec` and `bval` files.
///
/// Accepts the canonical `3xN` layout and the transposed `Nx3` layout.
pub fn read(bvec_path: impl AsRef<Path>, bval_path: impl AsRef<Path>) -> Result<GradientTable> {
    let bvec_path = bvec_path.as_ref();
    let bval_path = bval_path.as_ref();
    let bvals = parse_floats(&fs::read_to_string(bval_path)?, bval_path)?;
    let rows = parse_rows(&fs::read_to_string(bvec_path)?, bvec_path)?;
    let directions = to_directions(&rows, bvals.len(), bvec_path)?;
    GradientTable::new(directions, bvals)
}

/// Write an FSL gradient table to `bvec` (3xN) and `bval` files.
pub fn write(
    bvec_path: impl AsRef<Path>,
    bval_path: impl AsRef<Path>,
    table: &GradientTable,
) -> Result<()> {
    let axis_row = |axis: usize| {
        table
            .directions
            .iter()
            .map(|d| super::fmt_component(d[axis]))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let bvec = format!("{}\n{}\n{}\n", axis_row(0), axis_row(1), axis_row(2));
    fs::write(bvec_path, bvec)?;

    let bval: Vec<String> = table.bvals.iter().map(ToString::to_string).collect();
    fs::write(bval_path, format!("{}\n", bval.join(" ")))?;
    Ok(())
}

fn parse_floats(text: &str, path: &Path) -> Result<Vec<f64>> {
    text.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>().map_err(|_| Error::Parse {
                path: path.display().to_string(),
                reason: format!("invalid number: {tok:?}"),
            })
        })
        .collect()
}

fn parse_rows(text: &str, path: &Path) -> Result<Vec<Vec<f64>>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| parse_floats(line, path))
        .collect()
}

fn to_directions(rows: &[Vec<f64>], n: usize, path: &Path) -> Result<Vec<[f64; 3]>> {
    // A 3x3 table matches both layouts; it is read as 3xN (axis-major). Flip
    // detection needs far more than 3 directions, so the ambiguity never bites.
    if rows.len() == 3 && rows.iter().all(|r| r.len() == n) {
        Ok((0..n)
            .map(|i| [rows[0][i], rows[1][i], rows[2][i]])
            .collect())
    } else if rows.len() == n && rows.iter().all(|r| r.len() == 3) {
        Ok(rows.iter().map(|r| [r[0], r[1], r[2]]).collect())
    } else {
        Err(Error::Parse {
            path: path.display().to_string(),
            reason: "bvec shape does not match the b-value count".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_3xn() {
        let table = GradientTable::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![0.0, 1000.0, 1000.0, 1000.0],
        )
        .unwrap();
        let dir = tempdir().unwrap();
        let bvec = dir.path().join("dwi.bvec");
        let bval = dir.path().join("dwi.bval");
        write(&bvec, &bval, &table).unwrap();
        assert_eq!(read(&bvec, &bval).unwrap(), table);
    }

    #[test]
    fn reads_transposed_nx3() {
        let dir = tempdir().unwrap();
        let bvec = dir.path().join("t.bvec");
        let bval = dir.path().join("t.bval");
        fs::write(&bvec, "0 0 0\n1 0 0\n0 1 0\n0 0 1\n").unwrap();
        fs::write(&bval, "0 1000 1000 1000\n").unwrap();
        let table = read(&bvec, &bval).unwrap();
        assert_eq!(table.len(), 4);
        assert_eq!(table.directions[1], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn writes_fixed_precision_and_normalizes_neg_zero() {
        let table =
            GradientTable::new(vec![[-0.0, 0.5773502691896258, -1.0]], vec![1000.0]).unwrap();
        let dir = tempdir().unwrap();
        let bvec = dir.path().join("p.bvec");
        let bval = dir.path().join("p.bval");
        write(&bvec, &bval, &table).unwrap();
        let text = fs::read_to_string(&bvec).unwrap();
        assert_eq!(text, "0.000000\n0.577350\n-1.000000\n");
    }

    #[test]
    fn shape_mismatch_is_error() {
        let dir = tempdir().unwrap();
        let bvec = dir.path().join("m.bvec");
        let bval = dir.path().join("m.bval");
        fs::write(&bvec, "0 1\n0 0\n0 0\n").unwrap();
        fs::write(&bval, "0 1000 1000\n").unwrap();
        assert!(read(&bvec, &bval).is_err());
    }

    #[test]
    fn ambiguous_3x3_is_read_as_3xn() {
        let dir = tempdir().unwrap();
        let bvec = dir.path().join("sq.bvec");
        let bval = dir.path().join("sq.bval");
        fs::write(&bvec, "1 2 3\n4 5 6\n7 8 9\n").unwrap();
        fs::write(&bval, "1000 1000 1000\n").unwrap();
        let table = read(&bvec, &bval).unwrap();
        assert_eq!(table.directions[0], [1.0, 4.0, 7.0]);
    }
}
