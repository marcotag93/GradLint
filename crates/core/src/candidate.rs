//! The 48 signed-permutation candidate transforms for the gradient frame.

/// Number of signed permutation matrices: `2^3` sign flips × `3!` permutations.
pub const N_CANDIDATES: usize = 48;

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const PERMS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];
const AXES: [char; 3] = ['x', 'y', 'z'];

/// One signed-permutation transform `C` of the gradient axes.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Orthogonal signed-permutation matrix.
    pub matrix: [[f64; 3]; 3],
    /// Human-readable axis mapping, e.g. `+x-z+y`.
    pub label: String,
    /// `true` for the identity transform (the no-op convention).
    pub is_identity: bool,
}

/// Enumerate all 48 signed permutation matrices (identity included).
#[must_use]
pub fn all_candidates() -> Vec<Candidate> {
    let mut out = Vec::with_capacity(N_CANDIDATES);
    for perm in PERMS {
        for bits in 0..8u8 {
            let signs = [
                if bits & 1 != 0 { -1.0 } else { 1.0 },
                if bits & 2 != 0 { -1.0 } else { 1.0 },
                if bits & 4 != 0 { -1.0 } else { 1.0 },
            ];
            let mut matrix = [[0.0; 3]; 3];
            for (r, (&p, &s)) in perm.iter().zip(&signs).enumerate() {
                matrix[r][p] = s;
            }
            let label = perm
                .iter()
                .zip(&signs)
                .map(|(&p, &s)| format!("{}{}", if s < 0.0 { '-' } else { '+' }, AXES[p]))
                .collect();
            out.push(Candidate {
                is_identity: matrix == IDENTITY,
                matrix,
                label,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    #[test]
    fn enumerates_forty_eight_unique() {
        let all = all_candidates();
        assert_eq!(all.len(), N_CANDIDATES);
        let mut labels: Vec<&str> = all.iter().map(|c| c.label.as_str()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), N_CANDIDATES);
    }

    #[test]
    fn identity_present_and_unique() {
        let all = all_candidates();
        let ids: Vec<&Candidate> = all.iter().filter(|c| c.is_identity).collect();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].label, "+x+y+z");
        assert_eq!(apply(&ids[0].matrix, [0.3, -0.5, 0.8]), [0.3, -0.5, 0.8]);
    }

    #[test]
    fn matrices_are_orthogonal_signed_permutations() {
        for c in all_candidates() {
            let v = apply(&c.matrix, [1.0, 2.0, 3.0]);
            let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            assert!((norm_sq - 14.0).abs() < 1e-12);
        }
    }

    #[test]
    fn flip_x_negates_first_component() {
        let flip_x = all_candidates()
            .into_iter()
            .find(|c| c.label == "-x+y+z")
            .unwrap();
        assert_eq!(apply(&flip_x.matrix, [1.0, 2.0, 3.0]), [-1.0, 2.0, 3.0]);
    }
}
