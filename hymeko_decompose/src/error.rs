//! Why a decomposition did not return an answer.

use std::fmt;

/// Every failure is a typed refusal; no routine returns a partial or silently degraded result.
#[derive(Clone, Debug, PartialEq)]
pub enum DecompositionError {
    /// Dimensions, data length or requested ranks are inconsistent.
    Shape(String),
    /// The input contains NaN or an infinity.
    NonFinite,
    /// An option is outside its domain (negative ridge, non-positive tolerance, ...).
    InvalidOption(String),
    /// Jacobi sweeps ran out with column pairs still above the orthogonality tolerance.
    NotConverged {
        /// Sweeps performed.
        sweeps: usize,
        /// Largest relative column coupling `|xₚ·x_q| / (‖xₚ‖‖x_q‖)` in the last sweep.
        off_diagonal: f64,
    },
}

impl fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) => write!(f, "shape: {reason}"),
            Self::NonFinite => write!(f, "input contains a non-finite value"),
            Self::InvalidOption(reason) => write!(f, "invalid option: {reason}"),
            Self::NotConverged { sweeps, off_diagonal } => write!(
                f,
                "one-sided Jacobi did not converge in {sweeps} sweeps (relative off-diagonal {off_diagonal:e})"
            ),
        }
    }
}

impl std::error::Error for DecompositionError {}
