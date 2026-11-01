//! Batch linear least squares `min ‖X β − y‖² + λ‖β‖²` through the SVD, never the normal equations.
//!
//! `β = V diag(f(σ)) Uᵀ y` with `f(σ) = σ / (σ² + λ)` (ridge) or, for `λ = 0`, `1/σ` above the
//! cut-off and `0` below it — the minimum-norm (pseudo-inverse) solution, so a rank-deficient `X`
//! is answered, not refused. `U` is never formed: `Uᵀ y` is `left ᵀ (Qᵀ y)` on the reflectors.
//!
//! This is the batch solver only. The online, forgetting RLS readout is Nagare's
//! (`nagare_core::learning::online::EvolventHead`); nothing here is a second online estimator.

use crate::error::DecompositionError;
use crate::matrix::{ColMajor, MatrixRef, dot, norm2};
use crate::svd::{Convergence, SvdOptions, decompose_tall, rank_of};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeastSquaresOptions {
    /// Tikhonov weight `λ ≥ 0`.
    pub ridge: f64,
    /// Without ridge, singular values at or below `rcond · σ_max` are treated as zero; `None`
    /// means `max(m, n) · ε`.
    pub rcond: Option<f64>,
    pub svd: SvdOptions,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self { ridge: 0.0, rcond: None, svd: SvdOptions::default() }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Regression {
    pub coefficients: Vec<f64>,
    /// `‖y − X β‖₂`, computed directly from the data.
    pub residual_norm: f64,
    /// Numerical rank of `X` at the cut-off.
    pub rank: usize,
    pub singular_values: Vec<f64>,
    pub convergence: Convergence,
}

/// Solve `min ‖X β − y‖² + ridge ‖β‖²` for a row-major `X` (`samples × features`).
pub fn least_squares(x: MatrixRef<'_>, y: &[f64], options: &LeastSquaresOptions) -> Result<Regression, DecompositionError> {
    x.validate()?;
    options.svd.validate()?;
    let (m, n) = (x.rows, x.columns);
    if y.len() != m {
        return Err(DecompositionError::Shape(format!("{m} samples but {} targets", y.len())));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(DecompositionError::NonFinite);
    }
    if !(options.ridge.is_finite() && options.ridge >= 0.0) {
        return Err(DecompositionError::InvalidOption(format!("ridge {} must be finite and ≥ 0", options.ridge)));
    }
    if let Some(rcond) = options.rcond
        && !(rcond.is_finite() && rcond >= 0.0)
    {
        return Err(DecompositionError::InvalidOption(format!("rcond {rcond} must be finite and ≥ 0")));
    }
    let cut = options.rcond.unwrap_or(m.max(n) as f64 * f64::EPSILON);
    let filter = |sigma: &[f64]| -> Vec<f64> {
        let threshold = cut * sigma.first().copied().unwrap_or(0.0);
        sigma
            .iter()
            .map(|&s| {
                if options.ridge > 0.0 {
                    s / (s * s + options.ridge)
                } else if s > threshold {
                    1.0 / s
                } else {
                    0.0
                }
            })
            .collect()
    };
    let (coefficients, sigma, convergence) = if m >= n {
        // X = U Σ Vᵀ: β = V f(Σ) Uᵀ y.
        let tall = decompose_tall(ColMajor::from_row_major(x), &options.svd)?;
        let weights = filter(&tall.sigma);
        let projected = tall.left_transpose_times(y);
        let mut beta = vec![0.0; n];
        for (j, (&w, &p)) in weights.iter().zip(&projected).enumerate() {
            crate::matrix::axpy(w * p, tall.right.col(j), &mut beta);
        }
        (beta, tall.sigma, tall.convergence)
    } else {
        // Xᵀ = U' Σ V'ᵀ, so X = V' Σ U'ᵀ: β = U' f(Σ) V'ᵀ y.
        let tall = decompose_tall(ColMajor::transpose_of_row_major(x), &options.svd)?;
        let weights = filter(&tall.sigma);
        let z: Vec<f64> = (0..tall.sigma.len()).map(|j| weights[j] * dot(tall.right.col(j), y)).collect();
        (tall.left_times(&z), tall.sigma, tall.convergence)
    };
    let residual: Vec<f64> = x
        .data
        .chunks_exact(n)
        .zip(y)
        .map(|(row, &target)| target - dot(row, &coefficients))
        .collect();
    let rank = rank_of(&sigma, m.max(n), Some(cut * sigma.first().copied().unwrap_or(0.0)));
    Ok(Regression { coefficients, residual_norm: norm2(&residual), rank, singular_values: sigma, convergence })
}
