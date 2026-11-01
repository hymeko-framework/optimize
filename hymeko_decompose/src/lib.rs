//! Dense matrix and tensor decompositions, dependency-free.
//!
//! - [`svd()`] / [`truncated_svd`] — the thin SVD by QR-preconditioned one-sided Jacobi: high
//!   relative accuracy for small singular values, O(m n²) once plus O(n³) per sweep.
//! - [`hosvd`] / [`hooi`] — Tucker decompositions; mode factors from the transposed unfolding's QR
//!   factor (no Gram matrix), blocked mode products, parallel independent modes.
//! - [`least_squares`] — batch least squares and ridge through the same SVD, minimum-norm on
//!   rank-deficient data.
//!
//! Every result carries its convergence evidence, and every failure is a typed
//! [`DecompositionError`]; nothing returns a silently degraded answer.
//!
//! Not in scope: iterative large-scale methods (Lanczos, randomised SVD), CP/PARAFAC, GPU kernels.

mod error;
mod jacobi;
mod matrix;
mod qr;
mod regression;
mod svd;
mod tucker;

pub use error::DecompositionError;
pub use matrix::MatrixRef;
pub use regression::{LeastSquaresOptions, Regression, least_squares};
pub use svd::{Convergence, Svd, SvdOptions, TruncatedSvd, svd, truncated_svd};
pub use tucker::{Factor, TensorRef, Tucker, TuckerOptions, hooi, hosvd};
