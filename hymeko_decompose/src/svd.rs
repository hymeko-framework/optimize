//! The thin SVD `A = U Σ Vᵀ`, by QR-preconditioned one-sided Jacobi.
//!
//! For a tall `A` (`m ≥ n`; a wide one is decomposed through `Aᵀ`):
//!
//! ```text
//!   A P = Q R          pivoted Householder QR, O(m n²), Q kept as reflectors
//!   Rᵀ V_J = U_J Σ     one-sided Jacobi on the n × n factor, O(n³) per sweep
//!   U = Q V_J,  V = P U_J
//! ```
//!
//! Every Jacobi sweep therefore costs O(n³) instead of O(m n²), and the graded `Rᵀ` produced by
//! the pivoting converges in few sweeps. Without preconditioning (`SvdOptions::precondition =
//! false`) Jacobi runs on `A` itself; that path exists as a cross-check and for tiny matrices.
//!
//! The input is scaled by its largest entry before factoring and the singular values are scaled
//! back, so neither overflow nor gradual underflow depends on the caller's units.

use crate::error::DecompositionError;
use crate::jacobi;
use crate::matrix::{ColMajor, MatrixRef, complete_orthonormal, dot};
use crate::qr::PivotedQr;

/// Explicit configuration; [`SvdOptions::default`] is the recommended setting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvdOptions {
    /// Relative column-orthogonality tolerance at convergence; `None` means `√n_rows · ε` of the
    /// matrix Jacobi runs on (LAPACK `dgesvj`'s choice).
    pub tolerance: Option<f64>,
    /// Sweep budget; exhausting it is [`DecompositionError::NotConverged`], never a silent answer.
    pub max_sweeps: usize,
    /// Pivoted-QR preconditioning (on by default; off only for cross-checks).
    pub precondition: bool,
    /// Singular values at or below this count as zero for [`Svd::rank`]; `None` means
    /// `max(m, n) · ε · σ_max`.
    pub rank_tolerance: Option<f64>,
    /// Worker threads for a Jacobi sweep (used from 96 columns up); the result does not depend on it.
    pub threads: usize,
}

impl Default for SvdOptions {
    fn default() -> Self {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self { tolerance: None, max_sweeps: 64, precondition: true, rank_tolerance: None, threads }
    }
}

impl SvdOptions {
    pub(crate) fn validate(&self) -> Result<(), DecompositionError> {
        if let Some(tolerance) = self.tolerance
            && !(tolerance.is_finite() && tolerance > 0.0 && tolerance < 1.0)
        {
            return Err(DecompositionError::InvalidOption(format!("tolerance {tolerance} outside (0, 1)")));
        }
        if let Some(rank) = self.rank_tolerance
            && !(rank.is_finite() && rank >= 0.0)
        {
            return Err(DecompositionError::InvalidOption(format!("rank tolerance {rank} must be finite and ≥ 0")));
        }
        if self.threads == 0 {
            return Err(DecompositionError::InvalidOption("threads must be ≥ 1".into()));
        }
        Ok(())
    }
}

/// How the Jacobi stage converged; part of every result as evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Convergence {
    pub sweeps: usize,
    /// Largest relative column coupling in the final sweep.
    pub off_diagonal: f64,
    pub preconditioned: bool,
}

/// A thin SVD of a `rows × columns` matrix, `k = min(rows, columns)`, row-major factors.
#[derive(Clone, Debug, PartialEq)]
pub struct Svd {
    pub rows: usize,
    pub columns: usize,
    /// `rows × k`, orthonormal columns.
    pub u: Vec<f64>,
    /// `k` values, non-increasing, non-negative.
    pub sigma: Vec<f64>,
    /// `k × columns`, orthonormal rows.
    pub vt: Vec<f64>,
    /// Count of singular values above the rank tolerance.
    pub rank: usize,
    pub convergence: Convergence,
}

impl Svd {
    pub fn k(&self) -> usize {
        self.sigma.len()
    }

    /// `U Σ Vᵀ`, row-major `rows × columns`.
    pub fn reconstruct(&self) -> Vec<f64> {
        let k = self.k();
        let mut out = vec![0.0; self.rows * self.columns];
        for i in 0..self.rows {
            let row = &mut out[i * self.columns..(i + 1) * self.columns];
            for j in 0..k {
                let weight = self.u[i * k + j] * self.sigma[j];
                if weight != 0.0 {
                    crate::matrix::axpy(weight, &self.vt[j * self.columns..(j + 1) * self.columns], row);
                }
            }
        }
        out
    }

    /// The leading `k` triplets (Eckart–Young best rank-`k` approximation).
    pub fn truncate(&self, k: usize) -> Self {
        let full = self.k();
        let k = k.min(full);
        let mut u = Vec::with_capacity(self.rows * k);
        for i in 0..self.rows {
            u.extend_from_slice(&self.u[i * full..i * full + k]);
        }
        Self {
            rows: self.rows,
            columns: self.columns,
            u,
            sigma: self.sigma[..k].to_vec(),
            vt: self.vt[..k * self.columns].to_vec(),
            rank: self.rank.min(k),
            convergence: self.convergence,
        }
    }
}

/// A rank-`k` truncation and the energy it discards, `Σ_{i>k} σᵢ² = ‖A − A_k‖²_F`.
#[derive(Clone, Debug, PartialEq)]
pub struct TruncatedSvd {
    pub svd: Svd,
    pub discarded_energy: f64,
}

/// The thin SVD.
pub fn svd(a: MatrixRef<'_>, options: &SvdOptions) -> Result<Svd, DecompositionError> {
    a.validate()?;
    options.validate()?;
    let (m, n) = (a.rows, a.columns);
    let (u, vt, sigma, convergence) = if m >= n {
        let tall = decompose_tall(ColMajor::from_row_major(a), options)?;
        (tall.full_left().to_row_major(), tall.right.transpose_to_row_major(), tall.sigma, tall.convergence)
    } else {
        // Aᵀ = U' Σ V'ᵀ  ⇒  A = V' Σ U'ᵀ.
        let tall = decompose_tall(ColMajor::transpose_of_row_major(a), options)?;
        (tall.right.to_row_major(), tall.full_left().transpose_to_row_major(), tall.sigma, tall.convergence)
    };
    let rank = rank_of(&sigma, m.max(n), options.rank_tolerance);
    Ok(Svd { rows: m, columns: n, u, sigma, vt, rank, convergence })
}

/// The best rank-`k` approximation and its discarded energy.
pub fn truncated_svd(a: MatrixRef<'_>, k: usize, options: &SvdOptions) -> Result<TruncatedSvd, DecompositionError> {
    let full = svd(a, options)?;
    if k == 0 || k > full.k() {
        return Err(DecompositionError::Shape(format!("rank {k} outside 1..={}", full.k())));
    }
    let discarded_energy = full.sigma[k..].iter().map(|s| s * s).sum();
    Ok(TruncatedSvd { svd: full.truncate(k), discarded_energy })
}

pub(crate) fn rank_of(sigma: &[f64], max_dim: usize, tolerance: Option<f64>) -> usize {
    let largest = sigma.first().copied().unwrap_or(0.0);
    let threshold = tolerance.unwrap_or(max_dim as f64 * f64::EPSILON * largest);
    sigma.iter().filter(|&&s| s > threshold).count()
}

/// `X = U Σ Vᵀ` for a tall `X` (`m ≥ n`), with `U = Q · [left; 0]` when preconditioned and
/// `U = left` otherwise. Consumers that only need `Uᵀy`, `U z` or `V` never form `U`.
pub(crate) struct TallSvd {
    pub qr: Option<PivotedQr>,
    pub rows: usize,
    pub left: ColMajor,
    pub sigma: Vec<f64>,
    pub right: ColMajor,
    pub convergence: Convergence,
}

impl TallSvd {
    /// `U`, `m × n`, orthonormal columns.
    pub fn full_left(&self) -> ColMajor {
        match &self.qr {
            None => self.left.clone(),
            Some(qr) => {
                let n = self.left.cols;
                let mut u = ColMajor::zeros(self.rows, n);
                for j in 0..n {
                    u.col_mut(j)[..self.left.rows].copy_from_slice(self.left.col(j));
                }
                qr.apply_q(&mut u);
                u
            }
        }
    }

    /// `Uᵀ y`.
    pub fn left_transpose_times(&self, y: &[f64]) -> Vec<f64> {
        let mut work = y.to_vec();
        if let Some(qr) = &self.qr {
            qr.apply_qt_vec(&mut work);
        }
        let head = &work[..self.left.rows];
        (0..self.left.cols).map(|j| dot(self.left.col(j), head)).collect()
    }

    /// `U z`.
    pub fn left_times(&self, z: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.rows];
        for (j, &weight) in z.iter().enumerate() {
            if weight != 0.0 {
                crate::matrix::axpy(weight, self.left.col(j), &mut out[..self.left.rows]);
            }
        }
        if let Some(qr) = &self.qr {
            qr.apply_q_vec(&mut out);
        }
        out
    }
}

pub(crate) fn decompose_tall(mut x: ColMajor, options: &SvdOptions) -> Result<TallSvd, DecompositionError> {
    let (m, n) = (x.rows, x.cols);
    debug_assert!(m >= n);
    let scale = x.max_abs();
    if scale == 0.0 {
        // The zero matrix: Σ = 0 and any orthonormal pair is exact.
        let mut left = ColMajor::zeros(m, n);
        complete_orthonormal(&mut left, 0);
        return Ok(TallSvd {
            qr: None,
            rows: m,
            left,
            sigma: vec![0.0; n],
            right: ColMajor::identity(n),
            convergence: Convergence { sweeps: 0, off_diagonal: 0.0, preconditioned: false },
        });
    }
    x.scale(1.0 / scale);
    let tolerance = |rows: usize| options.tolerance.unwrap_or((rows as f64).sqrt() * f64::EPSILON);
    let tall = if options.precondition {
        let qr = PivotedQr::factor(x);
        let result = jacobi::orthogonalise(qr.r_transposed(), tolerance(n), options.max_sweeps, options.threads)?;
        // Rᵀ = U_J Σ V_Jᵀ  ⇒  R = V_J Σ U_Jᵀ  ⇒  X = Q V_J Σ (P U_J)ᵀ.
        let u_j = normalised(&result.w, &result.sigma);
        let mut right = ColMajor::zeros(n, n);
        for j in 0..n {
            for (i, &value) in u_j.col(j).iter().enumerate() {
                right.data[j * n + qr.perm[i]] = value;
            }
        }
        let convergence = Convergence { sweeps: result.sweeps, off_diagonal: result.off_diagonal, preconditioned: true };
        debug_assert_eq!(qr.cols(), n);
        TallSvd { qr: Some(qr), rows: m, left: result.v, sigma: result.sigma, right, convergence }
    } else {
        let result = jacobi::orthogonalise(x, tolerance(m), options.max_sweeps, options.threads)?;
        let left = normalised(&result.w, &result.sigma);
        let convergence = Convergence { sweeps: result.sweeps, off_diagonal: result.off_diagonal, preconditioned: false };
        TallSvd { qr: None, rows: m, left, sigma: result.sigma, right: result.v, convergence }
    };
    let mut tall = tall;
    tall.sigma.iter_mut().for_each(|s| *s *= scale);
    Ok(tall)
}

/// `W Σ⁻¹`, with the columns of exactly-zero singular values completed to an orthonormal set.
fn normalised(w: &ColMajor, sigma: &[f64]) -> ColMajor {
    let mut u = w.clone();
    let filled = sigma.iter().take_while(|&&s| s > 0.0).count();
    for (j, &s) in sigma.iter().enumerate().take(filled) {
        let inverse = 1.0 / s;
        u.col_mut(j).iter_mut().for_each(|value| *value *= inverse);
    }
    complete_orthonormal(&mut u, filled);
    u
}
