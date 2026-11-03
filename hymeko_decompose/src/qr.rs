//! Householder QR with column pivoting: `A P = Q R`, the Jacobi preconditioner.
//!
//! Pivoting by the largest remaining column norm makes `R` rank-revealing and its rows graded,
//! which is what makes one-sided Jacobi on `Rᵀ` converge in few sweeps (Drmač & Veselić, "New fast
//! and accurate Jacobi SVD algorithm", SIMAX 29, 2008). `Q` is never formed: it is kept as its
//! reflectors and applied on demand, which is all the SVD, HOSVD and least-squares paths need.

use crate::matrix::{ColMajor, dot, norm2};

/// The factored form. Reflector `k` is `H_k = I − τ_k v_k v_kᵀ` with `v_k[k] = 1` implicit and
/// `v_k[k+1..]` stored below the diagonal of column `k`; `R` is the upper triangle.
pub(crate) struct PivotedQr {
    factors: ColMajor,
    tau: Vec<f64>,
    /// Column `j` of `A P` is column `perm[j]` of `A`.
    pub perm: Vec<usize>,
}

impl PivotedQr {
    /// Factor a tall or square matrix (`rows ≥ cols`), consuming it.
    pub fn factor(mut a: ColMajor) -> Self {
        let (m, n) = (a.rows, a.cols);
        debug_assert!(m >= n);
        let mut perm: Vec<usize> = (0..n).collect();
        let mut tau = vec![0.0; n];
        let mut norms: Vec<f64> = (0..n).map(|j| norm2(a.col(j))).collect();
        let mut reference = norms.clone();
        // LAPACK dlaqp2's cancellation guard for the downdated partial norms.
        let guard = f64::EPSILON.sqrt();
        for k in 0..n {
            let pivot = k + (k..n).fold(0, |best, j| if norms[j] > norms[k + best] { j - k } else { best });
            if pivot != k {
                a.swap_cols(k, pivot);
                perm.swap(k, pivot);
                norms.swap(k, pivot);
                reference.swap(k, pivot);
            }
            tau[k] = householder(&mut a.col_mut(k)[k..]);
            if tau[k] != 0.0 {
                let (head, tail) = a.data.split_at_mut((k + 1) * m);
                let v = &head[k * m + k..k * m + m];
                for column in tail.chunks_exact_mut(m) {
                    reflect(v, tau[k], &mut column[k..]);
                }
            }
            for j in k + 1..n {
                if norms[j] == 0.0 {
                    continue;
                }
                let ratio = a.col(j)[k].abs() / norms[j];
                let shrink = (1.0 - ratio * ratio).max(0.0);
                let relative = shrink * (norms[j] / reference[j]).powi(2);
                if relative <= guard {
                    norms[j] = norm2(&a.col(j)[k + 1..]);
                    reference[j] = norms[j];
                } else {
                    norms[j] *= shrink.sqrt();
                }
            }
        }
        Self { factors: a, tau, perm }
    }

    pub fn cols(&self) -> usize {
        self.factors.cols
    }

    /// `Rᵀ` (`n × n`, lower triangular), column-major.
    pub fn r_transposed(&self) -> ColMajor {
        let n = self.factors.cols;
        let mut rt = ColMajor::zeros(n, n);
        for j in 0..n {
            for i in 0..=j {
                rt.data[i * n + j] = self.factors.col(j)[i];
            }
        }
        rt
    }

    /// `B ← Q B` for `B` with `rows` rows (`Q = H_0 H_1 ⋯ H_{n−1}`).
    pub fn apply_q(&self, b: &mut ColMajor) {
        let m = self.factors.rows;
        debug_assert_eq!(b.rows, m);
        for k in (0..self.factors.cols).rev() {
            if self.tau[k] == 0.0 {
                continue;
            }
            let v = &self.factors.col(k)[k..];
            for column in b.data.chunks_exact_mut(m) {
                reflect(v, self.tau[k], &mut column[k..]);
            }
        }
    }

    /// `y ← Q y`.
    pub fn apply_q_vec(&self, y: &mut [f64]) {
        for k in (0..self.factors.cols).rev() {
            if self.tau[k] != 0.0 {
                reflect(&self.factors.col(k)[k..], self.tau[k], &mut y[k..]);
            }
        }
    }

    /// `y ← Qᵀ y`.
    pub fn apply_qt_vec(&self, y: &mut [f64]) {
        for k in 0..self.factors.cols {
            if self.tau[k] != 0.0 {
                reflect(&self.factors.col(k)[k..], self.tau[k], &mut y[k..]);
            }
        }
    }
}

/// Turn `x` into `(β, 0, …, 0)` in place, storing `v[1..]` in `x[1..]` (dlarfg); returns `τ`.
fn householder(x: &mut [f64]) -> f64 {
    let alpha = x[0];
    let tail_norm = norm2(&x[1..]);
    if tail_norm == 0.0 {
        return 0.0;
    }
    let beta = -alpha.signum() * alpha.hypot(tail_norm);
    let tau = (beta - alpha) / beta;
    let scale = 1.0 / (alpha - beta);
    x[1..].iter_mut().for_each(|value| *value *= scale);
    x[0] = beta;
    tau
}

/// `y ← (I − τ v vᵀ) y` with `v[0] = 1` implicit (the stored `v[0]` is `R`'s diagonal, not read).
#[inline]
fn reflect(v: &[f64], tau: f64, y: &mut [f64]) {
    let w = tau * (y[0] + dot(&v[1..], &y[1..]));
    y[0] -= w;
    for (yi, vi) in y[1..].iter_mut().zip(&v[1..]) {
        *yi -= w * vi;
    }
}
