//! Tucker decompositions: HOSVD (De Lathauwer, De Moor & Vandewalle, SIMAX 21, 2000) and HOOI.
//!
//! A row-major order-N tensor is viewed, for mode `n`, as `L × Iₙ × R` with `L = ∏_{m<n} I_m` and
//! `R = ∏_{m>n} I_m`. Two choices keep it fast without giving up accuracy:
//!
//! - **No Gram matrix, no Q.** The mode-n factor is the leading left singular vectors of the
//!   unfolding `Y₍ₙ₎` (`Iₙ × J`). Forming `Y Yᵀ` would square the condition number. Instead the
//!   transposed unfolding `Yᵀ` (`J × Iₙ`, usually very tall) is QR-factored and Jacobi runs on the
//!   `Iₙ × Iₙ` factor: `Yᵀ P = Q R`, `Rᵀ = U_J Σ V_Jᵀ` gives `Y = (P U_J) Σ (Q V_J)ᵀ`, so the factor
//!   is `P U_J` and `Q` is never applied. Cost O(J Iₙ²) per mode.
//! - **Blocked mode products.** `T ×ₙ Uᵀ` streams contiguous `R`-blocks through an axpy (or a dot
//!   when `R = 1`), O(size · rank). Modes are contracted in order of decreasing compression, so the
//!   intermediate tensors shrink as early as possible.
//!
//! HOSVD's modes are independent and run on scoped threads when `parallel` is set; the result is
//! bit-identical to the sequential one because each mode's arithmetic is unchanged.

use crate::error::DecompositionError;
use crate::matrix::{ColMajor, axpy, complete_orthonormal, dot, norm2};
use crate::svd::{Convergence, SvdOptions, decompose_tall};

/// An order-N dense tensor, row-major (last index fastest).
#[derive(Clone, Copy, Debug)]
pub struct TensorRef<'a> {
    pub shape: &'a [usize],
    pub data: &'a [f64],
}

impl TensorRef<'_> {
    fn validate(&self) -> Result<usize, DecompositionError> {
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(DecompositionError::Shape(format!("tensor shape {:?} has no entries", self.shape)));
        }
        let size = self
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| DecompositionError::Shape("dimension product overflows".into()))?;
        if self.data.len() != size {
            return Err(DecompositionError::Shape(format!("shape {:?} needs {size} values, got {}", self.shape, self.data.len())));
        }
        if self.data.iter().any(|v| !v.is_finite()) {
            return Err(DecompositionError::NonFinite);
        }
        Ok(size)
    }
}

/// One orthonormal mode factor, row-major `rows × rank`, with the singular values of the
/// unfolding it was taken from (all of them, not only the retained ones).
#[derive(Clone, Debug, PartialEq)]
pub struct Factor {
    pub rows: usize,
    pub rank: usize,
    pub matrix: Vec<f64>,
    pub spectrum: Vec<f64>,
}

/// `T ≈ G ×₁ U₁ ×₂ ⋯ ×_N U_N` with orthonormal factors.
#[derive(Clone, Debug, PartialEq)]
pub struct Tucker {
    pub shape: Vec<usize>,
    pub core_shape: Vec<usize>,
    /// Row-major core tensor `G`.
    pub core: Vec<f64>,
    pub factors: Vec<Factor>,
    /// `‖T − T̂‖_F / ‖T‖_F` from `‖T‖² − ‖G‖²` (exact for orthonormal factors; resolution about
    /// `√ε` because of the subtraction — reconstruct explicitly for smaller errors).
    pub relative_error: f64,
    /// Relative error after the HOSVD start and after every HOOI sweep (one entry for HOSVD).
    pub history: Vec<f64>,
    /// Jacobi evidence of the last factor computation of each mode.
    pub convergence: Vec<Convergence>,
}

impl Tucker {
    /// `T̂ = G ×₁ U₁ ⋯ ×_N U_N`, row-major in the original shape.
    pub fn reconstruct(&self) -> Vec<f64> {
        let mut data = self.core.clone();
        let mut shape = self.core_shape.clone();
        for (mode, factor) in self.factors.iter().enumerate() {
            data = expand(&data, &shape, mode, factor);
            shape[mode] = factor.rows;
        }
        data
    }

    /// The HOSVD a-priori bound `√(Σₙ Σ_{i>Rₙ} σᵢ⁽ⁿ⁾²) / ‖T‖` (holds for HOSVD; HOOI only improves on it).
    pub fn error_bound(&self, norm: f64) -> f64 {
        let discarded: f64 = self.factors.iter().map(|f| f.spectrum.iter().skip(f.rank).map(|s| s * s).sum::<f64>()).sum();
        if norm == 0.0 { 0.0 } else { discarded.sqrt() / norm }
    }
}

/// Explicit configuration for HOSVD and HOOI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TuckerOptions {
    pub svd: SvdOptions,
    /// Compute HOSVD's independent mode factors on scoped threads.
    pub parallel: bool,
    /// HOOI stops when a sweep lowers the relative error by at most this much.
    pub tolerance: f64,
    /// HOOI sweep budget (reaching it is not an error; `history` records the trajectory).
    pub max_sweeps: usize,
}

impl Default for TuckerOptions {
    fn default() -> Self {
        Self { svd: SvdOptions::default(), parallel: true, tolerance: 1e-10, max_sweeps: 50 }
    }
}

/// One mode's orthonormal basis, its unfolding's full spectrum and the Jacobi evidence.
struct ModeBasis {
    basis: ColMajor,
    spectrum: Vec<f64>,
    convergence: Convergence,
}

/// The truncated HOSVD at multilinear rank `ranks`.
pub fn hosvd(t: TensorRef<'_>, ranks: &[usize], options: &TuckerOptions) -> Result<Tucker, DecompositionError> {
    validate(&t, ranks, options)?;
    let order = t.shape.len();
    let parallel = options.parallel && order > 1;
    // Concurrent modes share the thread budget instead of oversubscribing it.
    let inner = SvdOptions { threads: if parallel { (options.svd.threads / order).max(1) } else { options.svd.threads }, ..options.svd };
    let compute = |mode: usize| left_vectors(t.data, t.shape, mode, ranks[mode], &inner);
    let results: Vec<Result<ModeBasis, DecompositionError>> = if parallel {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..order).map(|mode| scope.spawn(move || compute(mode))).collect();
            handles.into_iter().map(|h| h.join().expect("mode factor thread panicked")).collect()
        })
    } else {
        (0..order).map(compute).collect()
    };
    let mut bases = Vec::with_capacity(order);
    let mut spectra = Vec::with_capacity(order);
    let mut convergence = Vec::with_capacity(order);
    for result in results {
        let mode = result?;
        bases.push(mode.basis);
        spectra.push(mode.spectrum);
        convergence.push(mode.convergence);
    }
    let modes: Vec<usize> = contraction_order(t.shape, ranks, None);
    let (core, core_shape) = project_all(t.data, t.shape, &bases, &modes);
    let norm = norm2(t.data);
    let relative_error = relative_error(norm, &core);
    Ok(Tucker {
        shape: t.shape.to_vec(),
        core_shape,
        core,
        factors: bases.iter().zip(spectra).map(|(b, s)| factor_of(b, s)).collect(),
        relative_error,
        history: vec![relative_error],
        convergence,
    })
}

/// Higher-order orthogonal iteration from the HOSVD start: alternate each factor as the leading
/// left singular vectors of the tensor projected on all other factors.
pub fn hooi(t: TensorRef<'_>, ranks: &[usize], options: &TuckerOptions) -> Result<Tucker, DecompositionError> {
    let mut tucker = hosvd(t, ranks, options)?;
    let order = t.shape.len();
    let norm = norm2(t.data);
    let mut bases: Vec<ColMajor> = tucker.factors.iter().map(basis_of).collect();
    let mut previous = tucker.relative_error;
    for _ in 0..options.max_sweeps {
        let mut core = Vec::new();
        let mut core_shape = Vec::new();
        for mode in 0..order {
            let others = contraction_order(t.shape, ranks, Some(mode));
            let (projected, shape) = project_all(t.data, t.shape, &bases, &others);
            let ModeBasis { basis, spectrum, convergence: conv } = left_vectors(&projected, &shape, mode, ranks[mode], &options.svd)?;
            if mode == order - 1 {
                core = project(&projected, &shape, mode, &basis);
                core_shape = shape.clone();
                core_shape[mode] = ranks[mode];
            }
            tucker.factors[mode] = factor_of(&basis, spectrum);
            tucker.convergence[mode] = conv;
            bases[mode] = basis;
        }
        let error = relative_error(norm, &core);
        tucker.core = core;
        tucker.core_shape = core_shape;
        tucker.relative_error = error;
        tucker.history.push(error);
        if previous - error <= options.tolerance {
            break;
        }
        previous = error;
    }
    Ok(tucker)
}

fn validate(t: &TensorRef<'_>, ranks: &[usize], options: &TuckerOptions) -> Result<(), DecompositionError> {
    t.validate()?;
    options.svd.validate()?;
    if !(options.tolerance.is_finite() && options.tolerance >= 0.0) {
        return Err(DecompositionError::InvalidOption(format!("HOOI tolerance {} must be finite and ≥ 0", options.tolerance)));
    }
    if ranks.len() != t.shape.len() {
        return Err(DecompositionError::Shape(format!("{} ranks for an order-{} tensor", ranks.len(), t.shape.len())));
    }
    for (mode, (&rank, &dim)) in ranks.iter().zip(t.shape).enumerate() {
        if rank == 0 || rank > dim {
            return Err(DecompositionError::Shape(format!("mode {mode}: rank {rank} outside 1..={dim}")));
        }
    }
    Ok(())
}

fn relative_error(norm: f64, core: &[f64]) -> f64 {
    if norm == 0.0 {
        return 0.0;
    }
    let kept = norm2(core);
    ((norm - kept) * (norm + kept)).max(0.0).sqrt() / norm
}

/// The `count` leading left singular vectors of the mode-`mode` unfolding (`Iₙ × count`,
/// column-major) and the unfolding's full spectrum.
fn left_vectors(data: &[f64], shape: &[usize], mode: usize, count: usize, options: &SvdOptions) -> Result<ModeBasis, DecompositionError> {
    let (outer, dim, inner) = split(shape, mode);
    let others = outer * inner;
    if others >= dim {
        // Yᵀ is J × Iₙ, tall: Y = right · Σ · (Q left)ᵀ, so the factor is `right`.
        let mut yt = ColMajor::zeros(others, dim);
        for i in 0..dim {
            let column = yt.col_mut(i);
            for l in 0..outer {
                let source = (l * dim + i) * inner;
                column[l * inner..(l + 1) * inner].copy_from_slice(&data[source..source + inner]);
            }
        }
        let tall = decompose_tall(yt, options)?;
        Ok(ModeBasis { basis: tall.right.leading_cols(count), spectrum: tall.sigma, convergence: tall.convergence })
    } else {
        // Y is Iₙ × J, tall: its thin left factor has J columns; complete to `count`.
        let mut y = ColMajor::zeros(dim, others);
        for l in 0..outer {
            for r in 0..inner {
                let column = y.col_mut(l * inner + r);
                for (i, slot) in column.iter_mut().enumerate() {
                    *slot = data[(l * dim + i) * inner + r];
                }
            }
        }
        let tall = decompose_tall(y, options)?;
        let thin = tall.full_left();
        let mut basis = ColMajor::zeros(dim, count);
        let kept = count.min(others);
        basis.data[..dim * kept].copy_from_slice(&thin.data[..dim * kept]);
        complete_orthonormal(&mut basis, kept);
        Ok(ModeBasis { basis, spectrum: tall.sigma, convergence: tall.convergence })
    }
}

/// `(∏_{m<mode} I_m, I_mode, ∏_{m>mode} I_m)`.
fn split(shape: &[usize], mode: usize) -> (usize, usize, usize) {
    (shape[..mode].iter().product(), shape[mode], shape[mode + 1..].iter().product())
}

/// `T ×ₙ Uᵀ` for `U` (`Iₙ × R`, column-major): the result has `R` in place of `Iₙ`.
fn project(data: &[f64], shape: &[usize], mode: usize, basis: &ColMajor) -> Vec<f64> {
    let (outer, dim, inner) = split(shape, mode);
    let rank = basis.cols;
    let mut out = vec![0.0; outer * rank * inner];
    for l in 0..outer {
        let slab = &data[l * dim * inner..(l + 1) * dim * inner];
        for r in 0..rank {
            let u = basis.col(r);
            if inner == 1 {
                out[l * rank + r] = dot(u, slab);
            } else {
                let block = &mut out[(l * rank + r) * inner..(l * rank + r + 1) * inner];
                for (i, &weight) in u.iter().enumerate() {
                    if weight != 0.0 {
                        axpy(weight, &slab[i * inner..(i + 1) * inner], block);
                    }
                }
            }
        }
    }
    out
}

/// `G ×ₙ U` for a row-major factor (`Iₙ × R`): the result has `Iₙ` in place of `R`.
fn expand(data: &[f64], shape: &[usize], mode: usize, factor: &Factor) -> Vec<f64> {
    let (outer, rank, inner) = split(shape, mode);
    let dim = factor.rows;
    let mut out = vec![0.0; outer * dim * inner];
    for l in 0..outer {
        let slab = &data[l * rank * inner..(l + 1) * rank * inner];
        for i in 0..dim {
            let row = &factor.matrix[i * rank..(i + 1) * rank];
            if inner == 1 {
                out[l * dim + i] = dot(row, slab);
            } else {
                let block = &mut out[(l * dim + i) * inner..(l * dim + i + 1) * inner];
                for (r, &weight) in row.iter().enumerate() {
                    if weight != 0.0 {
                        axpy(weight, &slab[r * inner..(r + 1) * inner], block);
                    }
                }
            }
        }
    }
    out
}

/// Project on the given modes in order; returns the data and its shape.
fn project_all(data: &[f64], shape: &[usize], bases: &[ColMajor], modes: &[usize]) -> (Vec<f64>, Vec<usize>) {
    let mut current = data.to_vec();
    let mut current_shape = shape.to_vec();
    for &mode in modes {
        current = project(&current, &current_shape, mode, &bases[mode]);
        current_shape[mode] = bases[mode].cols;
    }
    (current, current_shape)
}

/// Modes (optionally excluding one) by increasing `Rₙ / Iₙ`: the strongest compression first.
fn contraction_order(shape: &[usize], ranks: &[usize], skip: Option<usize>) -> Vec<usize> {
    let mut modes: Vec<usize> = (0..shape.len()).filter(|&m| Some(m) != skip).collect();
    modes.sort_by(|&a, &b| (ranks[a] * shape[b]).cmp(&(ranks[b] * shape[a])).then(a.cmp(&b)));
    modes
}

fn factor_of(basis: &ColMajor, spectrum: Vec<f64>) -> Factor {
    Factor { rows: basis.rows, rank: basis.cols, matrix: basis.to_row_major(), spectrum }
}

fn basis_of(factor: &Factor) -> ColMajor {
    let mut basis = ColMajor::zeros(factor.rows, factor.rank);
    for i in 0..factor.rows {
        for r in 0..factor.rank {
            basis.data[r * factor.rows + i] = factor.matrix[i * factor.rank + r];
        }
    }
    basis
}
