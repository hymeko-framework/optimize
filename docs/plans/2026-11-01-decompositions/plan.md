# SVD, HOSVD and linear regression — plan and API templates

Status: PLAN ONLY (user, 2026-11-01: "egyelőre csak tervek és sablonok"). No implementation, no
dependency change and no CORE.YAML edit is authorised by this document.

## What is wanted

Further numerical methods beside the constrained solver: the singular value decomposition (full
and truncated), the higher-order SVD (Tucker decomposition, with HOOI refinement) and linear
regression (batch least squares; the online RLS head already exists in Nagare).

## Decisions to take before implementing

1. **Owner.** SVD and HOSVD are dense (multi)linear algebra. Tensor's `hymeko_tensor::linalg` already
   owns the small dense kernels (`solve_in_place`, `damped_least_squares`) and the constrained
   solver, so the default proposal is Tensor; this repository would then hold the model-facing
   uses (projections that ask for a decomposition). The alternative is a `hymeko_decompose` crate
   here, which avoids Tensor's CORE.YAML process but splits linear algebra across two repositories.
2. **Implementation vs dependency.** `hymeko_tensor`'s default build is dependency-free by contract;
   nalgebra (which has SVD) is already behind its `interop` feature. The user's 2026-10-14 rule
   favours established libraries. Options: one-sided Jacobi SVD written here (exact, small, slow
   for large matrices); Golub–Kahan bidiagonalisation with implicit QR (standard, more code);
   nalgebra behind a feature.
3. **Online RLS.** Nagare's `nagare_core::learning::online::EvolventHead` is a forgetting RLS
   readout. A batch regression here must not become a second online one; if a shared RLS kernel is
   wanted outside learning, it moves to Tensor and Nagare consumes it — coordinated with Nagare's
   owner.

## API templates

Shapes only; names and signatures are proposals to review, not code.

```rust
/// Row-major dense matrix view: `rows × columns`, `data.len() == rows * columns`.
pub struct MatrixRef<'a> { pub rows: usize, pub columns: usize, pub data: &'a [f64] }

/// A thin SVD, `A = U Σ Vᵀ`, singular values non-increasing.
pub struct Svd { pub u: Vec<f64>, pub sigma: Vec<f64>, pub vt: Vec<f64>, pub rank: usize }

pub enum DecompositionError { Shape, NonFinite, NotConverged { sweeps: usize, off_diagonal: f64 } }

/// Full thin SVD; `tolerance` bounds the relative off-diagonal mass at convergence.
pub fn svd(a: MatrixRef<'_>, tolerance: f64) -> Result<Svd, DecompositionError>;

/// The best rank-`k` approximation (Eckart–Young), with the discarded energy.
pub fn truncated_svd(a: MatrixRef<'_>, k: usize, tolerance: f64) -> Result<(Svd, f64), DecompositionError>;

/// An order-N dense tensor in row-major (last index fastest) order.
pub struct TensorRef<'a> { pub shape: &'a [usize], pub data: &'a [f64] }

/// Tucker form: core tensor and one orthonormal factor per mode.
pub struct Tucker { pub core: Vec<f64>, pub core_shape: Vec<usize>, pub factors: Vec<Svd> }

/// HOSVD: the leading left singular vectors of each mode-n unfolding, truncated to `ranks`.
pub fn hosvd(t: TensorRef<'_>, ranks: &[usize], tolerance: f64) -> Result<Tucker, DecompositionError>;

/// HOOI: alternate the factors from a HOSVD start until the fit improves by less than `tolerance`.
pub fn hooi(t: TensorRef<'_>, ranks: &[usize], tolerance: f64, sweeps: usize) -> Result<Tucker, DecompositionError>;

/// Least squares `min ‖X β − y‖²` (+ `ridge ‖β‖²`), solved through QR or SVD, never normal equations.
pub struct Regression { pub coefficients: Vec<f64>, pub residual_norm: f64, pub rank: usize }
pub fn least_squares(x: MatrixRef<'_>, y: &[f64], ridge: f64) -> Result<Regression, DecompositionError>;
```

## Correctness oracles (to write first)

- SVD: reconstruction `‖A − U Σ Vᵀ‖_F / ‖A‖_F ≤ c·ε`, orthonormality of `U` and `V`, ordering; known
  spectra (diagonal, rank-1, Hilbert matrices) and rank-deficient cases.
- Truncated SVD: discarded energy equals `Σ_{i>k} σ_i²` (Eckart–Young).
- HOSVD/HOOI: exact recovery of a synthetic Tucker tensor at full multilinear rank; HOOI fit never
  worse than its HOSVD start.
- Regression: agreement with a closed-form two-parameter fit; rank-deficient `X` handled through
  the pseudo-inverse, not a failure; ridge shrinks monotonically.

## Not in scope

Iterative large-scale methods (Lanczos/randomised SVD), CP/PARAFAC decompositions, GPU kernels,
any benchmark campaign.
