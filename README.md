# optimize

Optimisation over HyMeKo models.

## Crates

| crate | contents | dependencies |
|---|---|---|
| `hymeko_optimize` | the optimisation projection: a compiled HyMeKo IR read as a solver-neutral constrained problem (`read`, `problem`), the branch-network constraints it builds (`network`, modified nodal analysis), the evidence report (`report`), the strategy seam with a deterministic multistart (`backend`), a browser surface (`web` feature) and the `hymeko-optimize` CLI | Core; Tensor's `hymeko_tensor::{dual, optimize}` (forward-mode duals, augmented-Lagrangian solver); Core's in-memory `MemProvider` (web); transformation's `hymeko_query` and kawakern's `kawakern_dynamics` (tests) |
| `hymeko_decompose` | dense decompositions: the thin SVD by QR-preconditioned one-sided Jacobi (`svd`, `truncated_svd`), Tucker by HOSVD and HOOI (`hosvd`, `hooi`), batch least squares and ridge through the same SVD (`least_squares`) | none |

The numerical solver is Tensor's: `hymeko_optimize::backend::AugmentedLagrangian` is
`hymeko_tensor::optimize::AugmentedLagrangian` reached through the `Constrained` view the
projection's `Problem` implements.

## Decompositions

`hymeko_decompose` has its own kernel (one-sided Jacobi, no dependency):

- **SVD.** Pivoted Householder QR (`A P = Q R`, `Q` kept as reflectors), then one-sided Jacobi on
  `Rᵀ`; O(m n²) once plus O(n³) per sweep, with Jacobi's high relative accuracy for small singular
  values. Squared column norms update in O(1) per rotation; sweeps follow a block round-robin
  ordering (blocks of 16) whose steps run on worker threads from 256 columns up. The ordering
  depends only on `n`, so results are bit-identical for every thread count.
- **HOSVD/HOOI.** Mode factors from the QR of the transposed unfolding (no Gram matrix, no `Q`),
  blocked mode products contracted strongest-compression first, independent modes in parallel.
- **Least squares.** `β = V f(Σ) Uᵀ y` without forming `U`; minimum-norm on rank-deficient or
  underdetermined data; ridge by filter factors.

Every result carries its Jacobi convergence evidence; every failure is a typed
`DecompositionError`. Not in scope: Lanczos/randomised SVD, CP/PARAFAC, GPU kernels.

Opt-in measurement (release build): `cargo run --release --example measure -p hymeko_decompose`.

## Checks

```
cargo test --locked -j 2
cargo check --locked -j 2 --features web
```

The tests read the optimisation models from Core's data library at `../../core/data/optimization`
relative to the crate, so clone [core](https://github.com/hymeko-framework/core) beside this
repository as `core`; without it those tests skip.

## Provenance

`hymeko_optimize` moved unchanged from the transformation repository at
209dee965d75ba33559e21b689ce72714a0669f2 (2026-11-01); its history is there. Its solver had moved to
Tensor one commit earlier (transformation 73976b2, tensor 6769dab).
