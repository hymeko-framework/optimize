# optimize

Optimisation over HyMeKo models.

## Crates

| crate | contents | dependencies |
|---|---|---|
| `hymeko_optimize` | the optimisation projection: a compiled HyMeKo IR read as a solver-neutral constrained problem (`read`, `problem`), the branch-network constraints it builds (`network`, modified nodal analysis), the evidence report (`report`), the strategy seam with a deterministic multistart (`backend`), a browser surface (`web` feature) and the `hymeko-optimize` CLI | Core; Tensor's `hymeko_tensor::{dual, optimize}` (forward-mode duals, augmented-Lagrangian solver); transformation's `hymeko_sources` (web) and `hymeko_query` (tests) |

The numerical solver is Tensor's: `hymeko_optimize::backend::AugmentedLagrangian` is
`hymeko_tensor::optimize::AugmentedLagrangian` reached through the `Constrained` view the
projection's `Problem` implements.

## Planned

`docs/plans/2026-11-01-decompositions/` holds the plan and API templates for further methods
(SVD, HOSVD/Tucker, linear regression). They are plans only: nothing is implemented here yet, and
where each method lives (here or Tensor's `linalg`) is decided in that plan.

## Checks

```
cargo test --offline -j 2
cargo check --offline -j 2 --features web
```

The tests read the optimisation models from Core's data library at `../../core/data/optimization`
relative to the crate (a sibling `core` checkout); without it they skip.

## Provenance

`hymeko_optimize` moved unchanged from the transformation repository at
209dee965d75ba33559e21b689ce72714a0669f2 (2026-11-01); its history is there. Its solver had moved to
Tensor one commit earlier (transformation 73976b2, tensor 6769dab).
