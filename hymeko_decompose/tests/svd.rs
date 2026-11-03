//! SVD oracles from the decomposition plan: reconstruction, orthonormality, ordering, known spectra,
//! rank-deficient inputs, Eckart–Young, and typed failures.

mod common;

use common::{Rng, difference, frobenius, matmul, orthonormality_defect, transpose};
use hymeko_decompose::{DecompositionError, MatrixRef, SvdOptions, svd, truncated_svd};

const PATHS: [bool; 2] = [true, false];

fn options(precondition: bool) -> SvdOptions {
    SvdOptions { precondition, ..SvdOptions::default() }
}

fn check_factorisation(a: &[f64], m: usize, n: usize, precondition: bool) -> hymeko_decompose::Svd {
    let s = svd(MatrixRef::new(m, n, a).unwrap(), &options(precondition)).unwrap();
    let k = m.min(n);
    assert_eq!((s.u.len(), s.sigma.len(), s.vt.len()), (m * k, k, k * n));
    let scale = frobenius(a).max(f64::MIN_POSITIVE);
    let bound = 1e-14 * (m + n) as f64;
    assert!(difference(&s.reconstruct(), a) / scale <= bound, "{m}×{n} reconstruction {}", difference(&s.reconstruct(), a) / scale);
    assert!(orthonormality_defect(&s.u, m, k) <= bound, "{m}×{n} U defect {}", orthonormality_defect(&s.u, m, k));
    assert!(orthonormality_defect(&transpose(&s.vt, k, n), n, k) <= bound, "{m}×{n} V defect");
    assert!(s.sigma.windows(2).all(|w| w[0] >= w[1]) && s.sigma.iter().all(|&x| x >= 0.0));
    assert_eq!(s.convergence.preconditioned, precondition && frobenius(a) > 0.0);
    s
}

#[test]
fn random_matrices_of_every_shape_factor_exactly_on_both_paths() {
    let mut rng = Rng::new(20261101);
    for &(m, n) in &[(1, 1), (1, 5), (5, 1), (3, 3), (10, 4), (4, 10), (40, 7), (7, 40), (64, 64), (100, 30)] {
        let a = rng.matrix(m, n);
        for precondition in PATHS {
            check_factorisation(&a, m, n, precondition);
        }
    }
}

#[test]
fn a_diagonal_matrix_has_its_absolute_entries_as_spectrum() {
    let a = [0.0, 0.0, -3.0, 0.0, 7.0, 0.0, 0.5, 0.0, 0.0];
    for precondition in PATHS {
        let s = check_factorisation(&a, 3, 3, precondition);
        assert_eq!(s.sigma, vec![7.0, 3.0, 0.5]);
    }
}

#[test]
fn an_outer_product_has_rank_one_and_the_product_of_norms() {
    let (x, y) = ([1.0, -2.0, 0.5, 4.0], [3.0, 0.25, -1.0]);
    let a: Vec<f64> = x.iter().flat_map(|xi| y.iter().map(move |yj| xi * yj)).collect();
    let expected = frobenius(&x) * frobenius(&y);
    for precondition in PATHS {
        let s = check_factorisation(&a, 4, 3, precondition);
        assert_eq!(s.rank, 1);
        assert!((s.sigma[0] - expected).abs() <= 1e-14 * expected);
        assert!(s.sigma[1..].iter().all(|&v| v <= 1e-15 * expected));
    }
}

#[test]
fn hilbert_five_matches_its_exact_trace_and_determinant() {
    // H₅ is symmetric positive definite: Σσ = trace, ∏σ = det = 1/266716800000 (exact), cond ≈ 4.8e5.
    let h: Vec<f64> = (0..5).flat_map(|i| (0..5).map(move |j| 1.0 / (i + j + 1) as f64)).collect();
    let trace: f64 = (0..5).map(|i| 1.0 / (2 * i + 1) as f64).sum();
    let determinant = 1.0 / 266_716_800_000.0;
    let mut smallest = Vec::new();
    for precondition in PATHS {
        let s = check_factorisation(&h, 5, 5, precondition);
        assert!((s.sigma.iter().sum::<f64>() - trace).abs() <= 1e-14);
        let product: f64 = s.sigma.iter().product();
        assert!((product / determinant - 1.0).abs() <= 1e-10, "det ratio {}", product / determinant);
        smallest.push(s.sigma[4]);
    }
    // Both paths agree on the smallest singular value to high relative accuracy.
    assert!((smallest[0] / smallest[1] - 1.0).abs() <= 1e-10);
}

#[test]
fn rank_deficient_and_zero_matrices_still_return_orthonormal_factors() {
    let mut rng = Rng::new(7);
    let basis = rng.matrix(8, 3);
    let mix = rng.matrix(3, 5);
    let a = matmul(&basis, &mix, 8, 3, 5);
    for precondition in PATHS {
        assert_eq!(check_factorisation(&a, 8, 5, precondition).rank, 3);
        let zero = check_factorisation(&[0.0; 12], 4, 3, precondition);
        assert_eq!((zero.rank, zero.sigma.clone()), (0, vec![0.0; 3]));
    }
}

#[test]
fn scaling_by_extreme_magnitudes_neither_overflows_nor_underflows() {
    let mut rng = Rng::new(11);
    let a = rng.matrix(6, 4);
    let reference = svd(MatrixRef::new(6, 4, &a).unwrap(), &SvdOptions::default()).unwrap();
    for factor in [1e300, 1e-300] {
        let scaled: Vec<f64> = a.iter().map(|v| v * factor).collect();
        let s = svd(MatrixRef::new(6, 4, &scaled).unwrap(), &SvdOptions::default()).unwrap();
        for (x, y) in s.sigma.iter().zip(&reference.sigma) {
            assert!((x / (y * factor) - 1.0).abs() <= 1e-13);
        }
    }
}

#[test]
fn truncation_discards_exactly_the_tail_energy_eckart_young() {
    let mut rng = Rng::new(3);
    let a = rng.matrix(12, 9);
    let full = svd(MatrixRef::new(12, 9, &a).unwrap(), &SvdOptions::default()).unwrap();
    for k in 1..=9 {
        let t = truncated_svd(MatrixRef::new(12, 9, &a).unwrap(), k, &SvdOptions::default()).unwrap();
        let tail: f64 = full.sigma[k..].iter().map(|s| s * s).sum();
        assert!((t.discarded_energy - tail).abs() <= 1e-13 * frobenius(&a).powi(2));
        let residual = difference(&t.svd.reconstruct(), &a).powi(2);
        assert!((residual - tail).abs() <= 1e-12 * frobenius(&a).powi(2), "k={k}");
    }
    assert!(matches!(truncated_svd(MatrixRef::new(12, 9, &a).unwrap(), 0, &SvdOptions::default()), Err(DecompositionError::Shape(_))));
    assert!(matches!(truncated_svd(MatrixRef::new(12, 9, &a).unwrap(), 10, &SvdOptions::default()), Err(DecompositionError::Shape(_))));
}

#[test]
fn invalid_inputs_and_exhausted_budgets_are_typed_refusals() {
    assert!(matches!(MatrixRef::new(0, 3, &[]), Err(DecompositionError::Shape(_))));
    assert!(matches!(MatrixRef::new(2, 2, &[1.0; 3]), Err(DecompositionError::Shape(_))));
    assert!(matches!(MatrixRef::new(1, 2, &[1.0, f64::NAN]), Err(DecompositionError::NonFinite)));
    let a = [1.0, 2.0, 3.0, 4.0];
    let view = MatrixRef::new(2, 2, &a).unwrap();
    for tolerance in [0.0, -1.0, 1.0, f64::NAN] {
        let bad = SvdOptions { tolerance: Some(tolerance), ..SvdOptions::default() };
        assert!(matches!(svd(view, &bad), Err(DecompositionError::InvalidOption(_))));
    }
    for precondition in PATHS {
        let none = SvdOptions { max_sweeps: 0, precondition, ..SvdOptions::default() };
        assert!(matches!(svd(view, &none), Err(DecompositionError::NotConverged { sweeps: 0, .. })));
        // Already orthogonal columns need no sweep at all.
        assert!(svd(MatrixRef::new(2, 2, &[2.0, 0.0, 0.0, -5.0]).unwrap(), &none).is_ok());
    }
}

#[test]
fn the_parallel_sweep_is_bit_identical_for_every_thread_count() {
    let mut rng = Rng::new(31);
    let (m, n) = (300, 256);
    let a = rng.matrix(m, n);
    let view = MatrixRef::new(m, n, &a).unwrap();
    let reference = svd(view, &SvdOptions { threads: 1, ..SvdOptions::default() }).unwrap();
    for threads in [2, 3, 8, 64] {
        assert_eq!(svd(view, &SvdOptions { threads, ..SvdOptions::default() }).unwrap(), reference, "threads={threads}");
    }
    check_factorisation(&a, m, n, true);
    let bad = SvdOptions { threads: 0, ..SvdOptions::default() };
    assert!(matches!(svd(view, &bad), Err(DecompositionError::InvalidOption(_))));
}

#[test]
fn sweep_counts_stay_within_the_measured_budget() {
    // A performance contract on fixed inputs: the O(1) norm updates and the ordering keep random
    // matrices at about 8–11 sweeps; a wrong in-sweep update still converges, but 2–3 sweeps later.
    let mut rng = Rng::new(20261102);
    for &(m, n, budget) in &[(80, 64, 10), (120, 100, 11)] {
        let a = rng.matrix(m, n);
        let s = svd(MatrixRef::new(m, n, &a).unwrap(), &SvdOptions::default()).unwrap();
        assert!(s.convergence.sweeps <= budget, "{m}×{n}: {} sweeps", s.convergence.sweeps);
    }
}
