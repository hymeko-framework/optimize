//! HOSVD/HOOI oracles: exact recovery at full multilinear rank, the HOSVD error bound, HOOI never
//! worse than its start, Eckart–Young for order 2, all-orthogonality of the core, determinism.

mod common;

use common::{Rng, difference, frobenius, orthonormality_defect};
use hymeko_decompose::{DecompositionError, MatrixRef, SvdOptions, TensorRef, TuckerOptions, hooi, hosvd, truncated_svd};

/// `G ×ₙ Uₙ` for row-major data and a row-major `rows × rank` factor.
fn mode_multiply(data: &[f64], shape: &[usize], mode: usize, u: &[f64], rows: usize) -> (Vec<f64>, Vec<usize>) {
    let outer: usize = shape[..mode].iter().product();
    let inner: usize = shape[mode + 1..].iter().product();
    let rank = shape[mode];
    let mut out = vec![0.0; outer * rows * inner];
    for l in 0..outer {
        for i in 0..rows {
            for r in 0..rank {
                let w = u[i * rank + r];
                for t in 0..inner {
                    out[(l * rows + i) * inner + t] += w * data[(l * rank + r) * inner + t];
                }
            }
        }
    }
    let mut new_shape = shape.to_vec();
    new_shape[mode] = rows;
    (out, new_shape)
}

/// A tensor of exact multilinear rank `ranks`: random core times random orthonormal factors.
fn synthetic(rng: &mut Rng, shape: &[usize], ranks: &[usize]) -> Vec<f64> {
    let mut data = rng.matrix(1, ranks.iter().product());
    let mut current = ranks.to_vec();
    for mode in 0..shape.len() {
        let u = rng.orthonormal(shape[mode], ranks[mode]);
        (data, current) = mode_multiply(&data, &current, mode, &u, shape[mode]);
    }
    assert_eq!(current, shape);
    data
}

#[test]
fn hosvd_recovers_a_tensor_of_exact_multilinear_rank() {
    let mut rng = Rng::new(42);
    for (shape, ranks) in [(vec![6, 5, 7], vec![3, 2, 4]), (vec![4, 3, 5, 2], vec![2, 3, 2, 2]), (vec![6, 2], vec![6, 2]), (vec![5], vec![3])] {
        let t = synthetic(&mut rng, &shape, &ranks);
        let tucker = hosvd(TensorRef { shape: &shape, data: &t }, &ranks, &TuckerOptions::default()).unwrap();
        assert!(difference(&tucker.reconstruct(), &t) <= 1e-12 * frobenius(&t), "{shape:?}");
        for (factor, (&dim, &rank)) in tucker.factors.iter().zip(shape.iter().zip(&ranks)) {
            assert_eq!((factor.rows, factor.rank), (dim, rank));
            assert!(orthonormality_defect(&factor.matrix, dim, rank) <= 1e-13);
            assert!(factor.spectrum.iter().skip(rank).all(|&s| s <= 1e-12 * frobenius(&t)));
        }
    }
}

#[test]
fn truncated_hosvd_respects_its_a_priori_bound_and_reports_its_error() {
    let mut rng = Rng::new(5);
    let shape = [6, 5, 4];
    let t = rng.matrix(1, 120);
    let ranks = [3, 3, 2];
    let tucker = hosvd(TensorRef { shape: &shape, data: &t }, &ranks, &TuckerOptions::default()).unwrap();
    let norm = frobenius(&t);
    let explicit = difference(&tucker.reconstruct(), &t) / norm;
    assert!(explicit <= tucker.error_bound + 1e-12);
    assert!((tucker.relative_error - explicit).abs() <= 1e-7);
}

#[test]
fn hooi_never_does_worse_than_its_hosvd_start_and_its_history_is_monotone() {
    let mut rng = Rng::new(9);
    let shape = [7, 6, 5];
    let t = rng.matrix(1, 210);
    let ranks = [2, 3, 2];
    let view = TensorRef { shape: &shape, data: &t };
    let start = hosvd(view, &ranks, &TuckerOptions::default()).unwrap();
    let refined = hooi(view, &ranks, &TuckerOptions::default()).unwrap();
    assert!(refined.history.len() >= 2);
    assert!(refined.history.windows(2).all(|w| w[1] <= w[0] + 1e-12), "{:?}", refined.history);
    let error = |tucker: &hymeko_decompose::Tucker| difference(&tucker.reconstruct(), &t);
    assert!(error(&refined) <= error(&start) + 1e-12);
    // The a-priori bound is HOSVD's and survives HOOI unchanged; both errors respect it.
    assert_eq!(refined.error_bound, start.error_bound);
    let norm = frobenius(&t);
    assert!(error(&start) / norm <= start.error_bound + 1e-12 && error(&refined) / norm <= refined.error_bound + 1e-12);
}

#[test]
fn order_two_hosvd_is_the_eckart_young_truncation() {
    let mut rng = Rng::new(13);
    let a = rng.matrix(9, 7);
    for k in 1..=7 {
        let tucker = hosvd(TensorRef { shape: &[9, 7], data: &a }, &[k, k], &TuckerOptions::default()).unwrap();
        let best = truncated_svd(MatrixRef::new(9, 7, &a).unwrap(), k, &SvdOptions::default()).unwrap();
        let tucker_error = difference(&tucker.reconstruct(), &a).powi(2);
        assert!((tucker_error - best.discarded_energy).abs() <= 1e-12 * frobenius(&a).powi(2), "k={k}");
    }
}

#[test]
fn the_full_rank_hosvd_core_is_all_orthogonal_and_ordered() {
    let mut rng = Rng::new(17);
    let shape = [4, 3, 5];
    let t = rng.matrix(1, 60);
    let tucker = hosvd(TensorRef { shape: &shape, data: &t }, &shape, &TuckerOptions::default()).unwrap();
    for mode in 0..3 {
        let outer: usize = shape[..mode].iter().product();
        let inner: usize = shape[mode + 1..].iter().product();
        let slice = |a: usize| -> Vec<f64> {
            (0..outer).flat_map(|l| (0..inner).map(move |r| (l, r))).map(|(l, r)| tucker.core[(l * shape[mode] + a) * inner + r]).collect()
        };
        for a in 0..shape[mode] {
            let sa = slice(a);
            assert!((frobenius(&sa) - tucker.factors[mode].spectrum[a]).abs() <= 1e-12 * frobenius(&t));
            for b in a + 1..shape[mode] {
                let inner_product: f64 = sa.iter().zip(slice(b)).map(|(x, y)| x * y).sum();
                assert!(inner_product.abs() <= 1e-12 * frobenius(&t).powi(2));
            }
        }
    }
}

#[test]
fn parallel_modes_are_bit_identical_to_sequential() {
    let mut rng = Rng::new(23);
    let shape = [8, 7, 6];
    let t = rng.matrix(1, 336);
    let view = TensorRef { shape: &shape, data: &t };
    let parallel = hooi(view, &[3, 4, 2], &TuckerOptions { parallel: true, ..TuckerOptions::default() }).unwrap();
    let sequential = hooi(view, &[3, 4, 2], &TuckerOptions { parallel: false, ..TuckerOptions::default() }).unwrap();
    assert_eq!(parallel, sequential);
}

#[test]
fn inconsistent_shapes_and_ranks_are_refused() {
    let t = [1.0; 24];
    let view = TensorRef { shape: &[2, 3, 4], data: &t };
    let options = TuckerOptions::default();
    for ranks in [vec![1, 1], vec![0, 1, 1], vec![3, 1, 1]] {
        assert!(matches!(hosvd(view, &ranks, &options), Err(DecompositionError::Shape(_))));
    }
    assert!(matches!(hosvd(TensorRef { shape: &[2, 3], data: &t }, &[1, 1], &options), Err(DecompositionError::Shape(_))));
    let mut bad = t;
    bad[3] = f64::INFINITY;
    assert!(matches!(hosvd(TensorRef { shape: &[2, 3, 4], data: &bad }, &[1, 1, 1], &options), Err(DecompositionError::NonFinite)));
    let negative = TuckerOptions { tolerance: -1.0, ..options };
    assert!(matches!(hooi(view, &[1, 1, 1], &negative), Err(DecompositionError::InvalidOption(_))));
}
