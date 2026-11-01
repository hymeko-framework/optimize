//! Least-squares oracles: closed-form line fit, orthogonal residual, minimum-norm solutions on
//! rank-deficient and underdetermined data, ridge against its closed form and its monotone shrinkage.

mod common;

use common::{Rng, frobenius, matmul, transpose};
use hymeko_decompose::{DecompositionError, LeastSquaresOptions, MatrixRef, least_squares};

fn solve(x: &[f64], y: &[f64], m: usize, n: usize, ridge: f64) -> hymeko_decompose::Regression {
    least_squares(MatrixRef::new(m, n, x).unwrap(), y, &LeastSquaresOptions { ridge, ..LeastSquaresOptions::default() }).unwrap()
}

/// Gaussian elimination with partial pivoting, for small reference systems only.
fn reference_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs())).unwrap();
        for j in 0..n {
            a.swap(k * n + j, p * n + j);
        }
        b.swap(k, p);
        for i in k + 1..n {
            let f = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for k in (0..n).rev() {
        x[k] = (b[k] - (k + 1..n).map(|j| a[k * n + j] * x[j]).sum::<f64>()) / a[k * n + k];
    }
    x
}

#[test]
fn a_line_fit_matches_the_closed_form() {
    let mut rng = Rng::new(1);
    let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 1.5 - 0.75 * x + 0.1 * rng.uniform()).collect();
    let design: Vec<f64> = xs.iter().flat_map(|&x| [1.0, x]).collect();
    let fit = solve(&design, &ys, 20, 2, 0.0);
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let slope = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
    let intercept = my - slope * mx;
    assert!((fit.coefficients[0] - intercept).abs() <= 1e-13 && (fit.coefficients[1] - slope).abs() <= 1e-13);
    assert_eq!(fit.rank, 2);
}

#[test]
fn a_consistent_system_is_solved_exactly_and_residuals_are_orthogonal_to_the_columns() {
    let mut rng = Rng::new(2);
    let (m, n) = (30, 6);
    let x = rng.matrix(m, n);
    let beta = rng.matrix(n, 1);
    let exact = matmul(&x, &beta, m, n, 1);
    let fit = solve(&x, &exact, m, n, 0.0);
    assert!(fit.residual_norm <= 1e-13 * frobenius(&exact));
    assert!(fit.coefficients.iter().zip(&beta).all(|(a, b)| (a - b).abs() <= 1e-12));
    let noisy: Vec<f64> = exact.iter().map(|v| v + rng.uniform()).collect();
    let fit = solve(&x, &noisy, m, n, 0.0);
    let residual: Vec<f64> = noisy.iter().zip(matmul(&x, &fit.coefficients, m, n, 1)).map(|(y, p)| y - p).collect();
    let normal = matmul(&transpose(&x, m, n), &residual, n, m, 1);
    assert!(frobenius(&normal) <= 1e-12 * frobenius(&noisy) * frobenius(&x));
    assert!((fit.residual_norm - frobenius(&residual)).abs() <= 1e-13);
}

#[test]
fn a_duplicated_feature_gets_the_minimum_norm_split() {
    let mut rng = Rng::new(3);
    let m = 15;
    let base = rng.matrix(m, 2);
    // Columns: a, b, a (duplicate) — rank 2.
    let x: Vec<f64> = (0..m).flat_map(|i| [base[2 * i], base[2 * i + 1], base[2 * i]]).collect();
    let y: Vec<f64> = (0..m).map(|i| 2.0 * base[2 * i] - base[2 * i + 1]).collect();
    let fit = solve(&x, &y, m, 3, 0.0);
    assert_eq!(fit.rank, 2);
    assert!((fit.coefficients[0] - 1.0).abs() <= 1e-12 && (fit.coefficients[2] - 1.0).abs() <= 1e-12);
    assert!((fit.coefficients[1] + 1.0).abs() <= 1e-12);
}

#[test]
fn an_underdetermined_system_gets_the_exact_minimum_norm_solution() {
    let mut rng = Rng::new(4);
    let (m, n) = (4, 9);
    let x = rng.matrix(m, n);
    let y = rng.matrix(m, 1);
    let fit = solve(&x, &y, m, n, 0.0);
    assert!(fit.residual_norm <= 1e-13);
    // Minimum norm ⇔ β = Xᵀ z: β is orthogonal to the null space; compare with z = (X Xᵀ)⁻¹ y.
    let gram = matmul(&x, &transpose(&x, m, n), m, n, m);
    let z = reference_solve(gram, y.clone(), m);
    let expected = matmul(&transpose(&x, m, n), &z, n, m, 1);
    assert!(fit.coefficients.iter().zip(&expected).all(|(a, b)| (a - b).abs() <= 1e-11));
}

#[test]
fn ridge_matches_its_closed_form_and_shrinks_monotonically() {
    let mut rng = Rng::new(5);
    let (m, n) = (25, 4);
    let x = rng.matrix(m, n);
    let y = rng.matrix(m, 1);
    let mut previous = f64::INFINITY;
    for ridge in [0.0, 0.1, 1.0, 10.0, 100.0] {
        let fit = solve(&x, &y, m, n, ridge);
        let mut gram = matmul(&transpose(&x, m, n), &x, n, m, n);
        for i in 0..n {
            gram[i * n + i] += ridge;
        }
        let expected = reference_solve(gram, matmul(&transpose(&x, m, n), &y, n, m, 1), n);
        assert!(fit.coefficients.iter().zip(&expected).all(|(a, b)| (a - b).abs() <= 1e-11), "ridge {ridge}");
        let norm = frobenius(&fit.coefficients);
        assert!(norm < previous);
        previous = norm;
    }
}

#[test]
fn invalid_regression_inputs_are_refused() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let view = MatrixRef::new(2, 2, &x).unwrap();
    let options = LeastSquaresOptions::default();
    assert!(matches!(least_squares(view, &[1.0], &options), Err(DecompositionError::Shape(_))));
    assert!(matches!(least_squares(view, &[1.0, f64::NAN], &options), Err(DecompositionError::NonFinite)));
    for ridge in [-1.0, f64::INFINITY] {
        assert!(matches!(least_squares(view, &[1.0, 2.0], &LeastSquaresOptions { ridge, ..options }), Err(DecompositionError::InvalidOption(_))));
    }
}
