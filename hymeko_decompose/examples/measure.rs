//! Opt-in measurement (not a test): Jacobi sweeps and wall time with and without QR preconditioning,
//! and HOSVD/HOOI with sequential versus parallel modes.
//!
//! cargo run --release --example measure -p hymeko_decompose

use std::time::Instant;

use hymeko_decompose::{MatrixRef, SvdOptions, TensorRef, TuckerOptions, hooi, hosvd, svd};

fn matrix(seed: u64, len: usize) -> Vec<f64> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn time<T>(repeats: usize, mut run: impl FnMut() -> T) -> (f64, T) {
    let start = Instant::now();
    let mut last = run();
    for _ in 1..repeats {
        last = run();
    }
    (start.elapsed().as_secs_f64() / repeats as f64, last)
}

fn main() {
    println!("{:>11} {:>14} {:>7} {:>11} {:>7} {:>8}", "shape", "path", "sweeps", "seconds", "speedup", "σ_min rel");
    for &(m, n) in &[(50, 50), (200, 50), (1000, 50), (2000, 100), (300, 300)] {
        let a = matrix(m as u64 * 31 + n as u64, m * n);
        let view = MatrixRef::new(m, n, &a).unwrap();
        let repeats = if m * n * n > 50_000_000 { 1 } else { 3 };
        let (plain_s, plain) = time(repeats, || svd(view, &SvdOptions { precondition: false, ..SvdOptions::default() }).unwrap());
        let (pre_s, pre) = time(repeats, || svd(view, &SvdOptions::default()).unwrap());
        let agreement = (pre.sigma[n - 1] / plain.sigma[n - 1] - 1.0).abs();
        println!("{:>11} {:>14} {:>7} {:>11.5} {:>7} {:>8}", format!("{m}×{n}"), "plain Jacobi", plain.convergence.sweeps, plain_s, "", "");
        println!("{:>11} {:>14} {:>7} {:>11.5} {:>6.1}× {:>8.1e}", "", "QR + Jacobi", pre.convergence.sweeps, pre_s, plain_s / pre_s, agreement);
    }
    println!();
    for n in [128usize, 300, 500] {
        let a = matrix(n as u64, n * n);
        let view = MatrixRef::new(n, n, &a).unwrap();
        let (serial_s, serial) = time(1, || svd(view, &SvdOptions { threads: 1, ..SvdOptions::default() }).unwrap());
        let (par_s, par) = time(1, || svd(view, &SvdOptions::default()).unwrap());
        assert_eq!(serial, par);
        println!("{n}×{n}: serial {serial_s:.4} s, {} threads {par_s:.4} s ({:.1}×), {} sweeps, bit-identical", SvdOptions::default().threads, serial_s / par_s, par.convergence.sweeps);
    }
    println!();
    for (shape, ranks) in [([40usize, 40, 40], [8usize, 8, 8]), ([120, 60, 30], [10, 10, 5]), ([60, 60, 60], [20, 20, 20])] {
        let t = matrix(shape.iter().product::<usize>() as u64, shape.iter().product());
        let view = TensorRef { shape: &shape, data: &t };
        let (seq_s, seq) = time(2, || hosvd(view, &ranks, &TuckerOptions { parallel: false, ..TuckerOptions::default() }).unwrap());
        let (par_s, par) = time(2, || hosvd(view, &ranks, &TuckerOptions::default()).unwrap());
        assert_eq!(seq, par);
        let (hooi_s, refined) = time(1, || hooi(view, &ranks, &TuckerOptions::default()).unwrap());
        println!(
            "HOSVD {:?} ranks {:?}: sequential {:.4} s, parallel {:.4} s ({:.1}×), rel. error {:.4}; HOOI {:.3} s, {} sweeps, rel. error {:.4}",
            shape, ranks, seq_s, par_s, seq_s / par_s, seq.relative_error, hooi_s, refined.history.len() - 1, refined.relative_error
        );
    }
}
