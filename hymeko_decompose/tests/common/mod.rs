//! Deterministic test fixtures: a seeded generator and dense row-major helpers (no dev-dependencies).
#![allow(dead_code)]

/// SplitMix64: reproducible, seedable, good enough for test matrices.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    pub fn matrix(&mut self, rows: usize, cols: usize) -> Vec<f64> {
        (0..rows * cols).map(|_| self.uniform()).collect()
    }

    /// `rows × cols` (rows ≥ cols) with orthonormal columns, by modified Gram–Schmidt applied twice.
    pub fn orthonormal(&mut self, rows: usize, cols: usize) -> Vec<f64> {
        assert!(rows >= cols);
        let mut q = self.matrix(rows, cols);
        for j in 0..cols {
            for _ in 0..2 {
                for k in 0..j {
                    let p: f64 = (0..rows).map(|i| q[i * cols + j] * q[i * cols + k]).sum();
                    for i in 0..rows {
                        q[i * cols + j] -= p * q[i * cols + k];
                    }
                }
            }
            let n: f64 = (0..rows).map(|i| q[i * cols + j].powi(2)).sum::<f64>().sqrt();
            for i in 0..rows {
                q[i * cols + j] /= n;
            }
        }
        q
    }
}

/// `A (m×k) · B (k×n)`, row-major.
pub fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let w = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += w * b[p * n + j];
            }
        }
    }
    c
}

pub fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

pub fn frobenius(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn difference(a: &[f64], b: &[f64]) -> f64 {
    frobenius(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>())
}

/// `max |QᵀQ − I|` for a row-major `rows × cols` matrix with supposedly orthonormal columns.
pub fn orthonormality_defect(q: &[f64], rows: usize, cols: usize) -> f64 {
    let qtq = matmul(&transpose(q, rows, cols), q, cols, rows, cols);
    let mut worst = 0.0_f64;
    for i in 0..cols {
        for j in 0..cols {
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((qtq[i * cols + j] - target).abs());
        }
    }
    worst
}
