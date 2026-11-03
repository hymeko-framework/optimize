//! Matrix views and the dense kernels every decomposition runs on.
//!
//! The public surface is row-major ([`MatrixRef`]), matching how callers usually hold data. The
//! numerical core works column-major ([`ColMajor`]): one-sided Jacobi and Householder QR touch whole
//! columns, so a column must be one contiguous slice for the inner loops to vectorise.

use crate::error::DecompositionError;

/// Row-major dense matrix view: `rows × columns`, `data.len() == rows * columns`.
#[derive(Clone, Copy, Debug)]
pub struct MatrixRef<'a> {
    pub rows: usize,
    pub columns: usize,
    pub data: &'a [f64],
}

impl<'a> MatrixRef<'a> {
    /// A checked view; empty dimensions are refused because no decomposition of them is meaningful.
    pub fn new(rows: usize, columns: usize, data: &'a [f64]) -> Result<Self, DecompositionError> {
        let view = Self { rows, columns, data };
        view.validate()?;
        Ok(view)
    }

    pub(crate) fn validate(&self) -> Result<(), DecompositionError> {
        if self.rows == 0 || self.columns == 0 {
            return Err(DecompositionError::Shape(format!("empty matrix {}×{}", self.rows, self.columns)));
        }
        let expected = self.rows.checked_mul(self.columns).ok_or_else(|| DecompositionError::Shape("dimension product overflows".into()))?;
        if self.data.len() != expected {
            return Err(DecompositionError::Shape(format!(
                "{}×{} needs {expected} values, got {}",
                self.rows,
                self.columns,
                self.data.len()
            )));
        }
        if self.data.iter().any(|value| !value.is_finite()) {
            return Err(DecompositionError::NonFinite);
        }
        Ok(())
    }
}

/// Column-major working storage: column `j` is `data[j * rows .. (j + 1) * rows]`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ColMajor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl ColMajor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for j in 0..n {
            m.data[j * n + j] = 1.0;
        }
        m
    }

    /// The column-major copy of a row-major view.
    pub fn from_row_major(view: MatrixRef<'_>) -> Self {
        let mut m = Self::zeros(view.rows, view.columns);
        for (i, row) in view.data.chunks_exact(view.columns).enumerate() {
            for (j, &value) in row.iter().enumerate() {
                m.data[j * view.rows + i] = value;
            }
        }
        m
    }

    /// The column-major copy of the transpose of a row-major view: rows of `view` become columns.
    pub fn transpose_of_row_major(view: MatrixRef<'_>) -> Self {
        Self { rows: view.columns, cols: view.rows, data: view.data.to_vec() }
    }

    /// Row-major copy of this matrix.
    pub fn to_row_major(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.rows * self.cols];
        for j in 0..self.cols {
            for (i, &value) in self.col(j).iter().enumerate() {
                out[i * self.cols + j] = value;
            }
        }
        out
    }

    /// Row-major copy of the transpose (`cols × rows`): column `j` becomes row `j`, contiguous.
    pub fn transpose_to_row_major(&self) -> Vec<f64> {
        self.data.clone()
    }

    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// Two distinct columns, mutably; `p < q` is required.
    #[inline]
    pub fn col_pair_mut(&mut self, p: usize, q: usize) -> (&mut [f64], &mut [f64]) {
        debug_assert!(p < q && q < self.cols);
        let rows = self.rows;
        let (head, tail) = self.data.split_at_mut(q * rows);
        (&mut head[p * rows..(p + 1) * rows], &mut tail[..rows])
    }

    pub fn swap_cols(&mut self, p: usize, q: usize) {
        if p == q {
            return;
        }
        let (a, b) = if p < q { self.col_pair_mut(p, q) } else { self.col_pair_mut(q, p) };
        a.swap_with_slice(b);
    }

    /// The leading `cols` columns, as a new matrix.
    pub fn leading_cols(&self, cols: usize) -> Self {
        Self { rows: self.rows, cols, data: self.data[..self.rows * cols].to_vec() }
    }

    /// Largest absolute entry (0 for a zero matrix).
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    pub fn scale(&mut self, factor: f64) {
        self.data.iter_mut().for_each(|value| *value *= factor);
    }
}

/// Dot product with four independent accumulators: breaks the add dependency chain so the loop
/// vectorises and pipelines, and reduces the rounding-error growth of a single running sum.
#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0_f64; 4];
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let (tail_a, tail_b) = (chunks_a.remainder(), chunks_b.remainder());
    for (x, y) in chunks_a.zip(chunks_b) {
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in tail_a.iter().zip(tail_b) {
        sum += x * y;
    }
    sum
}

/// `y ← y + alpha · x`.
#[inline]
pub(crate) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// The plane rotation `(x, y) ← (c·x − s·y, s·x + c·y)`, applied elementwise.
#[inline]
pub(crate) fn rotate(x: &mut [f64], y: &mut [f64], c: f64, s: f64) {
    debug_assert_eq!(x.len(), y.len());
    for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
        let (a, b) = (*xi, *yi);
        *xi = c * a - s * b;
        *yi = s * a + c * b;
    }
}

/// Euclidean norm without overflow or harmful underflow: scaled by the largest magnitude.
pub(crate) fn norm2(x: &[f64]) -> f64 {
    let scale = x.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    let inverse = 1.0 / scale;
    let sum: f64 = x.iter().map(|value| (value * inverse) * (value * inverse)).sum();
    scale * sum.sqrt()
}

/// Complete the leading `filled` orthonormal columns of `m` (square or tall) to `m.cols` orthonormal
/// columns: Gram–Schmidt, applied twice, of the canonical vector with the largest residual.
pub(crate) fn complete_orthonormal(m: &mut ColMajor, filled: usize) {
    let rows = m.rows;
    for j in filled..m.cols {
        let mut best: Option<(f64, Vec<f64>)> = None;
        for e in 0..rows {
            let mut v = vec![0.0; rows];
            v[e] = 1.0;
            for _ in 0..2 {
                for k in 0..j {
                    let projection = dot(m.col(k), &v);
                    axpy(-projection, m.col(k), &mut v);
                }
            }
            let residual = norm2(&v);
            if best.as_ref().is_none_or(|(norm, _)| residual > *norm) {
                best = Some((residual, v));
            }
            if residual > 0.5 {
                break;
            }
        }
        let (residual, mut v) = best.expect("rows > 0");
        v.iter_mut().for_each(|value| *value /= residual);
        m.col_mut(j).copy_from_slice(&v);
    }
}
