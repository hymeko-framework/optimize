//! One-sided (Hestenes) Jacobi: orthogonalise the columns of `X` by plane rotations, `X V = W`.
//!
//! At convergence `W = U Σ` has mutually orthogonal columns, so `σ_j = ‖w_j‖`, `u_j = w_j / σ_j`
//! and `X = U Σ Vᵀ`. The method is chosen for accuracy: it computes small singular values to high
//! *relative* accuracy where bidiagonalisation only reaches absolute accuracy (Demmel & Veselić,
//! SIMAX 13, 1992). Speed comes from three places:
//!
//! - the caller runs it on the triangular factor of a pivoted QR (see [`crate::svd`]), so a sweep
//!   costs O(n³) rather than O(m n²);
//! - squared column norms are recomputed exactly once per sweep and updated in O(1) after each
//!   rotation (`α' = α − tγ`, `β' = β + tγ`); a pair already below the tolerance costs one dot;
//! - each sweep follows a block round-robin ordering: columns form blocks of [`BLOCK`]; a sweep is
//!   one step of within-block pairs followed by the Brent–Luk tournament over blocks, each step a
//!   set of disjoint block pairs. Steps are coarse (one barrier per block step, both blocks' columns
//!   cache-resident) and their tasks run on worker threads. The blocking depends only on `n`, so the
//!   serial and parallel paths perform identical arithmetic and the result is bit-identical for
//!   every thread count.
//!
//! Columns are reordered by decreasing norm at each sweep start (de Rijk's pivoting).

use std::sync::{Barrier, Mutex};

use crate::error::DecompositionError;
use crate::matrix::{ColMajor, dot, norm2, rotate};

/// Columns per block of the sweep ordering.
pub(crate) const BLOCK: usize = 16;
/// Below this many columns a sweep is too short for threads to pay off (measured, see the example).
pub(crate) const PARALLEL_MIN_COLUMNS: usize = 256;

/// The Jacobi result: `w` holds `W = U Σ` (columns sorted by decreasing norm), `v` the rotations.
pub(crate) struct JacobiResult {
    pub w: ColMajor,
    pub v: ColMajor,
    pub sigma: Vec<f64>,
    pub sweeps: usize,
    pub off_diagonal: f64,
}

/// Orthogonalise the columns of `x` (`rows ≥ cols`) until every pair satisfies
/// `|xₚ·x_q| ≤ tolerance · ‖xₚ‖‖x_q‖`, or fail after `max_sweeps`.
pub(crate) fn orthogonalise(mut x: ColMajor, tolerance: f64, max_sweeps: usize, threads: usize) -> Result<JacobiResult, DecompositionError> {
    let n = x.cols;
    let mut v = ColMajor::identity(n);
    let mut norms2 = vec![0.0; n];
    let schedule = block_schedule(n);
    let width = schedule.iter().map(Vec::len).max().unwrap_or(1);
    let workers = if n >= PARALLEL_MIN_COLUMNS { threads.clamp(1, width) } else { 1 };
    let mut sweeps = 0;
    let mut off_diagonal: f64;
    let converged = loop {
        for (j, slot) in norms2.iter_mut().enumerate() {
            *slot = dot(x.col(j), x.col(j));
        }
        sort_by_norm(&mut x, &mut v, &mut norms2);
        if sweeps == max_sweeps {
            // One last measuring pass decides convergence without rotating.
            off_diagonal = coupling(&x, &norms2);
            break off_diagonal <= tolerance;
        }
        sweeps += 1;
        let (worst, rotations) = sweep(&mut x, &mut v, &mut norms2, &schedule, tolerance, workers);
        off_diagonal = worst;
        if rotations == 0 {
            break true;
        }
    };
    if !converged {
        return Err(DecompositionError::NotConverged { sweeps, off_diagonal });
    }
    let sigma: Vec<f64> = (0..n).map(|j| norm2(x.col(j))).collect();
    let mut result = JacobiResult { w: x, v, sigma, sweeps, off_diagonal };
    sort_result(&mut result);
    Ok(result)
}

/// The tournament schedule for `n` columns: `n − 1` (or `n` when odd) steps of disjoint `(p, q)`
/// pairs with `p < q`, every unordered pair exactly once per sweep.
fn round_robin(n: usize) -> Vec<Vec<(usize, usize)>> {
    if n < 2 {
        return Vec::new();
    }
    let players = n + n % 2; // a dummy player absorbs the bye when n is odd
    let rounds = players - 1;
    let seat = |t: usize| 1 + t % rounds;
    (0..rounds)
        .map(|round| {
            let mut pairs = Vec::with_capacity(players / 2);
            let mut push = |a: usize, b: usize| {
                if a < n && b < n {
                    pairs.push((a.min(b), a.max(b)));
                }
            };
            push(0, seat(round));
            for k in 1..players / 2 {
                push(seat(round + k), seat(round + rounds - k));
            }
            pairs
        })
        .collect()
}

/// One pair's rotation; returns the pair's relative coupling and whether it rotated. Identical
/// arithmetic on the serial and parallel paths.
#[inline]
fn rotate_pair(xp: &mut [f64], xq: &mut [f64], vp: &mut [f64], vq: &mut [f64], np: &mut f64, nq: &mut f64, tolerance: f64) -> (f64, bool) {
    let (alpha, beta) = (*np, *nq);
    if alpha == 0.0 || beta == 0.0 {
        return (0.0, false);
    }
    let gamma = dot(xp, xq);
    let relative = gamma.abs() / (alpha.sqrt() * beta.sqrt());
    if relative <= tolerance {
        return (relative, false);
    }
    let zeta = (beta - alpha) / (2.0 * gamma);
    // The smaller root of t² + 2ζt − 1 = 0; for huge ζ it tends to 1/(2ζ) without overflow.
    let t = zeta.signum() / (zeta.abs() + zeta.hypot(1.0));
    let c = 1.0 / t.hypot(1.0);
    let s = c * t;
    rotate(xp, xq, c, s);
    rotate(vp, vq, c, s);
    *np = (alpha - t * gamma).max(0.0);
    *nq = beta + t * gamma;
    (relative, true)
}

/// One unit of a sweep step: the pairs inside a block, or every pair across two blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Task {
    Within(usize),
    Across(usize, usize),
}

/// Step 0: every block's internal pairs (disjoint blocks). Then the tournament over blocks.
fn block_schedule(n: usize) -> Vec<Vec<Task>> {
    let blocks = n.div_ceil(BLOCK);
    let mut schedule = vec![(0..blocks).map(Task::Within).collect::<Vec<_>>()];
    schedule.extend(round_robin(blocks).into_iter().map(|step| step.into_iter().map(|(a, b)| Task::Across(a, b)).collect()));
    schedule
}

/// A column of `X`, the matching column of `V` and its squared norm.
struct Column<'a> {
    x: &'a mut [f64],
    v: &'a mut [f64],
    norm2: f64,
}

fn pair(a: &mut Column<'_>, b: &mut Column<'_>, tolerance: f64, worst: &mut f64, rotations: &mut usize) {
    let (relative, rotated) = rotate_pair(a.x, b.x, a.v, b.v, &mut a.norm2, &mut b.norm2, tolerance);
    *worst = worst.max(relative);
    *rotations += usize::from(rotated);
}

/// Run one task on its locked block(s); identical arithmetic whichever thread runs it.
fn run_task(task: Task, blocks: &[Mutex<Vec<Column<'_>>>], tolerance: f64, worst: &mut f64, rotations: &mut usize) {
    match task {
        Task::Within(b) => {
            let mut block = blocks[b].lock().expect("block lock");
            for p in 0..block.len() {
                let (head, tail) = block.split_at_mut(p + 1);
                for b in tail.iter_mut() {
                    pair(&mut head[p], b, tolerance, worst, rotations);
                }
            }
        }
        Task::Across(a, b) => {
            // Tasks within a step touch disjoint blocks: these locks are never contended.
            let mut left = blocks[a].lock().expect("block lock");
            let mut right = blocks[b].lock().expect("block lock");
            for a in left.iter_mut() {
                for b in right.iter_mut() {
                    pair(a, b, tolerance, worst, rotations);
                }
            }
        }
    }
}

fn sweep(x: &mut ColMajor, v: &mut ColMajor, norms2: &mut [f64], schedule: &[Vec<Task>], tolerance: f64, workers: usize) -> (f64, usize) {
    let (rows, n) = (x.rows, x.cols);
    let mut columns = x
        .data
        .chunks_exact_mut(rows)
        .zip(v.data.chunks_exact_mut(n))
        .zip(norms2.iter())
        .map(|((x, v), &norm2)| Column { x, v, norm2 });
    let blocks: Vec<Mutex<Vec<Column<'_>>>> = (0..n.div_ceil(BLOCK)).map(|_| Mutex::new(columns.by_ref().take(BLOCK).collect())).collect();
    let (worst, rotations) = if workers == 1 {
        let (mut worst, mut rotations) = (0.0_f64, 0usize);
        for step in schedule {
            for &task in step {
                run_task(task, &blocks, tolerance, &mut worst, &mut rotations);
            }
        }
        (worst, rotations)
    } else {
        let barrier = Barrier::new(workers);
        let (blocks, barrier) = (&blocks, &barrier);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|worker| {
                    scope.spawn(move || {
                        let (mut worst, mut rotations) = (0.0_f64, 0usize);
                        for step in schedule {
                            for &task in step.iter().skip(worker).step_by(workers) {
                                run_task(task, blocks, tolerance, &mut worst, &mut rotations);
                            }
                            barrier.wait();
                        }
                        (worst, rotations)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("Jacobi worker panicked"))
                .fold((0.0_f64, 0usize), |(w, r), (worst, rotations)| (w.max(worst), r + rotations))
        })
    };
    for (slot, column) in norms2.iter_mut().zip(blocks.into_iter().flat_map(|b| b.into_inner().expect("block lock"))) {
        *slot = column.norm2;
    }
    (worst, rotations)
}

/// The largest relative coupling over all pairs (no rotation).
fn coupling(x: &ColMajor, norms2: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for p in 0..x.cols {
        for q in p + 1..x.cols {
            if norms2[p] > 0.0 && norms2[q] > 0.0 {
                worst = worst.max(dot(x.col(p), x.col(q)).abs() / (norms2[p].sqrt() * norms2[q].sqrt()));
            }
        }
    }
    worst
}

/// de Rijk ordering: columns by decreasing norm, applied identically to `x`, `v` and the norms.
fn sort_by_norm(x: &mut ColMajor, v: &mut ColMajor, norms2: &mut [f64]) {
    for k in 0..norms2.len() {
        let best = (k..norms2.len()).fold(k, |best, j| if norms2[j] > norms2[best] { j } else { best });
        if best != k {
            x.swap_cols(k, best);
            v.swap_cols(k, best);
            norms2.swap(k, best);
        }
    }
}

fn sort_result(result: &mut JacobiResult) {
    let n = result.sigma.len();
    for k in 0..n {
        let best = (k..n).fold(k, |best, j| if result.sigma[j] > result.sigma[best] { j } else { best });
        if best != k {
            result.w.swap_cols(k, best);
            result.v.swap_cols(k, best);
            result.sigma.swap(k, best);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::round_robin;

    #[test]
    fn the_tournament_covers_every_pair_exactly_once_with_disjoint_steps() {
        for n in 1..40 {
            let schedule = round_robin(n);
            let mut seen = std::collections::BTreeSet::new();
            for step in &schedule {
                let mut used = std::collections::BTreeSet::new();
                for &(p, q) in step {
                    assert!(p < q && q < n);
                    assert!(used.insert(p) && used.insert(q), "n={n}: a column twice in one step");
                    assert!(seen.insert((p, q)), "n={n}: pair repeated");
                }
            }
            assert_eq!(seen.len(), n * n.saturating_sub(1) / 2, "n={n}");
        }
    }
}
