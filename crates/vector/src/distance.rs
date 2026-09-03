//! Low-level distance kernels over `f32` slices.
//!
//! # Determinism
//!
//! SpacetimeDB replicates a database by replaying its commitlog, so every replica must
//! compute *bit-identical* results. Floating-point summation is not associative, which
//! means a kernel written as a naive scalar loop and a kernel auto-vectorized by LLVM
//! into 4- or 8-wide SIMD produce *different* sums for the same input. If the host
//! binaries on two replicas were compiled for different target features, a naive loop
//! would silently diverge.
//!
//! To avoid this, every reduction in this module fixes its accumulation order *in the
//! source*: we accumulate into a fixed array of [`LANES`] partial sums and then combine
//! them with a fixed reduction tree. Rust never reassociates floating-point arithmetic
//! and does not enable FP contraction (fused multiply-add) by default, so the emitted
//! code performs exactly the additions written here, in exactly this order, on every
//! target. The compiler is still free to vectorize the per-lane loop — it just cannot
//! change *which* numbers get added to which.
//!
//! [`LANES`] is 8 so that the pattern maps cleanly onto 128-bit (4 x f32) and
//! 256-bit (8 x f32) SIMD registers.

/// The number of independent accumulators used by the reductions in this module.
///
/// See the module documentation for why this is fixed.
pub const LANES: usize = 8;

/// Combines the per-lane accumulators with a fixed reduction tree.
///
/// The order of additions here is part of this crate's observable behaviour;
/// changing it changes the results of every search.
#[inline]
fn reduce(acc: [f32; LANES]) -> f32 {
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// Runs `lane` over `a` and `b` in [`LANES`]-wide chunks, then folds in the tail.
///
/// `lane(x, y)` computes the per-element contribution, which is summed into the
/// accumulator for that lane.
#[inline]
fn fold(a: &[f32], b: &[f32], lane: impl Fn(f32, f32) -> f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut acc = [0.0f32; LANES];
    let chunks = a.len() / LANES;

    for i in 0..chunks {
        let base = i * LANES;
        // Bounds checks are hoisted out of the inner loop by slicing to a fixed size,
        // which is also what lets LLVM vectorize the lane loop.
        let a = &a[base..base + LANES];
        let b = &b[base..base + LANES];
        for l in 0..LANES {
            acc[l] += lane(a[l], b[l]);
        }
    }

    let mut sum = reduce(acc);
    for i in (chunks * LANES)..a.len() {
        sum += lane(a[i], b[i]);
    }
    sum
}

/// Returns the dot (inner) product of `a` and `b`.
///
/// # Panics
///
/// In debug builds, panics if `a` and `b` have different lengths.
/// In release builds, the shorter length wins and the result is meaningless,
/// so callers must validate dimensions first — every entry point in this crate does.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    fold(a, b, |x, y| x * y)
}

/// Returns the squared Euclidean (L2) distance between `a` and `b`.
///
/// This is monotone in the true L2 distance, so it can be used for ranking
/// without paying for a square root per candidate.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    fold(a, b, |x, y| {
        let d = x - y;
        d * d
    })
}

/// Returns the Euclidean (L2) distance between `a` and `b`.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Returns the Manhattan (L1) distance between `a` and `b`.
#[inline]
pub fn l1(a: &[f32], b: &[f32]) -> f32 {
    fold(a, b, |x, y| (x - y).abs())
}

/// Returns the squared L2 norm of `v`, i.e. `dot(v, v)`.
#[inline]
pub fn norm_squared(v: &[f32]) -> f32 {
    dot(v, v)
}

/// Returns the L2 norm (magnitude) of `v`.
#[inline]
pub fn norm(v: &[f32]) -> f32 {
    norm_squared(v).sqrt()
}

/// Returns `1.0 / norm(v)`, or `0.0` when `v` is the zero vector.
///
/// Returning `0.0` for the zero vector makes cosine similarity against it evaluate to
/// `0.0` (i.e. maximally dissimilar) rather than `NaN`. Cosine similarity is genuinely
/// undefined there; this is a deliberate, documented choice so that a stray zero vector
/// cannot poison a search with `NaN`s.
#[inline]
pub fn inv_norm(v: &[f32]) -> f32 {
    let n2 = norm_squared(v);
    if n2 > 0.0 {
        1.0 / n2.sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A deliberately naive reference implementation, used to check the chunked kernels
    /// agree with the obvious definition up to floating-point rounding.
    fn naive(a: &[f32], b: &[f32], lane: impl Fn(f32, f32) -> f32) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| lane(x, y) as f64).sum()
    }

    fn close(got: f32, want: f64) -> bool {
        let tol = 1e-4 * want.abs().max(1.0);
        ((got as f64) - want).abs() <= tol
    }

    #[test]
    fn dot_matches_by_hand() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn l2_matches_by_hand() {
        assert_eq!(l2_squared(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        assert_eq!(l2(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
        assert_eq!(l1(&[1.0, -1.0], &[4.0, 3.0]), 7.0);
    }

    #[test]
    fn inv_norm_of_zero_is_zero() {
        assert_eq!(inv_norm(&[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(inv_norm(&[3.0, 4.0]), 0.2);
    }

    /// The chunked kernels must be exercised across every possible tail length,
    /// since the tail is handled by a separate code path.
    #[test]
    fn every_tail_length_is_handled() {
        for dim in 0..(3 * LANES + 1) {
            let a: Vec<f32> = (0..dim).map(|i| i as f32 + 0.5).collect();
            let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.25).collect();

            assert!(close(dot(&a, &b), naive(&a, &b, |x, y| x * y)), "dot dim={dim}");
            assert!(
                close(l2_squared(&a, &b), naive(&a, &b, |x, y| (x - y) * (x - y))),
                "l2_squared dim={dim}"
            );
            assert!(close(l1(&a, &b), naive(&a, &b, |x, y| (x - y).abs())), "l1 dim={dim}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn kernels_agree_with_naive_reference(
            v in prop::collection::vec((-10.0f32..10.0, -10.0f32..10.0), 0..64)
        ) {
            let a: Vec<f32> = v.iter().map(|p| p.0).collect();
            let b: Vec<f32> = v.iter().map(|p| p.1).collect();

            prop_assert!(close(dot(&a, &b), naive(&a, &b, |x, y| x * y)));
            prop_assert!(close(l2_squared(&a, &b), naive(&a, &b, |x, y| (x - y) * (x - y))));
            prop_assert!(close(l1(&a, &b), naive(&a, &b, |x, y| (x - y).abs())));
        }

        #[test]
        fn l2_is_symmetric_and_zero_on_equal(v in prop::collection::vec(-10.0f32..10.0, 1..64)) {
            prop_assert_eq!(l2_squared(&v, &v), 0.0);
            let rev: Vec<f32> = v.iter().rev().copied().collect();
            prop_assert_eq!(l2_squared(&v, &rev), l2_squared(&rev, &v));
        }

        #[test]
        fn dot_is_commutative(v in prop::collection::vec((-10.0f32..10.0, -10.0f32..10.0), 0..64)) {
            let a: Vec<f32> = v.iter().map(|p| p.0).collect();
            let b: Vec<f32> = v.iter().map(|p| p.1).collect();
            prop_assert_eq!(dot(&a, &b), dot(&b, &a));
        }
    }
}
