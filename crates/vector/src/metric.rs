//! The distance metrics a vector index can be built with.

use crate::distance;
use core::fmt;

/// How similarity between two vectors is measured.
///
/// Every metric is expressed as a *distance*: smaller means more similar. A k-nearest
/// neighbour search always returns the `k` entries with the smallest distance.
///
/// Searching computes a cheaper, order-preserving [ranking score](Self::rank) and only
/// converts the survivors to a true distance with [`Self::finalize`]. For example, L2
/// ranks on the squared distance and takes the square root just `k` times, rather than
/// once per indexed vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum DistanceMetric {
    /// Euclidean distance: `sqrt(sum((a[i] - b[i])^2))`.
    ///
    /// The default, and the right choice when the magnitude of an embedding carries
    /// meaning.
    #[default]
    L2,

    /// Cosine distance: `1 - (a . b) / (|a| * |b|)`, in `[0, 2]`.
    ///
    /// The usual choice for text embeddings, which are compared by direction rather
    /// than magnitude. A zero vector has no direction, so its cosine distance to
    /// anything is defined here as `1.0` (i.e. orthogonal) rather than `NaN`.
    Cosine,

    /// Negated inner product: `-(a . b)`.
    ///
    /// Ranks by *maximum* inner product, which is what models trained with a dot-product
    /// objective expect. Note that the reported distance is negative for similar vectors;
    /// it is a distance only in the sense that smaller is more similar.
    DotProduct,

    /// Manhattan distance: `sum(|a[i] - b[i]|)`.
    L1,
}

impl DistanceMetric {
    /// All metrics, in declaration order. Useful for exhaustive tests.
    pub const ALL: [Self; 4] = [Self::L2, Self::Cosine, Self::DotProduct, Self::L1];

    /// Whether this metric needs the L2 norm of each indexed vector precomputed.
    ///
    /// Only cosine does; storing `1 / |v|` alongside each vector turns cosine into a
    /// dot product plus two multiplies.
    #[inline]
    pub const fn needs_norms(self) -> bool {
        matches!(self, Self::Cosine)
    }

    /// Returns a score that is monotonically increasing in the true distance.
    ///
    /// `inv_norm_a` and `inv_norm_b` must be `1 / |a|` and `1 / |b|` (or `0.0` for a
    /// zero vector), as produced by [`distance::inv_norm`]. They are ignored by every
    /// metric except [`DistanceMetric::Cosine`], so passing `0.0` for the others is fine.
    ///
    /// `a` and `b` must have the same length; callers are responsible for checking that.
    #[inline]
    pub fn rank(self, a: &[f32], b: &[f32], inv_norm_a: f32, inv_norm_b: f32) -> f32 {
        match self {
            Self::L2 => distance::l2_squared(a, b),
            // The two norms are multiplied together *first*. Written as
            // `dot * inv_a * inv_b`, this would associate left-to-right, and
            // floating-point multiplication is not associative — so swapping `a` and `b`
            // would give a subtly different answer, making the metric asymmetric.
            Self::Cosine => 1.0 - distance::dot(a, b) * (inv_norm_a * inv_norm_b),
            Self::DotProduct => -distance::dot(a, b),
            Self::L1 => distance::l1(a, b),
        }
    }

    /// Converts a [ranking score](Self::rank) into the metric's true distance.
    #[inline]
    pub fn finalize(self, rank: f32) -> f32 {
        match self {
            // `rank` is the *squared* distance for L2; every other metric ranks on the
            // distance itself.
            Self::L2 => rank.sqrt(),
            Self::Cosine | Self::DotProduct | Self::L1 => rank,
        }
    }

    /// Computes the distance between `a` and `b` directly.
    ///
    /// This is the straightforward, self-contained definition: it recomputes norms as
    /// needed and applies [`Self::finalize`]. Indexes use [`Self::rank`] instead so they
    /// can reuse precomputed norms and skip the final square root for rejected
    /// candidates; this method is for one-off comparisons and for tests.
    ///
    /// `a` and `b` must have the same length; callers are responsible for checking that.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        let (ia, ib) = if self.needs_norms() {
            (distance::inv_norm(a), distance::inv_norm(b))
        } else {
            (0.0, 0.0)
        };
        self.finalize(self.rank(a, b, ia, ib))
    }

    /// The metric's name as it appears in module definitions and SQL, e.g. `"cosine"`.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::DotProduct => "dot_product",
            Self::L1 => "l1",
        }
    }

    /// Parses a metric from its [name](Self::name).
    ///
    /// Also accepts the common aliases `euclidean` for L2, `ip`/`inner_product`/`dot`
    /// for dot product, and `manhattan`/`taxicab` for L1. Matching is case-insensitive.
    pub fn from_name(name: &str) -> Option<Self> {
        // `to_ascii_lowercase` allocates, but this only runs when validating a module
        // definition or parsing SQL, never in a search loop.
        match &*name.to_ascii_lowercase() {
            "l2" | "euclidean" | "l2_distance" => Some(Self::L2),
            "cosine" | "cos" | "cosine_distance" => Some(Self::Cosine),
            "dot_product" | "dot" | "ip" | "inner_product" | "negative_inner_product" => Some(Self::DotProduct),
            "l1" | "manhattan" | "taxicab" => Some(Self::L1),
            _ => None,
        }
    }
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl spacetimedb_memory_usage::MemoryUsage for DistanceMetric {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn l2_distance_is_euclidean() {
        assert_eq!(DistanceMetric::L2.distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
    }

    #[test]
    fn cosine_of_identical_direction_is_zero() {
        let d = DistanceMetric::Cosine.distance(&[1.0, 0.0], &[7.0, 0.0]);
        assert!(d.abs() < 1e-6, "{d}");
    }

    #[test]
    fn cosine_of_opposite_direction_is_two() {
        let d = DistanceMetric::Cosine.distance(&[1.0, 0.0], &[-7.0, 0.0]);
        assert!((d - 2.0).abs() < 1e-6, "{d}");
    }

    #[test]
    fn cosine_of_orthogonal_is_one() {
        let d = DistanceMetric::Cosine.distance(&[1.0, 0.0], &[0.0, 5.0]);
        assert!((d - 1.0).abs() < 1e-6, "{d}");
    }

    #[test]
    fn cosine_against_zero_vector_is_one_not_nan() {
        let d = DistanceMetric::Cosine.distance(&[0.0, 0.0], &[1.0, 2.0]);
        assert_eq!(d, 1.0);
        let d = DistanceMetric::Cosine.distance(&[0.0, 0.0], &[0.0, 0.0]);
        assert_eq!(d, 1.0);
    }

    #[test]
    fn dot_product_prefers_larger_inner_products() {
        let q = [1.0, 1.0];
        let near = DistanceMetric::DotProduct.distance(&q, &[5.0, 5.0]);
        let far = DistanceMetric::DotProduct.distance(&q, &[1.0, 1.0]);
        assert!(near < far, "{near} !< {far}");
    }

    #[test]
    fn names_round_trip() {
        for m in DistanceMetric::ALL {
            assert_eq!(DistanceMetric::from_name(m.name()), Some(m));
            assert_eq!(DistanceMetric::from_name(&m.name().to_uppercase()), Some(m));
            assert_eq!(m.to_string(), m.name());
        }
        assert_eq!(DistanceMetric::from_name("nonsense"), None);
    }

    #[test]
    fn aliases_are_accepted() {
        assert_eq!(DistanceMetric::from_name("euclidean"), Some(DistanceMetric::L2));
        assert_eq!(DistanceMetric::from_name("Manhattan"), Some(DistanceMetric::L1));
        assert_eq!(DistanceMetric::from_name("ip"), Some(DistanceMetric::DotProduct));
    }

    proptest! {
        /// `rank` must order candidates exactly as `distance` does, since search ranks on
        /// the former and reports the latter.
        #[test]
        fn rank_is_monotone_in_distance(
            q in prop::collection::vec(-5.0f32..5.0, 4..16),
            a in prop::collection::vec(-5.0f32..5.0, 4..16),
            b in prop::collection::vec(-5.0f32..5.0, 4..16),
        ) {
            let dim = q.len().min(a.len()).min(b.len());
            let (q, a, b) = (&q[..dim], &a[..dim], &b[..dim]);

            for m in DistanceMetric::ALL {
                let iq = crate::distance::inv_norm(q);
                let ra = m.rank(q, a, iq, crate::distance::inv_norm(a));
                let rb = m.rank(q, b, iq, crate::distance::inv_norm(b));
                let da = m.finalize(ra);
                let db = m.finalize(rb);
                prop_assert_eq!(ra.partial_cmp(&rb), da.partial_cmp(&db), "metric {:?}", m);
            }
        }

        /// The convenience `distance` method must agree with the `rank`/`finalize` pair
        /// the indexes actually use.
        #[test]
        fn distance_agrees_with_rank_then_finalize(
            a in prop::collection::vec(-5.0f32..5.0, 1..32),
            b in prop::collection::vec(-5.0f32..5.0, 1..32),
        ) {
            let dim = a.len().min(b.len());
            let (a, b) = (&a[..dim], &b[..dim]);
            for m in DistanceMetric::ALL {
                let via_rank = m.finalize(m.rank(a, b, crate::distance::inv_norm(a), crate::distance::inv_norm(b)));
                prop_assert_eq!(m.distance(a, b), via_rank, "metric {:?}", m);
            }
        }

        /// Distances must be symmetric. (Dot product is not a metric in the mathematical
        /// sense, but it is still symmetric in its two arguments.)
        #[test]
        fn distances_are_symmetric(
            a in prop::collection::vec(-5.0f32..5.0, 1..32),
            b in prop::collection::vec(-5.0f32..5.0, 1..32),
        ) {
            let dim = a.len().min(b.len());
            let (a, b) = (&a[..dim], &b[..dim]);
            for m in DistanceMetric::ALL {
                prop_assert_eq!(m.distance(a, b), m.distance(b, a), "metric {:?}", m);
            }
        }
    }
}
