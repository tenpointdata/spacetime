//! Vector similarity search for SpacetimeDB.
//!
//! This crate is the engine behind SpacetimeDB's vector indexes — the machinery that lets
//! a table of embeddings answer "which `k` rows are most similar to this one?" without
//! scanning the table.
//!
//! It is deliberately free of any dependency on the rest of SpacetimeDB: it deals in
//! `&[f32]` vectors and opaque payloads. The storage engine instantiates the payload as a
//! row pointer, but the same code runs standalone, in tests, and inside WebAssembly
//! modules.
//!
//! # What is here
//!
//! - [`DistanceMetric`] — how similarity is measured: [`L2`](DistanceMetric::L2),
//!   [`Cosine`](DistanceMetric::Cosine), [`DotProduct`](DistanceMetric::DotProduct), and
//!   [`L1`](DistanceMetric::L1).
//! - [`ExactVectorIndex`] — exhaustive search over a contiguous vector arena. Always
//!   returns the true nearest neighbours. **The default.**
//! - [`HnswIndex`] — an approximate graph index for collections too large to scan.
//! - [`distance`] — the underlying kernels, if you want to compute a distance directly.
//!
//! # Choosing an index
//!
//! | | [`ExactVectorIndex`] | [`HnswIndex`] |
//! |---|---|---|
//! | Results | exact | approximate (typically >95% recall) |
//! | Query cost | `O(n * d)` over contiguous memory | `~O(log n * d)` |
//! | Insert cost | `O(d)` | `O(ef_construction * m * d)` |
//! | Memory beyond the vectors | negligible | `~2 * m * 4` bytes per vector |
//! | Same answer after a rebuild | yes | not guaranteed |
//!
//! Exact search is not the slow option people assume. The vectors live end to end in one
//! allocation, so a query is a single sequential pass; a few hundred thousand embeddings
//! are handled in low single-digit milliseconds. Reach for HNSW when that stops fitting
//! the latency budget and approximate answers are acceptable.
//!
//! # Determinism
//!
//! SpacetimeDB replicates by replaying a commitlog, so identical inputs must produce
//! identical outputs on every replica. Three things here exist to guarantee that, and are
//! worth knowing about before changing any of this code:
//!
//! 1. **Distance kernels fix their floating-point accumulation order in the source**, so
//!    results do not change with the target's SIMD width. See [`distance`].
//! 2. **Every result ordering breaks ties on the payload**, so equal distances cannot be
//!    ordered by heap internals. See [`Neighbor`].
//! 3. **HNSW derives its layer assignment by hashing the vector**, not from a random
//!    number generator, and avoids `ln` because libm is not bit-identical across
//!    platforms. See [`hnsw`].
//!
//! [`ExactVectorIndex`] additionally returns results that depend only on the *set* of
//! indexed vectors — never on the order they arrived in. [`HnswIndex`] cannot promise
//! that, because a proximity graph is shaped by its build order.
//!
//! # Example
//!
//! ```
//! use spacetimedb_vector::{DistanceMetric, ExactVectorIndex};
//!
//! // Payloads are opaque; a database uses row pointers, this example uses ids.
//! let mut index = ExactVectorIndex::new(3, DistanceMetric::Cosine)?;
//! index.insert(&[1.0, 0.0, 0.0], 1u64)?;
//! index.insert(&[0.9, 0.1, 0.0], 2u64)?;
//! index.insert(&[0.0, 0.0, 1.0], 3u64)?;
//!
//! let hits = index.search(&[1.0, 0.0, 0.0], 2)?;
//! assert_eq!(hits.iter().map(|n| n.payload).collect::<Vec<_>>(), vec![1, 2]);
//! # Ok::<(), spacetimedb_vector::VectorError>(())
//! ```

pub mod distance;
pub mod error;
pub mod exact;
pub mod hnsw;
pub mod metric;
pub mod store;
pub mod topk;

pub use error::{VectorError, MAX_DIMENSION};
pub use exact::ExactVectorIndex;
pub use hnsw::{HnswIndex, HnswParams};
pub use metric::DistanceMetric;
pub use store::VectorStore;
pub use topk::{Neighbor, TopK};

use spacetimedb_memory_usage::MemoryUsage;

/// A description of a vector index's contents, for metrics and query plan output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorIndexStats {
    /// The number of live vectors indexed.
    pub vectors: usize,
    /// The dimensionality of those vectors.
    pub dimension: usize,
    /// The metric the index ranks by.
    pub metric: DistanceMetric,
    /// Whether searches are approximate.
    pub approximate: bool,
    /// The number of nodes in the graph, including tombstones. Zero for an exact index.
    pub graph_nodes: usize,
    /// The number of removed-but-not-yet-reclaimed vectors.
    pub tombstones: usize,
    /// The bytes of vector data held, excluding bookkeeping.
    pub data_bytes: u64,
}

impl MemoryUsage for VectorIndexStats {}

/// Either kind of vector index, so that callers can be written once and configured later.
///
/// The storage engine holds one of these per vector index, choosing the variant from the
/// table's schema.
#[derive(Debug, Clone)]
pub enum VectorIndex<P> {
    /// Exhaustive search; always correct.
    Exact(ExactVectorIndex<P>),
    /// Approximate graph search; faster on large collections.
    Hnsw(HnswIndex<P>),
}

impl<P: MemoryUsage> MemoryUsage for VectorIndex<P> {
    fn heap_usage(&self) -> usize {
        match self {
            Self::Exact(i) => i.heap_usage(),
            Self::Hnsw(i) => i.heap_usage(),
        }
    }
}

impl<P: Copy + Eq + core::hash::Hash + Ord> VectorIndex<P> {
    /// Creates an exact index.
    pub fn exact(dimension: usize, metric: DistanceMetric) -> Result<Self, VectorError> {
        ExactVectorIndex::new(dimension, metric).map(Self::Exact)
    }

    /// Creates an approximate HNSW index.
    pub fn hnsw(dimension: usize, metric: DistanceMetric, params: HnswParams) -> Result<Self, VectorError> {
        HnswIndex::new(dimension, metric, params).map(Self::Hnsw)
    }

    /// Creates an empty index with the same configuration as `self`.
    pub fn clone_structure(&self) -> Self {
        match self {
            Self::Exact(i) => Self::Exact(i.clone_structure()),
            Self::Hnsw(i) => Self::Hnsw(i.clone_structure()),
        }
    }

    /// The dimensionality of the indexed vectors.
    pub fn dimension(&self) -> usize {
        match self {
            Self::Exact(i) => i.dimension(),
            Self::Hnsw(i) => i.dimension(),
        }
    }

    /// The metric this index ranks by.
    pub fn metric(&self) -> DistanceMetric {
        match self {
            Self::Exact(i) => i.metric(),
            Self::Hnsw(i) => i.metric(),
        }
    }

    /// The number of live vectors indexed.
    pub fn len(&self) -> usize {
        match self {
            Self::Exact(i) => i.len(),
            Self::Hnsw(i) => i.len(),
        }
    }

    /// Whether nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `payload` is indexed.
    pub fn contains(&self, payload: &P) -> bool {
        match self {
            Self::Exact(i) => i.contains(payload),
            Self::Hnsw(i) => i.contains(payload),
        }
    }

    /// Indexes `vector` under `payload`, replacing anything already stored under it.
    pub fn insert(&mut self, vector: &[f32], payload: P) -> Result<(), VectorError> {
        match self {
            Self::Exact(i) => i.insert(vector, payload),
            Self::Hnsw(i) => i.insert(vector, payload),
        }
    }

    /// Removes whatever is indexed under `payload`, returning whether anything was.
    pub fn remove(&mut self, payload: &P) -> bool {
        match self {
            Self::Exact(i) => i.remove(payload),
            Self::Hnsw(i) => i.remove(payload),
        }
    }

    /// Empties the index, keeping allocated capacity.
    pub fn clear(&mut self) {
        match self {
            Self::Exact(i) => i.clear(),
            Self::Hnsw(i) => i.clear(),
        }
    }

    /// Returns the `k` nearest vectors to `query`, nearest first.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor<P>>, VectorError> {
        self.search_filtered(query, k, |_| true)
    }

    /// Like [`Self::search`], but skips candidates for which `keep` returns `false`.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: impl Fn(&P) -> bool,
    ) -> Result<Vec<Neighbor<P>>, VectorError> {
        match self {
            Self::Exact(i) => i.search_filtered(query, k, keep),
            Self::Hnsw(i) => i.search_filtered(query, k, keep),
        }
    }

    /// Iterates over the payloads of every live vector.
    pub fn payloads(&self) -> Box<dyn Iterator<Item = P> + '_> {
        match self {
            Self::Exact(i) => Box::new(i.payloads()),
            Self::Hnsw(i) => Box::new(i.payloads()),
        }
    }

    /// Statistics for metrics and query plan output.
    pub fn stats(&self) -> VectorIndexStats {
        match self {
            Self::Exact(i) => i.stats(),
            Self::Hnsw(i) => i.stats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_enum_dispatches_to_both_kinds() {
        for mut index in [
            VectorIndex::<u64>::exact(2, DistanceMetric::L2).unwrap(),
            VectorIndex::<u64>::hnsw(2, DistanceMetric::L2, HnswParams::default()).unwrap(),
        ] {
            assert!(index.is_empty());
            index.insert(&[1.0, 0.0], 1).unwrap();
            index.insert(&[0.0, 1.0], 2).unwrap();
            assert_eq!(index.len(), 2);
            assert!(index.contains(&1));
            assert_eq!(index.dimension(), 2);
            assert_eq!(index.metric(), DistanceMetric::L2);

            let got = index.search(&[0.9, 0.0], 1).unwrap();
            assert_eq!(got[0].payload, 1);

            assert!(index.remove(&1));
            assert!(!index.contains(&1));
            assert_eq!(index.payloads().collect::<Vec<_>>(), vec![2]);

            let fresh = index.clone_structure();
            assert!(fresh.is_empty());
            assert_eq!(fresh.dimension(), 2);

            index.clear();
            assert!(index.is_empty());
            assert_eq!(index.stats().vectors, 0);
        }
    }

    #[test]
    fn stats_report_approximation() {
        let exact = VectorIndex::<u64>::exact(2, DistanceMetric::L2).unwrap();
        let hnsw = VectorIndex::<u64>::hnsw(2, DistanceMetric::L2, HnswParams::default()).unwrap();
        assert!(!exact.stats().approximate);
        assert!(hnsw.stats().approximate);
    }

    #[test]
    fn a_bad_dimension_is_rejected_by_both_kinds() {
        assert!(VectorIndex::<u64>::exact(0, DistanceMetric::L2).is_err());
        assert!(VectorIndex::<u64>::hnsw(MAX_DIMENSION + 1, DistanceMetric::L2, HnswParams::default()).is_err());
    }
}
