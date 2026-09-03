//! An exact (brute-force) k-nearest-neighbour index.

use crate::error::{validate_vector, VectorError};
use crate::metric::DistanceMetric;
use crate::store::VectorStore;
use crate::topk::{Neighbor, TopK};
use crate::{distance, VectorIndexStats};
use spacetimedb_memory_usage::MemoryUsage;
use std::hash::Hash;

/// An index that answers k-NN queries by comparing the query against every stored vector.
///
/// "Brute force" undersells it: because the vectors live end to end in one allocation
/// (see [`VectorStore`]), a search is a single linear pass over contiguous memory with a
/// branch-free inner loop. On a few hundred thousand embeddings that is typically a
/// millisecond or two — and unlike a graph index it is *exact*, so it returns the true
/// nearest neighbours every time, in an order that depends only on the set of rows
/// indexed. That determinism is what makes it the default for a replicated database.
///
/// Use [`HnswIndex`](crate::hnsw::HnswIndex) instead when the vector count grows past the
/// point where a linear pass fits the latency budget, and approximate results are
/// acceptable.
#[derive(Debug, Clone)]
pub struct ExactVectorIndex<P> {
    store: VectorStore<P>,
    metric: DistanceMetric,
}

impl<P: MemoryUsage> MemoryUsage for ExactVectorIndex<P> {
    fn heap_usage(&self) -> usize {
        self.store.heap_usage()
    }
}

impl<P: Copy + Eq + Hash + Ord> ExactVectorIndex<P> {
    /// Creates an empty index over `dimension`-dimensional vectors.
    ///
    /// Fails if `dimension` is zero or exceeds [`MAX_DIMENSION`](crate::MAX_DIMENSION).
    pub fn new(dimension: usize, metric: DistanceMetric) -> Result<Self, VectorError> {
        Ok(Self {
            // An exact index has no edges pointing at slots, so freed slots are safe to
            // hand out again, keeping memory proportional to the live row count.
            store: VectorStore::new(dimension, metric, true)?,
            metric,
        })
    }

    /// Creates an empty index with the same dimensionality and metric as `self`.
    pub fn clone_structure(&self) -> Self {
        Self {
            store: self.store.clone_structure(),
            metric: self.metric,
        }
    }

    /// The dimensionality of the indexed vectors.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.store.dimension()
    }

    /// The metric this index ranks by.
    #[inline]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// The number of indexed vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether nothing is indexed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Whether `payload` is indexed.
    #[inline]
    pub fn contains(&self, payload: &P) -> bool {
        self.store.slot_of(payload).is_some()
    }

    /// Returns the vector indexed under `payload`.
    #[inline]
    pub fn get(&self, payload: &P) -> Option<&[f32]> {
        self.store.slot_of(payload).and_then(|s| self.store.get(s))
    }

    /// Iterates over `(vector, payload)` for everything indexed.
    pub fn iter(&self) -> impl Iterator<Item = (&[f32], P)> + '_ {
        self.store.iter()
    }

    /// Iterates over the payloads of everything indexed.
    pub fn payloads(&self) -> impl Iterator<Item = P> + '_ {
        self.store.iter().map(|(_, p)| p)
    }

    /// Indexes `vector` under `payload`, replacing any vector already stored under it.
    pub fn insert(&mut self, vector: &[f32], payload: P) -> Result<(), VectorError> {
        self.store.insert(vector, payload).map(|_| ())
    }

    /// Removes whatever is indexed under `payload`, returning whether anything was.
    pub fn remove(&mut self, payload: &P) -> bool {
        self.store.remove(payload).is_some()
    }

    /// Empties the index, keeping its allocated capacity.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// Returns the `k` vectors nearest to `query`, nearest first.
    ///
    /// Fails if `query` has the wrong dimensionality or contains a non-finite component.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor<P>>, VectorError> {
        self.search_filtered(query, k, |_| true)
    }

    /// Like [`Self::search`], but skips any candidate for which `keep` returns `false`.
    ///
    /// The database uses this to hide rows a still-open transaction has deleted: those
    /// rows are physically in the committed index, but must not be visible to the
    /// transaction that deleted them.
    ///
    /// The filter runs on every candidate, so it should be cheap; it is applied *before*
    /// the distance is computed, which means an expensive filter that rejects most rows
    /// still saves work.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: impl Fn(&P) -> bool,
    ) -> Result<Vec<Neighbor<P>>, VectorError> {
        validate_vector(query, self.dimension())?;

        let mut top = TopK::new(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        let inv_norm_query = if self.metric.needs_norms() {
            distance::inv_norm(query)
        } else {
            0.0
        };

        for slot in self.store.iter_slots() {
            let payload = self.store.payload(slot).expect("slot came from `iter_slots`");
            if !keep(&payload) {
                continue;
            }
            let rank = self.metric.rank(
                query,
                self.store.vector_unchecked(slot),
                inv_norm_query,
                self.store.inv_norm(slot),
            );
            top.offer(rank, payload);
        }

        Ok(top.into_sorted_vec(|rank| self.metric.finalize(rank)))
    }

    /// Statistics for metrics and `EXPLAIN`-style output.
    pub fn stats(&self) -> VectorIndexStats {
        VectorIndexStats {
            vectors: self.len(),
            dimension: self.dimension(),
            metric: self.metric,
            approximate: false,
            graph_nodes: 0,
            tombstones: self.store.vacant_slots(),
            data_bytes: self.store.data_bytes(),
        }
    }

    /// The slot backing `payload`, exposed for tests.
    #[cfg(test)]
    fn slot_of(&self, payload: &P) -> Option<crate::store::Slot> {
        self.store.slot_of(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn index(dim: usize, metric: DistanceMetric) -> ExactVectorIndex<u64> {
        ExactVectorIndex::new(dim, metric).unwrap()
    }

    fn payloads(res: &[Neighbor<u64>]) -> Vec<u64> {
        res.iter().map(|n| n.payload).collect()
    }

    #[test]
    fn searching_an_empty_index_returns_nothing() {
        let idx = index(3, DistanceMetric::L2);
        assert!(idx.search(&[1.0, 2.0, 3.0], 5).unwrap().is_empty());
        assert!(idx.is_empty());
    }

    #[test]
    fn finds_the_nearest_neighbours_in_order() {
        let mut idx = index(2, DistanceMetric::L2);
        idx.insert(&[0.0, 0.0], 0).unwrap();
        idx.insert(&[1.0, 0.0], 1).unwrap();
        idx.insert(&[5.0, 0.0], 2).unwrap();
        idx.insert(&[10.0, 0.0], 3).unwrap();

        let got = idx.search(&[0.9, 0.0], 3).unwrap();
        assert_eq!(payloads(&got), vec![1, 0, 2]);
        assert!((got[0].distance - 0.1).abs() < 1e-5, "{:?}", got[0]);
        assert!((got[1].distance - 0.9).abs() < 1e-5, "{:?}", got[1]);
    }

    #[test]
    fn k_larger_than_the_index_returns_everything() {
        let mut idx = index(1, DistanceMetric::L2);
        idx.insert(&[1.0], 1).unwrap();
        idx.insert(&[2.0], 2).unwrap();
        assert_eq!(payloads(&idx.search(&[0.0], 100).unwrap()), vec![1, 2]);
    }

    #[test]
    fn k_zero_returns_nothing() {
        let mut idx = index(1, DistanceMetric::L2);
        idx.insert(&[1.0], 1).unwrap();
        assert!(idx.search(&[0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn rejects_a_query_of_the_wrong_dimension() {
        let idx = index(3, DistanceMetric::L2);
        assert_eq!(
            idx.search(&[1.0, 2.0], 1),
            Err(VectorError::DimensionMismatch { expected: 3, actual: 2 })
        );
    }

    #[test]
    fn rejects_a_non_finite_query() {
        let idx = index(2, DistanceMetric::L2);
        assert_eq!(
            idx.search(&[1.0, f32::NAN], 1),
            Err(VectorError::NonFinite { position: 1 })
        );
    }

    #[test]
    fn rejects_a_non_finite_insert() {
        let mut idx = index(2, DistanceMetric::L2);
        assert!(idx.insert(&[1.0, f32::INFINITY], 1).is_err());
        assert!(idx.is_empty());
    }

    #[test]
    fn removal_hides_a_vector() {
        let mut idx = index(1, DistanceMetric::L2);
        idx.insert(&[1.0], 1).unwrap();
        idx.insert(&[2.0], 2).unwrap();
        assert!(idx.remove(&1));
        assert!(!idx.remove(&1), "removing twice is a no-op");
        assert_eq!(payloads(&idx.search(&[0.0], 10).unwrap()), vec![2]);
        assert!(!idx.contains(&1));
    }

    #[test]
    fn reinsert_updates_in_place() {
        let mut idx = index(1, DistanceMetric::L2);
        idx.insert(&[1.0], 1).unwrap();
        let slot = idx.slot_of(&1).unwrap();
        idx.insert(&[100.0], 1).unwrap();
        assert_eq!(idx.slot_of(&1), Some(slot));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.get(&1), Some(&[100.0][..]));
        assert!(idx.search(&[0.0], 1).unwrap()[0].distance > 99.0);
    }

    #[test]
    fn the_filter_hides_candidates() {
        let mut idx = index(1, DistanceMetric::L2);
        for i in 0..10u64 {
            idx.insert(&[i as f32], i).unwrap();
        }
        let got = idx.search_filtered(&[0.0], 3, |p| p % 2 == 1).unwrap();
        assert_eq!(payloads(&got), vec![1, 3, 5]);
    }

    #[test]
    fn a_filter_rejecting_everything_returns_nothing() {
        let mut idx = index(1, DistanceMetric::L2);
        idx.insert(&[1.0], 1).unwrap();
        assert!(idx.search_filtered(&[0.0], 5, |_| false).unwrap().is_empty());
    }

    #[test]
    fn cosine_ranks_by_direction_not_magnitude() {
        let mut idx = index(2, DistanceMetric::Cosine);
        idx.insert(&[100.0, 0.0], 1).unwrap(); // same direction, huge magnitude
        idx.insert(&[0.1, 0.1], 2).unwrap(); // 45 degrees off, tiny magnitude
        idx.insert(&[-1.0, 0.0], 3).unwrap(); // opposite

        assert_eq!(payloads(&idx.search(&[1.0, 0.0], 3).unwrap()), vec![1, 2, 3]);
    }

    #[test]
    fn dot_product_ranks_by_largest_inner_product() {
        let mut idx = index(2, DistanceMetric::DotProduct);
        idx.insert(&[1.0, 0.0], 1).unwrap();
        idx.insert(&[10.0, 0.0], 2).unwrap();
        idx.insert(&[-5.0, 0.0], 3).unwrap();
        assert_eq!(payloads(&idx.search(&[1.0, 0.0], 3).unwrap()), vec![2, 1, 3]);
    }

    #[test]
    fn l1_ranks_by_manhattan_distance() {
        let mut idx = index(2, DistanceMetric::L1);
        idx.insert(&[1.0, 1.0], 1).unwrap(); // L1 = 2, L2 = sqrt(2)
        idx.insert(&[1.9, 0.0], 2).unwrap(); // L1 = 1.9, L2 = 1.9
        assert_eq!(payloads(&idx.search(&[0.0, 0.0], 2).unwrap()), vec![2, 1]);
    }

    #[test]
    fn ties_resolve_by_payload_not_insertion_order() {
        let mut a = index(1, DistanceMetric::L2);
        let mut b = index(1, DistanceMetric::L2);
        for p in [7u64, 2, 5] {
            a.insert(&[1.0], p).unwrap();
        }
        for p in [5u64, 7, 2] {
            b.insert(&[1.0], p).unwrap();
        }
        assert_eq!(payloads(&a.search(&[0.0], 2).unwrap()), vec![2, 5]);
        assert_eq!(
            payloads(&a.search(&[0.0], 2).unwrap()),
            payloads(&b.search(&[0.0], 2).unwrap())
        );
    }

    #[test]
    fn results_survive_a_churn_of_inserts_and_deletes() {
        let mut idx = index(2, DistanceMetric::L2);
        for i in 0..100u64 {
            idx.insert(&[i as f32, 0.0], i).unwrap();
        }
        for i in (0..100u64).step_by(2) {
            assert!(idx.remove(&i));
        }
        for i in 100..150u64 {
            idx.insert(&[i as f32, 0.0], i).unwrap();
        }
        assert_eq!(idx.len(), 50 + 50);
        let got = payloads(&idx.search(&[0.0, 0.0], 4).unwrap());
        assert_eq!(got, vec![1, 3, 5, 7]);
    }

    #[test]
    fn clear_empties_the_index() {
        let mut idx = index(1, DistanceMetric::L2);
        idx.insert(&[1.0], 1).unwrap();
        idx.clear();
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0], 1).unwrap().is_empty());
    }

    #[test]
    fn stats_describe_the_index() {
        let mut idx = index(4, DistanceMetric::Cosine);
        idx.insert(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        idx.insert(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
        let stats = idx.stats();
        assert_eq!(stats.vectors, 2);
        assert_eq!(stats.dimension, 4);
        assert_eq!(stats.metric, DistanceMetric::Cosine);
        assert!(!stats.approximate);
        assert_eq!(stats.data_bytes, 2 * 4 * 4);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The exact index must agree with the definition of "k nearest": sort everything
        /// by distance and take the first k.
        #[test]
        fn agrees_with_a_full_sort(
            vectors in prop::collection::vec(prop::collection::vec(-10.0f32..10.0, 5), 1..60),
            query in prop::collection::vec(-10.0f32..10.0, 5),
            k in 1usize..12,
            metric_idx in 0usize..4,
        ) {
            let metric = DistanceMetric::ALL[metric_idx];
            let mut idx = index(5, metric);
            for (i, v) in vectors.iter().enumerate() {
                idx.insert(v, i as u64).unwrap();
            }

            let mut want: Vec<(f32, u64)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (metric.distance(&query, v), i as u64))
                .collect();
            want.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            want.truncate(k);

            let got = idx.search(&query, k).unwrap();
            prop_assert_eq!(payloads(&got), want.iter().map(|(_, p)| *p).collect::<Vec<_>>());
            for (g, w) in got.iter().zip(&want) {
                prop_assert!((g.distance - w.0).abs() <= 1e-4 * w.0.abs().max(1.0));
            }
        }

        /// Results must depend only on the set of indexed vectors, never on the order they
        /// were inserted in, or two replicas replaying the same log could disagree.
        #[test]
        fn results_are_independent_of_insertion_order(
            vectors in prop::collection::vec(prop::collection::vec(-5.0f32..5.0, 4), 2..40),
            query in prop::collection::vec(-5.0f32..5.0, 4),
            k in 1usize..8,
        ) {
            let mut forward = index(4, DistanceMetric::L2);
            for (i, v) in vectors.iter().enumerate() {
                forward.insert(v, i as u64).unwrap();
            }
            let mut backward = index(4, DistanceMetric::L2);
            for (i, v) in vectors.iter().enumerate().rev() {
                backward.insert(v, i as u64).unwrap();
            }

            let a = forward.search(&query, k).unwrap();
            let b = backward.search(&query, k).unwrap();
            prop_assert_eq!(payloads(&a), payloads(&b));
        }

        /// Deleting a vector must give exactly the index that never held it.
        #[test]
        fn delete_matches_never_inserting(
            vectors in prop::collection::vec(prop::collection::vec(-5.0f32..5.0, 3), 2..30),
            query in prop::collection::vec(-5.0f32..5.0, 3),
            victim in 0usize..30,
            k in 1usize..6,
        ) {
            let victim = victim % vectors.len();

            let mut with_delete = index(3, DistanceMetric::L2);
            for (i, v) in vectors.iter().enumerate() {
                with_delete.insert(v, i as u64).unwrap();
            }
            prop_assert!(with_delete.remove(&(victim as u64)));

            let mut without = index(3, DistanceMetric::L2);
            for (i, v) in vectors.iter().enumerate() {
                if i != victim {
                    without.insert(v, i as u64).unwrap();
                }
            }

            prop_assert_eq!(
                payloads(&with_delete.search(&query, k).unwrap()),
                payloads(&without.search(&query, k).unwrap())
            );
        }
    }
}
