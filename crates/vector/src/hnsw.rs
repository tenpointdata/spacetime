//! A deterministic HNSW graph index for approximate nearest-neighbour search.
//!
//! HNSW ("Hierarchical Navigable Small World", Malkov & Yashunin 2016) builds a layered
//! proximity graph. The bottom layer contains every vector; each layer above holds an
//! exponentially thinning sample, acting as an express lane. A search greedily descends
//! the express lanes to land near the query, then explores the bottom layer. That turns a
//! linear scan into something closer to logarithmic, at the cost of occasionally missing
//! a true neighbour.
//!
//! # Determinism
//!
//! A replicated database cannot tolerate a randomized index: two replicas replaying the
//! same commitlog must agree. This implementation removes every source of randomness:
//!
//! - **Layer assignment** is derived by hashing the vector's own bits with a fixed seed
//!   (see [`HnswIndex::level_for`]), not from a random number generator. The same vector
//!   always lands on the same layer, on every replica and across a rebuild.
//! - **Every heap and candidate ordering** breaks ties on the slot id, so equal distances
//!   never leave the order up to heap internals.
//! - **Distance kernels** fix their floating-point accumulation order in the source, so
//!   they do not vary with the target's SIMD width. See [`crate::distance`].
//!
//! What remains is that the *graph itself* depends on insertion order: inserting A then B
//! can produce different edges than B then A. Replicas applying the same ordered
//! commitlog stay in lockstep, but a graph rebuilt from a snapshot may differ from the one
//! it replaced, and can therefore return a different approximate answer. This is inherent
//! to incrementally-built graph indexes. When a query must return the same answer
//! regardless of how the index was built, use
//! [`ExactVectorIndex`](crate::exact::ExactVectorIndex) instead — that is why it is the
//! default.

use crate::error::{validate_vector, VectorError};
use crate::metric::DistanceMetric;
use crate::store::{Slot, VectorStore};
use crate::topk::{Neighbor, TopK};
use crate::{distance, VectorIndexStats};
use spacetimedb_memory_usage::MemoryUsage;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::hash::Hash;

/// The most layers the graph will ever have.
///
/// With the default `m` of 16, layer `l` holds roughly `n / 16^l` vectors, so 16 layers
/// covers more vectors than can be addressed by a `u32` slot several times over.
pub const MAX_LEVEL: usize = 16;

/// Tuning knobs for [`HnswIndex`].
///
/// The defaults (`m = 16`, `ef_construction = 200`, `ef_search = 64`) are the values the
/// HNSW paper and most implementations settle on, and give recall above 0.95 on typical
/// embedding workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HnswParams {
    /// The maximum number of edges a node keeps on layers above the bottom.
    ///
    /// The bottom layer allows `2 * m`, since that is where recall is won or lost.
    /// Larger `m` means better recall and more memory: roughly `2 * m * 4` bytes of edges
    /// per vector.
    pub m: usize,

    /// How wide a search to run while *inserting*, in candidates.
    ///
    /// Higher values build a better-connected graph — improving every later query — at
    /// the cost of slower inserts. It is clamped up to at least `m`.
    pub ef_construction: usize,

    /// How wide a search to run while *querying*, in candidates.
    ///
    /// The effective width is always at least `k`. Raising it trades latency for recall,
    /// and unlike `m` or `ef_construction` it can be changed per query without rebuilding.
    pub ef_search: usize,

    /// Seed for the layer-assignment hash.
    ///
    /// Fixed by default so that every replica assigns the same layers. Changing it
    /// reshapes the graph but does not change which vectors are stored.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x5EED_0DDB_A5E0_1234,
        }
    }
}

impl HnswParams {
    /// Clamps the parameters into the ranges the implementation supports.
    ///
    /// `m` is forced into `2..=512` and `ef_construction` up to at least `m`, so a
    /// nonsensical module definition degrades into a slow index rather than a broken one.
    pub fn normalized(self) -> Self {
        let m = self.m.clamp(2, 512);
        Self {
            m,
            ef_construction: self.ef_construction.max(m),
            ef_search: self.ef_search.max(1),
            seed: self.seed,
        }
    }

    /// The edge budget for layer `level`.
    #[inline]
    fn max_edges(&self, level: usize) -> usize {
        if level == 0 {
            self.m * 2
        } else {
            self.m
        }
    }
}

impl MemoryUsage for HnswParams {}

/// One vector's edges, one list per layer it appears on.
#[derive(Debug, Clone, Default)]
struct Node {
    /// `links[l]` holds the neighbours on layer `l`. The node appears on layers
    /// `0..links.len()`.
    links: Vec<Vec<Slot>>,
}

impl MemoryUsage for Node {
    fn heap_usage(&self) -> usize {
        self.links.heap_usage()
    }
}

/// A candidate under consideration, ordered by distance and then by slot.
///
/// The slot tie-break is what makes the traversal reproducible: without it, two equal
/// distances would be ordered by whatever the binary heap happened to do.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    rank: f32,
    slot: Slot,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.total_cmp(&other.rank).then(self.slot.cmp(&other.slot))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// An approximate nearest-neighbour index built on a hierarchical navigable small-world
/// graph.
///
/// See the [module documentation](self) for the determinism guarantees and their limits.
///
/// # Deletion
///
/// Removing a vector *tombstones* it: the node stays in the graph as a navigation
/// waypoint, but is filtered out of results. Physically detaching a node would tear holes
/// in the graph's connectivity. Once tombstones outnumber live vectors the index rebuilds
/// itself from the survivors, which bounds the wasted memory and search work at roughly
/// 2x.
#[derive(Debug, Clone)]
pub struct HnswIndex<P> {
    store: VectorStore<P>,
    metric: DistanceMetric,
    params: HnswParams,
    /// One entry per slot ever allocated, parallel to the store's slots.
    nodes: Vec<Node>,
    /// The slot the search descends from, on the highest populated layer.
    entry: Option<Slot>,
    /// The highest layer any node currently occupies.
    max_level: usize,
    /// Below this many tombstones, never rebuild; small indexes churn without benefit.
    min_tombstones_before_rebuild: usize,
}

impl<P: MemoryUsage> MemoryUsage for HnswIndex<P> {
    fn heap_usage(&self) -> usize {
        self.store.heap_usage() + self.nodes.heap_usage()
    }
}

impl<P: Copy + Eq + Hash + Ord> HnswIndex<P> {
    /// Creates an empty index over `dimension`-dimensional vectors.
    ///
    /// Fails if `dimension` is zero or exceeds [`MAX_DIMENSION`](crate::MAX_DIMENSION).
    pub fn new(dimension: usize, metric: DistanceMetric, params: HnswParams) -> Result<Self, VectorError> {
        Ok(Self {
            // Graph edges refer to slots, so a freed slot must never be handed to a
            // different vector. Tombstoned slots are only reclaimed by a full rebuild.
            store: VectorStore::new(dimension, metric, false)?,
            metric,
            params: params.normalized(),
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
            min_tombstones_before_rebuild: 32,
        })
    }

    /// Creates an empty index with the same configuration as `self`.
    pub fn clone_structure(&self) -> Self {
        Self {
            store: self.store.clone_structure(),
            metric: self.metric,
            params: self.params,
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
            min_tombstones_before_rebuild: self.min_tombstones_before_rebuild,
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

    /// The graph's tuning parameters, after normalization.
    #[inline]
    pub fn params(&self) -> HnswParams {
        self.params
    }

    /// The number of live (non-tombstoned) vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether nothing live is indexed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Whether `payload` is indexed and not tombstoned.
    #[inline]
    pub fn contains(&self, payload: &P) -> bool {
        self.store.slot_of(payload).is_some()
    }

    /// Returns the vector indexed under `payload`.
    #[inline]
    pub fn get(&self, payload: &P) -> Option<&[f32]> {
        self.store.slot_of(payload).and_then(|s| self.store.get(s))
    }

    /// Iterates over `(vector, payload)` for every live vector.
    pub fn iter(&self) -> impl Iterator<Item = (&[f32], P)> + '_ {
        self.store.iter()
    }

    /// Iterates over the payloads of every live vector.
    pub fn payloads(&self) -> impl Iterator<Item = P> + '_ {
        self.store.iter().map(|(_, p)| p)
    }

    /// Indexes `vector` under `payload`.
    ///
    /// If `payload` is already indexed, its old node is tombstoned and a fresh one is
    /// linked in. Rewriting the vector in place would leave every edge that was chosen
    /// for the *old* vector pointing at the new one.
    pub fn insert(&mut self, vector: &[f32], payload: P) -> Result<(), VectorError> {
        validate_vector(vector, self.dimension())?;

        if self.contains(&payload) {
            self.remove(&payload);
        }

        let slot = self.store.insert(vector, payload)?;
        debug_assert_eq!(slot as usize, self.nodes.len(), "slots are handed out densely");
        let level = self.level_for(vector);
        self.nodes.resize_with(slot as usize + 1, Node::default);
        self.nodes[slot as usize].links = vec![Vec::new(); level + 1];

        self.link_into_graph(slot, vector, level);
        self.maybe_rebuild();
        Ok(())
    }

    /// Tombstones whatever is indexed under `payload`, returning whether anything was.
    pub fn remove(&mut self, payload: &P) -> bool {
        let removed = self.store.remove(payload).is_some();
        if removed {
            self.maybe_rebuild();
        }
        removed
    }

    /// Empties the index, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.store.clear();
        self.nodes.clear();
        self.entry = None;
        self.max_level = 0;
    }

    /// Returns approximately the `k` vectors nearest to `query`, nearest first.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor<P>>, VectorError> {
        self.search_with_ef(query, k, self.params.ef_search, |_| true)
    }

    /// Like [`Self::search`], but skips any candidate for which `keep` returns `false`.
    ///
    /// Rejected candidates are still traversed — they are needed to reach their
    /// neighbours — so a filter that hides most of the index will lose recall. Raise `ef`
    /// with [`Self::search_with_ef`] when filtering aggressively.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: impl Fn(&P) -> bool,
    ) -> Result<Vec<Neighbor<P>>, VectorError> {
        self.search_with_ef(query, k, self.params.ef_search, keep)
    }

    /// Like [`Self::search_filtered`], but with an explicit search width.
    ///
    /// The effective width is at least `k`, and is scaled up in proportion to the
    /// tombstone ratio so that a half-tombstoned index still finds `k` live results.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        keep: impl Fn(&P) -> bool,
    ) -> Result<Vec<Neighbor<P>>, VectorError> {
        validate_vector(query, self.dimension())?;
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };

        let inv_q = self.query_inv_norm(query);
        let ef = self.effective_ef(ef.max(k));

        // Descend the express lanes, greedily, one candidate wide.
        let mut ep = entry;
        for level in (1..=self.max_level).rev() {
            ep = self.greedy_descend(query, inv_q, ep, level);
        }

        // Explore the bottom layer properly.
        let found = self.search_layer(query, inv_q, ep, ef, 0);

        let mut top = TopK::new(k);
        for cand in found {
            // Tombstoned nodes navigate but never surface.
            if let Some(payload) = self.store.payload(cand.slot).filter(&keep) {
                top.offer(cand.rank, payload);
            }
        }
        Ok(top.into_sorted_vec(|rank| self.metric.finalize(rank)))
    }

    /// Statistics for metrics and `EXPLAIN`-style output.
    pub fn stats(&self) -> VectorIndexStats {
        VectorIndexStats {
            vectors: self.len(),
            dimension: self.dimension(),
            metric: self.metric,
            approximate: true,
            graph_nodes: self.nodes.len(),
            tombstones: self.store.vacant_slots(),
            data_bytes: self.store.data_bytes(),
        }
    }

    // =========================================================================
    // Construction internals
    // =========================================================================

    /// The layer a vector belongs on, derived from its own bits.
    ///
    /// The reference implementation draws `floor(-ln(uniform) * 1/ln(m))` from a random
    /// number generator. Two changes make that reproducible. First, the "randomness" comes
    /// from hashing the vector, so it is a pure function of the data. Second, the
    /// exponential is replaced by its discrete equivalent — repeatedly promote with
    /// probability `1/m` — because `ln` is provided by the platform's libm and is not
    /// guaranteed to be bit-identical across operating systems and architectures, while
    /// integer arithmetic is.
    fn level_for(&self, vector: &[f32]) -> usize {
        let mut h = self.params.seed;
        for &component in vector {
            // Normalize `-0.0` to `0.0` so that two numerically equal vectors, which
            // compare equal everywhere else, cannot land on different layers.
            let bits = if component == 0.0 { 0 } else { component.to_bits() };
            h = splitmix64(h ^ bits as u64);
        }

        let m = self.params.m as u64;
        let mut level = 0;
        while level < MAX_LEVEL && h.is_multiple_of(m) {
            level += 1;
            h = splitmix64(h);
        }
        level
    }

    /// Connects a freshly stored node into the graph at every layer up to `level`.
    fn link_into_graph(&mut self, slot: Slot, vector: &[f32], level: usize) {
        let inv_v = self.query_inv_norm(vector);

        let Some(entry) = self.entry else {
            // First node: it is the whole graph.
            self.entry = Some(slot);
            self.max_level = level;
            return;
        };

        // Descend the layers above the new node's own, one candidate wide.
        let mut ep = entry;
        for lc in ((level + 1)..=self.max_level).rev() {
            ep = self.greedy_descend(vector, inv_v, ep, lc);
        }

        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(vector, inv_v, ep, self.params.ef_construction, lc);
            if let Some(nearest) = candidates.first() {
                ep = nearest.slot;
            }

            let max_edges = self.params.max_edges(lc);
            let selected = self.select_neighbors(&candidates, max_edges);

            self.nodes[slot as usize].links[lc] = selected.clone();

            // Back-links, shrunk back to budget with the same heuristic.
            for neighbor in selected {
                let n = neighbor as usize;
                if self.nodes[n].links.len() <= lc {
                    // Cannot happen: a node found on layer `lc` occupies layer `lc`.
                    continue;
                }
                self.nodes[n].links[lc].push(slot);
                if self.nodes[n].links[lc].len() > max_edges {
                    self.shrink_edges(neighbor, lc, max_edges);
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(slot);
        }
    }

    /// Re-selects `node`'s edges on `level` down to `max_edges`, keeping a diverse set.
    fn shrink_edges(&mut self, node: Slot, level: usize, max_edges: usize) {
        let vector = self.store.vector_unchecked(node).to_vec();
        let inv = self.store.inv_norm(node);

        let mut candidates: Vec<Candidate> = std::mem::take(&mut self.nodes[node as usize].links[level])
            .into_iter()
            .map(|slot| Candidate {
                rank: self.rank_between(&vector, inv, slot),
                slot,
            })
            .collect();
        candidates.sort_unstable();

        self.nodes[node as usize].links[level] = self.select_neighbors(&candidates, max_edges);
    }

    /// Picks up to `max_edges` neighbours for `base` from `candidates` (nearest first).
    ///
    /// This is the paper's neighbour-selection heuristic rather than a plain "keep the
    /// nearest `m`". A candidate is kept only if it is closer to `base` than to every
    /// neighbour already kept — which favours edges pointing in *different directions*
    /// over a tight cluster of near-duplicates all on one side. Diverse edges are what
    /// let a greedy walk escape local minima, and they are worth far more to recall than
    /// a marginally shorter edge.
    ///
    /// Candidates rejected by the heuristic are used as filler if too few survive, so the
    /// edge budget is always spent.
    ///
    /// `candidates` must be sorted nearest-first, and each candidate's `rank` must already
    /// be its distance to the base vector — which is why the base vector itself is not a
    /// parameter here.
    fn select_neighbors(&self, candidates: &[Candidate], max_edges: usize) -> Vec<Slot> {
        let mut selected: Vec<Slot> = Vec::with_capacity(max_edges);
        let mut pruned: Vec<Slot> = Vec::new();

        for cand in candidates {
            if selected.len() >= max_edges {
                break;
            }
            let cand_vec = self.store.vector_unchecked(cand.slot);
            let cand_inv = self.store.inv_norm(cand.slot);

            // Keep `cand` only if no already-selected neighbour is closer to it than the
            // base is; otherwise that neighbour already covers this direction.
            let diverse = selected
                .iter()
                .all(|&sel| self.rank_between(cand_vec, cand_inv, sel) > cand.rank);

            if diverse {
                selected.push(cand.slot);
            } else {
                pruned.push(cand.slot);
            }
        }

        // Spend the rest of the budget on the nearest rejects rather than leaving the
        // node under-connected.
        for slot in pruned {
            if selected.len() >= max_edges {
                break;
            }
            selected.push(slot);
        }
        selected
    }

    /// Rebuilds from the live vectors once tombstones outnumber them.
    fn maybe_rebuild(&mut self) {
        let tombstones = self.store.vacant_slots();
        if tombstones < self.min_tombstones_before_rebuild || tombstones <= self.len() {
            return;
        }
        self.rebuild();
    }

    /// Rebuilds the graph from the live vectors, discarding tombstones.
    ///
    /// Exposed for tests and for callers that want to reclaim memory eagerly. The
    /// resulting graph may answer approximate queries differently from the one it
    /// replaces; see the [module docs](self).
    pub fn rebuild(&mut self) {
        let live: Vec<(Vec<f32>, P)> = self.store.iter().map(|(v, p)| (v.to_vec(), p)).collect();

        self.store.clear();
        self.nodes.clear();
        self.entry = None;
        self.max_level = 0;

        for (vector, payload) in live {
            let slot = self
                .store
                .insert(&vector, payload)
                .expect("a vector already in the index must still be valid");
            let level = self.level_for(&vector);
            self.nodes.resize_with(slot as usize + 1, Node::default);
            self.nodes[slot as usize].links = vec![Vec::new(); level + 1];
            self.link_into_graph(slot, &vector, level);
        }
    }

    // =========================================================================
    // Traversal internals
    // =========================================================================

    /// Walks greedily downhill on `level`, returning the local minimum reached.
    fn greedy_descend(&self, query: &[f32], inv_q: f32, entry: Slot, level: usize) -> Slot {
        let mut best = entry;
        let mut best_rank = self.rank_between(query, inv_q, entry);

        loop {
            let mut improved = false;
            for &neighbor in self.links(best, level) {
                let rank = self.rank_between(query, inv_q, neighbor);
                // The slot tie-break keeps the walk reproducible when distances are equal.
                if (rank, neighbor) < (best_rank, best) {
                    best = neighbor;
                    best_rank = rank;
                    improved = true;
                }
            }
            if !improved {
                return best;
            }
        }
    }

    /// Explores `level` from `entry`, returning up to `ef` candidates, nearest first.
    ///
    /// This is the paper's SEARCH-LAYER: a best-first traversal that stops once the
    /// closest unexplored candidate is further away than the worst result already held.
    fn search_layer(&self, query: &[f32], inv_q: f32, entry: Slot, ef: usize, level: usize) -> Vec<Candidate> {
        let start = Candidate {
            rank: self.rank_between(query, inv_q, entry),
            slot: entry,
        };

        let mut visited = HashSet::with_capacity(ef * 4);
        visited.insert(entry);

        // Nearest-first frontier.
        let mut frontier = BinaryHeap::new();
        frontier.push(Reverse(start));
        // Furthest-first results, capped at `ef`.
        let mut results = BinaryHeap::new();
        results.push(start);

        while let Some(Reverse(current)) = frontier.pop() {
            let worst = *results.peek().expect("results is never empty");
            if results.len() >= ef && current.rank > worst.rank {
                // Everything left in the frontier is at least this far away.
                break;
            }

            for &neighbor in self.links(current.slot, level) {
                if !visited.insert(neighbor) {
                    continue;
                }
                let cand = Candidate {
                    rank: self.rank_between(query, inv_q, neighbor),
                    slot: neighbor,
                };
                let worst = *results.peek().expect("results is never empty");
                if results.len() < ef || cand < worst {
                    frontier.push(Reverse(cand));
                    results.push(cand);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out = results.into_vec();
        out.sort_unstable();
        out
    }

    /// The neighbours of `slot` on `level`, or an empty slice if it does not reach it.
    #[inline]
    fn links(&self, slot: Slot, level: usize) -> &[Slot] {
        self.nodes[slot as usize].links.get(level).map_or(&[][..], |l| &l[..])
    }

    /// The ranking score between an arbitrary vector and the vector in `slot`.
    #[inline]
    fn rank_between(&self, vector: &[f32], inv_vector: f32, slot: Slot) -> f32 {
        self.metric.rank(
            vector,
            self.store.vector_unchecked(slot),
            inv_vector,
            self.store.inv_norm(slot),
        )
    }

    #[inline]
    fn query_inv_norm(&self, query: &[f32]) -> f32 {
        if self.metric.needs_norms() {
            distance::inv_norm(query)
        } else {
            0.0
        }
    }

    /// Widens the search in proportion to the tombstone ratio.
    ///
    /// Tombstoned nodes occupy result slots during traversal but are discarded at the end,
    /// so a half-dead index would otherwise return about half as many live results as
    /// asked for.
    fn effective_ef(&self, ef: usize) -> usize {
        let live = self.len().max(1);
        let total = self.nodes.len().max(1);
        let scaled = ef.saturating_mul(total) / live;
        scaled.clamp(ef, ef.saturating_mul(4)).min(total.max(ef))
    }
}

/// A fast, high-quality integer mixer.
///
/// Used to derive layer assignments from vector bits. Pure integer arithmetic, so the
/// result is identical on every platform.
#[inline]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::ExactVectorIndex;
    use proptest::prelude::*;

    fn index(dim: usize, metric: DistanceMetric) -> HnswIndex<u64> {
        HnswIndex::new(dim, metric, HnswParams::default()).unwrap()
    }

    fn payloads(res: &[Neighbor<u64>]) -> Vec<u64> {
        res.iter().map(|n| n.payload).collect()
    }

    /// A deterministic pseudo-random vector generator, so tests do not depend on `rand`.
    fn pseudo_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut h = seed;
        (0..count)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        h = splitmix64(h);
                        // Map into [-1, 1).
                        ((h >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn searching_an_empty_index_returns_nothing() {
        let idx = index(3, DistanceMetric::L2);
        assert!(idx.search(&[1.0, 2.0, 3.0], 5).unwrap().is_empty());
    }

    #[test]
    fn a_single_vector_is_found() {
        let mut idx = index(2, DistanceMetric::L2);
        idx.insert(&[1.0, 1.0], 42).unwrap();
        let got = idx.search(&[1.0, 1.0], 3).unwrap();
        assert_eq!(payloads(&got), vec![42]);
        assert_eq!(got[0].distance, 0.0);
    }

    #[test]
    fn k_zero_returns_nothing() {
        let mut idx = index(1, DistanceMetric::L2);
        idx.insert(&[1.0], 1).unwrap();
        assert!(idx.search(&[0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn rejects_bad_queries_and_inserts() {
        let mut idx = index(3, DistanceMetric::L2);
        assert!(idx.insert(&[1.0, 2.0], 1).is_err());
        assert!(idx.insert(&[1.0, 2.0, f32::NAN], 1).is_err());
        assert!(idx.search(&[1.0], 1).is_err());
        assert!(idx.is_empty());
    }

    #[test]
    fn params_are_normalized() {
        let idx = HnswIndex::<u64>::new(
            4,
            DistanceMetric::L2,
            HnswParams {
                m: 0,
                ef_construction: 1,
                ef_search: 0,
                seed: 7,
            },
        )
        .unwrap();
        let p = idx.params();
        assert_eq!(p.m, 2);
        assert_eq!(p.ef_construction, 2, "ef_construction is raised to at least m");
        assert_eq!(p.ef_search, 1);
        assert_eq!(p.seed, 7);
    }

    #[test]
    fn levels_are_a_pure_function_of_the_vector() {
        let a = index(4, DistanceMetric::L2);
        let b = index(4, DistanceMetric::L2);
        for v in pseudo_vectors(200, 4, 99) {
            assert_eq!(a.level_for(&v), b.level_for(&v), "level must not depend on index state");
        }
    }

    #[test]
    fn negative_zero_and_zero_get_the_same_level() {
        let idx = index(2, DistanceMetric::L2);
        assert_eq!(idx.level_for(&[0.0, 1.0]), idx.level_for(&[-0.0, 1.0]));
    }

    #[test]
    fn levels_follow_the_expected_distribution() {
        let idx = index(8, DistanceMetric::L2);
        let vectors = pseudo_vectors(20_000, 8, 7);
        let level0 = vectors.iter().filter(|v| idx.level_for(v) == 0).count();
        // With m = 16 the promotion probability is 1/16, so ~93.75% stay on layer 0.
        let ratio = level0 as f64 / vectors.len() as f64;
        assert!((0.92..0.96).contains(&ratio), "ratio was {ratio}");
        assert!(
            vectors.iter().any(|v| idx.level_for(v) >= 2),
            "no vector was promoted twice"
        );
    }

    #[test]
    fn recall_is_high_against_exact_search() {
        let dim = 16;
        let vectors = pseudo_vectors(2_000, dim, 12345);
        let mut hnsw = index(dim, DistanceMetric::L2);
        let mut exact = ExactVectorIndex::new(dim, DistanceMetric::L2).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            hnsw.insert(v, i as u64).unwrap();
            exact.insert(v, i as u64).unwrap();
        }

        let queries = pseudo_vectors(100, dim, 777);
        let k = 10;
        let mut hits = 0;
        let mut total = 0;
        for q in &queries {
            let want: HashSet<u64> = exact.search(q, k).unwrap().into_iter().map(|n| n.payload).collect();
            let got = hnsw.search(q, k).unwrap();
            assert_eq!(got.len(), k, "HNSW must still return k results");
            hits += got.iter().filter(|n| want.contains(&n.payload)).count();
            total += k;
        }
        let recall = hits as f64 / total as f64;
        assert!(recall > 0.95, "recall was {recall}");
    }

    #[test]
    fn recall_stays_high_for_cosine() {
        let dim = 12;
        let vectors = pseudo_vectors(1_000, dim, 4242);
        let mut hnsw = HnswIndex::new(dim, DistanceMetric::Cosine, HnswParams::default()).unwrap();
        let mut exact = ExactVectorIndex::new(dim, DistanceMetric::Cosine).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            hnsw.insert(v, i as u64).unwrap();
            exact.insert(v, i as u64).unwrap();
        }

        let mut hits = 0;
        let mut total = 0;
        for q in pseudo_vectors(50, dim, 31337) {
            let want: HashSet<u64> = exact.search(&q, 10).unwrap().into_iter().map(|n| n.payload).collect();
            hits += hnsw
                .search(&q, 10)
                .unwrap()
                .iter()
                .filter(|n| want.contains(&n.payload))
                .count();
            total += 10;
        }
        let recall = hits as f64 / total as f64;
        assert!(recall > 0.90, "recall was {recall}");
    }

    #[test]
    fn removal_hides_a_vector_from_results() {
        let dim = 8;
        let vectors = pseudo_vectors(300, dim, 55);
        let mut idx = index(dim, DistanceMetric::L2);
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(v, i as u64).unwrap();
        }

        let query = vectors[7].clone();
        assert_eq!(idx.search(&query, 1).unwrap()[0].payload, 7);
        assert!(idx.remove(&7));
        assert!(!idx.remove(&7));
        assert!(
            !idx.search(&query, 20).unwrap().iter().any(|n| n.payload == 7),
            "a tombstoned vector must not surface"
        );
        assert_eq!(idx.len(), 299);
    }

    #[test]
    fn tombstones_trigger_a_rebuild_and_results_stay_correct() {
        let dim = 6;
        let vectors = pseudo_vectors(400, dim, 909);
        let mut idx = index(dim, DistanceMetric::L2);
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(v, i as u64).unwrap();
        }
        assert_eq!(idx.stats().graph_nodes, 400);

        // Remove 300 of 400: crosses the "tombstones exceed live vectors" threshold.
        for i in 0..300u64 {
            assert!(idx.remove(&i));
        }
        assert_eq!(idx.len(), 100);
        assert!(
            idx.stats().graph_nodes <= 200,
            "the graph should have been rebuilt, but has {} nodes",
            idx.stats().graph_nodes
        );

        // Survivors must still be findable.
        for i in 300..400u64 {
            let got = idx.search(&vectors[i as usize], 1).unwrap();
            assert_eq!(got.len(), 1);
            assert_eq!(got[0].payload, i, "survivor {i} was not its own nearest neighbour");
        }
    }

    #[test]
    fn reinsert_replaces_the_vector() {
        let mut idx = index(2, DistanceMetric::L2);
        idx.insert(&[0.0, 0.0], 1).unwrap();
        idx.insert(&[5.0, 5.0], 2).unwrap();
        idx.insert(&[100.0, 100.0], 1).unwrap();

        assert_eq!(idx.len(), 2);
        assert_eq!(idx.get(&1), Some(&[100.0, 100.0][..]));
        let got = idx.search(&[0.0, 0.0], 2).unwrap();
        assert_eq!(got[0].payload, 2, "the updated vector must no longer be nearest");
    }

    #[test]
    fn the_filter_hides_candidates() {
        let dim = 4;
        let vectors = pseudo_vectors(200, dim, 616);
        let mut idx = index(dim, DistanceMetric::L2);
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(v, i as u64).unwrap();
        }
        let got = idx.search_filtered(&vectors[0], 5, |p| p % 2 == 1).unwrap();
        assert!(got.iter().all(|n| n.payload % 2 == 1), "{got:?}");
    }

    #[test]
    fn duplicate_vectors_do_not_break_the_graph() {
        let mut idx = index(3, DistanceMetric::L2);
        for i in 0..100u64 {
            idx.insert(&[1.0, 2.0, 3.0], i).unwrap();
        }
        let got = idx.search(&[1.0, 2.0, 3.0], 5).unwrap();
        assert_eq!(got.len(), 5);
        assert!(got.iter().all(|n| n.distance == 0.0));
        // The tie-break makes the answer the lowest payloads, deterministically.
        assert_eq!(payloads(&got), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn clear_empties_the_index() {
        let mut idx = index(2, DistanceMetric::L2);
        idx.insert(&[1.0, 1.0], 1).unwrap();
        idx.clear();
        assert!(idx.is_empty());
        assert!(idx.search(&[1.0, 1.0], 1).unwrap().is_empty());
        // Still usable.
        idx.insert(&[2.0, 2.0], 2).unwrap();
        assert_eq!(payloads(&idx.search(&[2.0, 2.0], 1).unwrap()), vec![2]);
    }

    #[test]
    fn identical_build_sequences_give_identical_answers() {
        let dim = 8;
        let vectors = pseudo_vectors(500, dim, 2024);
        let build = || {
            let mut idx = index(dim, DistanceMetric::L2);
            for (i, v) in vectors.iter().enumerate() {
                idx.insert(v, i as u64).unwrap();
            }
            idx
        };
        let (a, b) = (build(), build());
        for q in pseudo_vectors(40, dim, 4) {
            assert_eq!(
                payloads(&a.search(&q, 10).unwrap()),
                payloads(&b.search(&q, 10).unwrap()),
                "two identically-built graphs disagreed"
            );
        }
    }

    #[test]
    fn every_live_vector_is_its_own_nearest_neighbour() {
        let dim = 10;
        let vectors = pseudo_vectors(500, dim, 8);
        let mut idx = index(dim, DistanceMetric::L2);
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(v, i as u64).unwrap();
        }
        for (i, v) in vectors.iter().enumerate() {
            let got = idx.search(v, 1).unwrap();
            assert_eq!(got[0].payload, i as u64, "vector {i} did not find itself");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        /// On a small index the graph is dense enough that HNSW should be exact.
        #[test]
        fn small_indexes_match_exact_search(
            vectors in prop::collection::vec(prop::collection::vec(-5.0f32..5.0, 4), 1..30),
            query in prop::collection::vec(-5.0f32..5.0, 4),
            k in 1usize..5,
        ) {
            let mut hnsw = index(4, DistanceMetric::L2);
            let mut exact = ExactVectorIndex::new(4, DistanceMetric::L2).unwrap();
            for (i, v) in vectors.iter().enumerate() {
                hnsw.insert(v, i as u64).unwrap();
                exact.insert(v, i as u64).unwrap();
            }
            let want = exact.search(&query, k).unwrap();
            let got = hnsw.search(&query, k).unwrap();
            prop_assert_eq!(got.len(), want.len());
            // Compare distances rather than payloads: equidistant duplicates are
            // interchangeable and the two indexes may pick different representatives.
            for (g, w) in got.iter().zip(&want) {
                prop_assert!(
                    (g.distance - w.distance).abs() <= 1e-4 * w.distance.abs().max(1.0),
                    "{:?} vs {:?}", g, w
                );
            }
        }

        /// Insert/remove churn must leave a consistent index.
        #[test]
        fn churn_leaves_a_consistent_index(
            ops in prop::collection::vec((0u64..40, any::<bool>()), 1..120),
        ) {
            let mut hnsw = index(3, DistanceMetric::L2);
            let mut exact = ExactVectorIndex::new(3, DistanceMetric::L2).unwrap();
            for (payload, insert) in ops {
                let v = [payload as f32 * 0.1, 1.0, -(payload as f32) * 0.2];
                if insert {
                    hnsw.insert(&v, payload).unwrap();
                    exact.insert(&v, payload).unwrap();
                } else {
                    prop_assert_eq!(hnsw.remove(&payload), exact.remove(&payload));
                }
            }
            prop_assert_eq!(hnsw.len(), exact.len());

            let mut a: Vec<u64> = hnsw.payloads().collect();
            let mut b: Vec<u64> = exact.payloads().collect();
            a.sort_unstable();
            b.sort_unstable();
            prop_assert_eq!(a, b);
        }
    }
}
