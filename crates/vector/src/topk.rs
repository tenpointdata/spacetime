//! Bounded top-k selection with a deterministic total order.

use core::cmp::Ordering;

/// One result of a nearest-neighbour search.
///
/// Deliberately not `Eq`: `distance` is an `f32`, and comparing search results for exact
/// float equality is almost always a mistake. Compare payloads, or compare distances with
/// a tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor<P> {
    /// The distance from the query, under the index's metric.
    ///
    /// While a search is running this holds the metric's cheaper *ranking* score; it is
    /// converted to a true distance exactly once, when results are handed back.
    pub distance: f32,
    /// Whatever the caller associated with the vector — a row pointer, a row id, an
    /// index into their own storage.
    pub payload: P,
}

impl<P: Ord> Neighbor<P> {
    /// Orders neighbours nearest-first, breaking ties by payload.
    ///
    /// Two vectors can sit at exactly the same distance from a query — this is common
    /// with duplicate embeddings — and floating-point comparison alone would leave their
    /// relative order up to the internal state of the heap. Falling back to the payload
    /// makes the returned order a function of the result *set* alone, so replicas that
    /// hold the same rows return the same list in the same order.
    ///
    /// `f32::total_cmp` is used rather than `partial_cmp` so the order is total even
    /// though non-finite distances are rejected at insert time.
    #[inline]
    fn cmp_nearest_first(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.payload.cmp(&other.payload))
    }
}

/// Collects the `k` nearest neighbours seen so far.
///
/// Implemented as a bounded max-heap keyed on [`Neighbor::cmp_nearest_first`]: the
/// *worst* of the current best `k` sits at the root, so a new candidate can be rejected
/// with a single comparison. That makes a scan over `n` vectors `O(n + m log k)`, where
/// `m` is the number of candidates that actually made it in.
///
/// The heap is a hand-rolled binary heap over a `Vec` rather than
/// [`std::collections::BinaryHeap`] so that ties use the payload-aware ordering above and
/// so that the capacity is fixed up front, with no reallocation during a search.
#[derive(Debug, Clone)]
pub struct TopK<P> {
    heap: Vec<Neighbor<P>>,
    k: usize,
}

impl<P: Ord + Copy> TopK<P> {
    /// Creates a collector that keeps the `k` nearest neighbours.
    ///
    /// A `k` of zero produces a collector that accepts nothing.
    pub fn new(k: usize) -> Self {
        Self {
            heap: Vec::with_capacity(k),
            k,
        }
    }

    /// The number of neighbours collected so far, at most `k`.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether nothing has been collected yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Whether `k` neighbours have been collected, so that further candidates can only
    /// get in by displacing the current worst.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// The distance of the current worst neighbour, once full.
    ///
    /// A search can skip any candidate that is no better than this. Returns `None` while
    /// fewer than `k` neighbours have been collected, since nothing can be rejected yet.
    #[inline]
    pub fn worst_distance(&self) -> Option<f32> {
        if self.is_full() {
            self.heap.first().map(|n| n.distance)
        } else {
            None
        }
    }

    /// Offers a candidate, keeping it if it belongs in the top `k`.
    ///
    /// Returns whether the candidate was kept.
    pub fn offer(&mut self, distance: f32, payload: P) -> bool {
        if self.k == 0 {
            return false;
        }
        let candidate = Neighbor { distance, payload };

        if self.heap.len() < self.k {
            self.heap.push(candidate);
            self.sift_up(self.heap.len() - 1);
            return true;
        }

        // Full: only keep the candidate if it beats the current worst, which is the root.
        if candidate.cmp_nearest_first(&self.heap[0]).is_lt() {
            self.heap[0] = candidate;
            self.sift_down(0);
            true
        } else {
            false
        }
    }

    /// Returns the collected neighbours, nearest first.
    ///
    /// `finalize` converts each ranking score into the metric's reported distance; pass
    /// the identity function if the offered distances were already final.
    pub fn into_sorted_vec(mut self, finalize: impl Fn(f32) -> f32) -> Vec<Neighbor<P>> {
        self.heap.sort_by(|a, b| a.cmp_nearest_first(b));
        for n in &mut self.heap {
            n.distance = finalize(n.distance);
        }
        self.heap
    }

    /// Whether `distance` could still enter the top `k`.
    ///
    /// Cheaper than [`Self::offer`] when the caller would have to do work to produce the
    /// payload. Note this ignores tie-breaking, so it can return `true` for a candidate
    /// that [`Self::offer`] would then reject; that is the safe direction.
    #[inline]
    pub fn would_accept(&self, distance: f32) -> bool {
        self.k != 0 && (!self.is_full() || distance.total_cmp(&self.heap[0].distance).is_le())
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].cmp_nearest_first(&self.heap[parent]).is_gt() {
                self.heap.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let len = self.heap.len();
        loop {
            let (left, right) = (2 * i + 1, 2 * i + 2);
            let mut worst = i;
            if left < len && self.heap[left].cmp_nearest_first(&self.heap[worst]).is_gt() {
                worst = left;
            }
            if right < len && self.heap[right].cmp_nearest_first(&self.heap[worst]).is_gt() {
                worst = right;
            }
            if worst == i {
                break;
            }
            self.heap.swap(i, worst);
            i = worst;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn collect(k: usize, items: &[(f32, u32)]) -> Vec<(f32, u32)> {
        let mut top = TopK::new(k);
        for &(d, p) in items {
            top.offer(d, p);
        }
        top.into_sorted_vec(|d| d)
            .into_iter()
            .map(|n| (n.distance, n.payload))
            .collect()
    }

    #[test]
    fn keeps_the_k_smallest_in_order() {
        let items = [(5.0, 0), (1.0, 1), (4.0, 2), (2.0, 3), (3.0, 4)];
        assert_eq!(collect(3, &items), vec![(1.0, 1), (2.0, 3), (3.0, 4)]);
    }

    #[test]
    fn k_larger_than_input_returns_everything() {
        let items = [(2.0, 0), (1.0, 1)];
        assert_eq!(collect(10, &items), vec![(1.0, 1), (2.0, 0)]);
    }

    #[test]
    fn k_zero_collects_nothing() {
        let mut top = TopK::new(0);
        assert!(!top.offer(1.0, 7u32));
        assert!(!top.would_accept(f32::NEG_INFINITY));
        assert!(top.into_sorted_vec(|d| d).is_empty());
    }

    #[test]
    fn ties_are_broken_by_payload_regardless_of_insertion_order() {
        let forward = collect(2, &[(1.0, 9), (1.0, 3), (1.0, 5)]);
        let backward = collect(2, &[(1.0, 5), (1.0, 3), (1.0, 9)]);
        assert_eq!(forward, vec![(1.0, 3), (1.0, 5)]);
        assert_eq!(forward, backward);
    }

    #[test]
    fn worst_distance_is_none_until_full() {
        let mut top = TopK::new(2);
        assert_eq!(top.worst_distance(), None);
        top.offer(1.0, 0u32);
        assert_eq!(top.worst_distance(), None);
        top.offer(3.0, 1u32);
        assert_eq!(top.worst_distance(), Some(3.0));
        top.offer(2.0, 2u32);
        assert_eq!(top.worst_distance(), Some(2.0));
    }

    #[test]
    fn finalize_is_applied_to_every_result() {
        let mut top = TopK::new(2);
        top.offer(4.0, 0u32);
        top.offer(9.0, 1u32);
        let got: Vec<f32> = top.into_sorted_vec(f32::sqrt).into_iter().map(|n| n.distance).collect();
        assert_eq!(got, vec![2.0, 3.0]);
    }

    #[test]
    fn would_accept_agrees_with_offer_when_not_tied() {
        let mut top = TopK::new(2);
        top.offer(1.0, 0u32);
        top.offer(2.0, 1u32);
        assert!(top.would_accept(0.5));
        assert!(!top.would_accept(3.0));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// The heap must agree with the obvious "sort everything and truncate".
        #[test]
        fn matches_sort_and_truncate(
            items in prop::collection::vec((0.0f32..100.0, 0u32..1000), 0..200),
            k in 0usize..25,
        ) {
            // Payloads must be distinct for the reference to be unambiguous.
            let mut seen = std::collections::HashSet::new();
            let items: Vec<_> = items.into_iter().filter(|(_, p)| seen.insert(*p)).collect();

            let mut want: Vec<_> = items.clone();
            want.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            want.truncate(k);

            prop_assert_eq!(collect(k, &items), want);
        }

        /// Shuffling the input must not change the result, or replicas could disagree.
        #[test]
        fn result_is_independent_of_offer_order(
            items in prop::collection::vec((0.0f32..10.0, 0u32..50), 1..60),
            k in 1usize..10,
        ) {
            let mut seen = std::collections::HashSet::new();
            let items: Vec<_> = items.into_iter().filter(|(_, p)| seen.insert(*p)).collect();
            let reversed: Vec<_> = items.iter().rev().copied().collect();
            prop_assert_eq!(collect(k, &items), collect(k, &reversed));
        }
    }
}
