//! Contiguous storage for the vectors backing an index.

use crate::distance;
use crate::error::{validate_dimension, validate_vector, VectorError};
use crate::metric::DistanceMetric;
use spacetimedb_memory_usage::MemoryUsage;
use std::collections::HashMap;
use std::hash::Hash;

/// Identifies one vector inside a [`VectorStore`].
///
/// Slots are dense and reused (unless the store was built with reuse disabled), so a slot
/// is only meaningful together with the store it came from, and only until the vector it
/// refers to is removed.
pub type Slot = u32;

/// The vectors of one index, laid out end to end in a single allocation.
///
/// This is the reason a vector index beats scanning the table: rather than walking rows,
/// decoding a variable-length array out of each one, and chasing pointers, a search walks
/// one contiguous `Vec<f32>`. The whole working set is sequential, prefetch-friendly, and
/// free of per-candidate branching.
///
/// Each occupied slot holds:
/// - `dimension` consecutive floats in [`VectorStore::data`]'s backing allocation,
/// - the caller's payload (typically a row pointer),
/// - `1 / |v|`, but only when the metric [needs norms](DistanceMetric::needs_norms).
///
/// # Slot reuse
///
/// [`VectorStore::new`] takes `reuse_slots`. An exact index reuses freed slots, keeping
/// storage proportional to the live row count. A graph index must *not*, because its
/// edges refer to slots: handing a freed slot to a new vector would silently re-point
/// every edge that still refers to it. Such a store instead grows until the owning index
/// decides to rebuild.
#[derive(Debug, Clone)]
pub struct VectorStore<P> {
    /// `slots * dimension` floats. The vector for slot `s` is
    /// `data[s * dimension..][..dimension]`. Vacant slots keep their stale contents,
    /// which are never read.
    data: Vec<f32>,
    /// One entry per slot; meaningful only for occupied slots.
    payloads: Vec<P>,
    /// One entry per slot, `1 / |v|`. Empty when the metric does not need norms.
    inv_norms: Vec<f32>,
    /// Whether each slot is occupied, one bit per slot.
    occupied: Vec<u64>,
    /// Vacant slots below `slot_count`, available for reuse. Always empty when
    /// `reuse_slots` is false.
    free: Vec<Slot>,
    /// Reverse map, so a vector can be removed by payload without a scan.
    slot_of: HashMap<P, Slot>,
    /// The number of slots ever allocated, occupied or not.
    slot_count: usize,
    dimension: usize,
    metric: DistanceMetric,
    reuse_slots: bool,
}

impl<P: MemoryUsage> MemoryUsage for VectorStore<P> {
    fn heap_usage(&self) -> usize {
        self.data.heap_usage()
            + self.payloads.heap_usage()
            + self.inv_norms.heap_usage()
            + self.occupied.heap_usage()
            + self.free.heap_usage()
            // `std::collections::HashMap` exposes no allocation size, so estimate from
            // capacity. Hashbrown lays out one `(K, V)` plus one control byte per bucket.
            + self.slot_of.capacity() * (size_of::<(P, Slot)>() + 1)
    }
}

impl<P: Copy + Eq + Hash> VectorStore<P> {
    /// Creates an empty store for `dimension`-dimensional vectors.
    ///
    /// See [the type docs](VectorStore#slot-reuse) for `reuse_slots`.
    pub fn new(dimension: usize, metric: DistanceMetric, reuse_slots: bool) -> Result<Self, VectorError> {
        validate_dimension(dimension)?;
        Ok(Self {
            data: Vec::new(),
            payloads: Vec::new(),
            inv_norms: Vec::new(),
            occupied: Vec::new(),
            free: Vec::new(),
            slot_of: HashMap::new(),
            slot_count: 0,
            dimension,
            metric,
            reuse_slots,
        })
    }

    /// Creates an empty store with the same configuration as `self`.
    pub fn clone_structure(&self) -> Self {
        Self {
            data: Vec::new(),
            payloads: Vec::new(),
            inv_norms: Vec::new(),
            occupied: Vec::new(),
            free: Vec::new(),
            slot_of: HashMap::new(),
            slot_count: 0,
            dimension: self.dimension,
            metric: self.metric,
            reuse_slots: self.reuse_slots,
        }
    }

    /// The dimensionality every vector in this store has.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The metric this store precomputes for.
    #[inline]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// The number of vectors currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.slot_of.len()
    }

    /// Whether no vectors are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slot_of.is_empty()
    }

    /// The number of slots ever allocated, including vacant ones.
    ///
    /// Slots are dense, so this is also an exclusive upper bound on any valid slot.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// The number of allocated-but-vacant slots.
    ///
    /// For a store with reuse disabled this is the tombstone count, which is what an
    /// owning graph index watches to decide when to rebuild.
    #[inline]
    pub fn vacant_slots(&self) -> usize {
        self.slot_count - self.len()
    }

    /// Whether `slot` currently holds a vector.
    #[inline]
    pub fn is_occupied(&self, slot: Slot) -> bool {
        let slot = slot as usize;
        slot < self.slot_count && (self.occupied[slot / 64] >> (slot % 64)) & 1 == 1
    }

    /// Returns the vector in `slot`, or `None` if the slot is vacant or out of range.
    #[inline]
    pub fn get(&self, slot: Slot) -> Option<&[f32]> {
        self.is_occupied(slot).then(|| self.vector_unchecked(slot))
    }

    /// Returns the vector in `slot` without checking that it is occupied.
    ///
    /// Reading a vacant slot yields stale floats rather than undefined behaviour, so this
    /// is safe; it is only used on paths that have already established occupancy.
    #[inline]
    pub fn vector_unchecked(&self, slot: Slot) -> &[f32] {
        let start = slot as usize * self.dimension;
        &self.data[start..start + self.dimension]
    }

    /// Returns the payload associated with `slot`, or `None` if the slot is vacant.
    #[inline]
    pub fn payload(&self, slot: Slot) -> Option<P> {
        self.is_occupied(slot).then(|| self.payloads[slot as usize])
    }

    /// Returns `1 / |v|` for `slot`, or `0.0` when the metric does not need norms.
    #[inline]
    pub fn inv_norm(&self, slot: Slot) -> f32 {
        if self.inv_norms.is_empty() {
            0.0
        } else {
            self.inv_norms[slot as usize]
        }
    }

    /// Returns the slot holding `payload`, if any.
    #[inline]
    pub fn slot_of(&self, payload: &P) -> Option<Slot> {
        self.slot_of.get(payload).copied()
    }

    /// Iterates over the occupied slots, in ascending slot order.
    pub fn iter_slots(&self) -> impl Iterator<Item = Slot> + '_ {
        (0..self.slot_count as Slot).filter(|&s| self.is_occupied(s))
    }

    /// Iterates over `(vector, payload)` for every occupied slot, in ascending slot order.
    pub fn iter(&self) -> impl Iterator<Item = (&[f32], P)> + '_ {
        self.iter_slots()
            .map(move |s| (self.vector_unchecked(s), self.payloads[s as usize]))
    }

    /// Stores `vector` under `payload`, returning the slot it landed in.
    ///
    /// If `payload` is already present its vector is replaced in place, reusing the same
    /// slot. That keeps a graph index's edges valid across an update.
    ///
    /// Returns an error if `vector` has the wrong length or contains a non-finite
    /// component; the store is unchanged in that case.
    pub fn insert(&mut self, vector: &[f32], payload: P) -> Result<Slot, VectorError> {
        validate_vector(vector, self.dimension)?;

        if let Some(slot) = self.slot_of(&payload) {
            self.write_vector(slot, vector);
            return Ok(slot);
        }

        let slot = match self.free.pop() {
            Some(slot) => {
                self.write_vector(slot, vector);
                self.payloads[slot as usize] = payload;
                slot
            }
            None => {
                let slot = self.slot_count as Slot;
                self.slot_count += 1;
                self.data.extend_from_slice(vector);
                self.payloads.push(payload);
                if self.metric.needs_norms() {
                    self.inv_norms.push(distance::inv_norm(vector));
                }
                if self.slot_count.div_ceil(64) > self.occupied.len() {
                    self.occupied.push(0);
                }
                slot
            }
        };

        self.set_occupied(slot, true);
        self.slot_of.insert(payload, slot);
        Ok(slot)
    }

    /// Removes the vector stored under `payload`.
    ///
    /// Returns the slot it occupied, or `None` if the payload was not present.
    pub fn remove(&mut self, payload: &P) -> Option<Slot> {
        let slot = self.slot_of.remove(payload)?;
        self.set_occupied(slot, false);
        if self.reuse_slots {
            self.free.push(slot);
        }
        Some(slot)
    }

    /// Removes every vector, keeping the allocated capacity.
    pub fn clear(&mut self) {
        self.data.clear();
        self.payloads.clear();
        self.inv_norms.clear();
        self.occupied.clear();
        self.free.clear();
        self.slot_of.clear();
        self.slot_count = 0;
    }

    /// The number of bytes of vector data held, excluding bookkeeping.
    #[inline]
    pub fn data_bytes(&self) -> u64 {
        (self.data.len() * size_of::<f32>()) as u64
    }

    fn write_vector(&mut self, slot: Slot, vector: &[f32]) {
        let start = slot as usize * self.dimension;
        self.data[start..start + self.dimension].copy_from_slice(vector);
        if self.metric.needs_norms() {
            self.inv_norms[slot as usize] = distance::inv_norm(vector);
        }
    }

    fn set_occupied(&mut self, slot: Slot, occupied: bool) {
        let (word, bit) = (slot as usize / 64, slot as usize % 64);
        if occupied {
            self.occupied[word] |= 1 << bit;
        } else {
            self.occupied[word] &= !(1 << bit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(dim: usize, reuse: bool) -> VectorStore<u64> {
        VectorStore::new(dim, DistanceMetric::L2, reuse).unwrap()
    }

    #[test]
    fn rejects_invalid_dimension() {
        assert!(VectorStore::<u64>::new(0, DistanceMetric::L2, true).is_err());
        assert!(VectorStore::<u64>::new(crate::error::MAX_DIMENSION + 1, DistanceMetric::L2, true).is_err());
    }

    #[test]
    fn insert_and_read_back() {
        let mut s = store(3, true);
        let slot = s.insert(&[1.0, 2.0, 3.0], 10).unwrap();
        assert_eq!(s.get(slot), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(s.payload(slot), Some(10));
        assert_eq!(s.slot_of(&10), Some(slot));
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
    }

    #[test]
    fn rejects_bad_vectors_without_mutating() {
        let mut s = store(3, true);
        assert!(s.insert(&[1.0, 2.0], 1).is_err());
        assert!(s.insert(&[1.0, 2.0, f32::NAN], 1).is_err());
        assert_eq!(s.len(), 0);
        assert_eq!(s.slot_count(), 0);
    }

    #[test]
    fn reinsert_same_payload_updates_in_place() {
        let mut s = store(2, true);
        let a = s.insert(&[1.0, 1.0], 7).unwrap();
        let b = s.insert(&[9.0, 9.0], 7).unwrap();
        assert_eq!(a, b, "an update must keep the same slot");
        assert_eq!(s.get(a), Some(&[9.0, 9.0][..]));
        assert_eq!(s.len(), 1);
        assert_eq!(s.slot_count(), 1);
    }

    #[test]
    fn removal_frees_slots_for_reuse() {
        let mut s = store(2, true);
        s.insert(&[1.0, 1.0], 1).unwrap();
        let slot = s.insert(&[2.0, 2.0], 2).unwrap();
        assert_eq!(s.remove(&2), Some(slot));
        assert_eq!(s.remove(&2), None, "removing twice is a no-op");
        assert_eq!(s.len(), 1);
        assert_eq!(s.vacant_slots(), 1);
        assert!(!s.is_occupied(slot));
        assert_eq!(s.get(slot), None);

        let reused = s.insert(&[3.0, 3.0], 3).unwrap();
        assert_eq!(reused, slot, "a freed slot should be reused");
        assert_eq!(s.slot_count(), 2);
    }

    #[test]
    fn removal_tombstones_when_reuse_is_disabled() {
        let mut s = store(2, false);
        s.insert(&[1.0, 1.0], 1).unwrap();
        let slot = s.insert(&[2.0, 2.0], 2).unwrap();
        s.remove(&2).unwrap();
        assert_eq!(s.vacant_slots(), 1);

        let fresh = s.insert(&[3.0, 3.0], 3).unwrap();
        assert_ne!(fresh, slot, "a tombstoned slot must never be handed out again");
        assert_eq!(s.slot_count(), 3);
    }

    #[test]
    fn iteration_skips_vacant_slots_and_is_ordered() {
        let mut s = store(1, true);
        for i in 0..5u64 {
            s.insert(&[i as f32], i).unwrap();
        }
        s.remove(&1).unwrap();
        s.remove(&3).unwrap();
        let got: Vec<u64> = s.iter().map(|(_, p)| p).collect();
        assert_eq!(got, vec![0, 2, 4]);
        assert_eq!(s.iter_slots().collect::<Vec<_>>(), vec![0, 2, 4]);
    }

    #[test]
    fn occupancy_bitset_spans_multiple_words() {
        let mut s = store(1, true);
        for i in 0..200u64 {
            s.insert(&[i as f32], i).unwrap();
        }
        for i in (0..200u64).step_by(3) {
            s.remove(&i).unwrap();
        }
        let want: Vec<u64> = (0..200u64).filter(|i| i % 3 != 0).collect();
        assert_eq!(s.iter().map(|(_, p)| p).collect::<Vec<_>>(), want);
        assert_eq!(s.len(), want.len());
    }

    #[test]
    fn norms_are_cached_only_for_cosine() {
        let mut l2 = VectorStore::<u64>::new(2, DistanceMetric::L2, true).unwrap();
        let slot = l2.insert(&[3.0, 4.0], 1).unwrap();
        assert_eq!(l2.inv_norm(slot), 0.0, "L2 does not need norms");

        let mut cos = VectorStore::<u64>::new(2, DistanceMetric::Cosine, true).unwrap();
        let slot = cos.insert(&[3.0, 4.0], 1).unwrap();
        assert_eq!(cos.inv_norm(slot), 0.2);
        // An update must refresh the cached norm.
        cos.insert(&[6.0, 8.0], 1).unwrap();
        assert_eq!(cos.inv_norm(slot), 0.1);
    }

    #[test]
    fn clear_empties_everything() {
        let mut s = store(2, true);
        s.insert(&[1.0, 1.0], 1).unwrap();
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.slot_count(), 0);
        assert_eq!(s.slot_of(&1), None);
        // The store must still be usable afterwards.
        let slot = s.insert(&[2.0, 2.0], 2).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(s.get(0), Some(&[2.0, 2.0][..]));
    }

    #[test]
    fn clone_structure_is_empty_but_configured() {
        let mut s = VectorStore::<u64>::new(4, DistanceMetric::Cosine, false).unwrap();
        s.insert(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        let fresh = s.clone_structure();
        assert!(fresh.is_empty());
        assert_eq!(fresh.dimension(), 4);
        assert_eq!(fresh.metric(), DistanceMetric::Cosine);
    }
}
