//! The storage engine's adapter around [`spacetimedb_vector`].
//!
//! [`VectorTableIndex`] holds the vector engine's index keyed by [`RowPointer`], and knows
//! how to pull an embedding out of a row. Everything else about vector search lives in
//! `spacetimedb-vector`, which knows nothing about rows, pages, or transactions.

use super::RowPointer;
use crate::table::RowRef;
use core::fmt;
use spacetimedb_data_structures::map::HashSet;
use spacetimedb_memory_usage::MemoryUsage;
use spacetimedb_primitives::ColList;
use spacetimedb_sats::{AlgebraicValue, ArrayValue};
use spacetimedb_schema::def::{VectorAlgorithm, VectorStrategy};
use spacetimedb_vector::{DistanceMetric, Neighbor, VectorError, VectorIndex};

/// Why a k-nearest-neighbour search could not be answered.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum VectorSearchError {
    /// The index named by the caller is a B-tree, hash, or direct index.
    #[error("index is not a vector index, so it cannot answer a nearest-neighbour search")]
    NotAVectorIndex,

    /// The query vector was malformed.
    #[error(transparent)]
    BadQuery(#[from] VectorError),
}

/// A vector similarity index over one column of a table.
///
/// # Rows that cannot be indexed
///
/// The indexed column has type `Vec<f32>`, which says nothing about length: SATS has no
/// fixed-size array type, so the dimensionality is declared on the index instead. A row
/// can therefore arrive carrying a vector of the wrong length, or one containing `NaN`.
///
/// Such a row is still *accepted* — refusing it would mean threading a brand new failure
/// mode through [`Table::insert`](crate::table::Table::insert)'s unsafe, roll-back-on-error
/// path, which today only ever fails on a unique constraint. Instead the row is recorded in
/// [`VectorTableIndex::unindexed`] and excluded from search results: a vector of a
/// different dimension has no defined distance to the query, so it is not a neighbour of
/// anything.
///
/// Tracking those rows rather than dropping them on the floor matters for two reasons.
/// Deletion still works, because the index is asked to forget a `RowPointer` it must
/// actually know about — and `RowPointer`s are recycled, so a missed deletion would not
/// dangle, it would silently alias a *different* row. And
/// [`Table::num_rows_in_indexes`](crate::table::Table::num_rows_in_indexes) assumes every
/// index holds every row, which stays true.
///
/// Callers that want a hard error instead should validate the vector's length before
/// inserting; the module bindings do exactly that, so a reducer sees a clean failure.
/// [`VectorTableIndex::num_unindexed`] exposes the count for anyone who wants to check.
pub struct VectorTableIndex {
    /// The vector engine's index, exact or approximate per the schema.
    index: VectorIndex<RowPointer>,

    /// Rows whose vector could not be indexed. See the type docs.
    ///
    /// Expected to be empty in practice, so this stays cheap.
    unindexed: HashSet<RowPointer>,

    /// The declared dimensionality. Also held by `index`, but needed to construct
    /// [`Self::clone_structure`] and to report errors.
    dimension: usize,

    /// How similarity is measured.
    metric: DistanceMetric,

    /// Exact or approximate, retained so the structure can be cloned.
    strategy: VectorStrategy,

    /// Reused buffer for the vector read out of a row, so that inserting does not
    /// allocate once per row.
    ///
    /// Not part of the index's value: excluded from equality and from memory reporting.
    scratch: Vec<f32>,
}

impl VectorTableIndex {
    /// Creates an empty index per `algorithm`.
    ///
    /// Fails only if `algorithm`'s dimension is out of range, which schema validation has
    /// already rejected.
    pub fn new(algorithm: &VectorAlgorithm) -> Result<Self, VectorError> {
        let dimension = algorithm.dimension as usize;
        let index = match algorithm.strategy {
            VectorStrategy::Exact => VectorIndex::exact(dimension, algorithm.metric)?,
            VectorStrategy::Hnsw(params) => VectorIndex::hnsw(dimension, algorithm.metric, params)?,
        };
        Ok(Self {
            index,
            unindexed: HashSet::default(),
            dimension,
            metric: algorithm.metric,
            strategy: algorithm.strategy,
            scratch: Vec::new(),
        })
    }

    /// Creates an empty index with the same configuration as `self`.
    pub fn clone_structure(&self) -> Self {
        Self {
            index: self.index.clone_structure(),
            unindexed: HashSet::default(),
            dimension: self.dimension,
            metric: self.metric,
            strategy: self.strategy,
            scratch: Vec::new(),
        }
    }

    /// The dimensionality every indexed vector has.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// How similarity is measured.
    #[inline]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Whether searches against this index are approximate.
    #[inline]
    pub fn is_approximate(&self) -> bool {
        matches!(self.strategy, VectorStrategy::Hnsw(_))
    }

    /// The number of rows this index knows about, indexed or not.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.index.len() + self.unindexed.len()
    }

    /// The number of rows accepted but excluded from search. See the type docs.
    #[inline]
    pub fn num_unindexed(&self) -> usize {
        self.unindexed.len()
    }

    /// The number of live user-supplied bytes held in this index's keys.
    ///
    /// Every indexed vector contributes `4 * dimension` bytes, matching what
    /// [`KeySize`](super::KeySize) reports for an `ArrayValue::F32`. This deliberately
    /// excludes the graph edges of an approximate index, which are representational
    /// overhead rather than user data; `heap_usage` accounts for those.
    #[inline]
    pub fn num_key_bytes(&self) -> u64 {
        (self.index.len() * self.dimension * size_of::<f32>()) as u64
    }

    /// Whether this index holds `ptr`.
    #[inline]
    pub fn contains(&self, ptr: RowPointer) -> bool {
        self.index.contains(&ptr) || self.unindexed.contains(&ptr)
    }

    /// Iterates over every row pointer this index holds, indexed or not.
    ///
    /// The order is unspecified.
    pub fn iter(&self) -> impl Iterator<Item = RowPointer> + '_ {
        self.index.payloads().chain(self.unindexed.iter().copied())
    }

    /// Indexes the vector in `row_ref`'s column `cols`, under `row_ref`'s pointer.
    ///
    /// Never fails: a vector that cannot be indexed is recorded as unindexed instead.
    /// See the type docs for why.
    pub fn insert(&mut self, cols: &ColList, row_ref: RowRef<'_>) {
        let ptr = row_ref.pointer();
        // Take the buffer so the borrow of `self` in `read_vector` does not conflict.
        let mut scratch = core::mem::take(&mut self.scratch);
        let read = read_vector_into(cols, row_ref, &mut scratch);

        if read && self.index.insert(&scratch, ptr).is_ok() {
            // Re-inserting the same pointer replaces its vector, so an update whose old
            // vector was unindexable and whose new one is fine must leave the stale
            // record behind.
            self.unindexed.remove(&ptr);
        } else {
            // Wrong dimension, a non-finite component, or a column that is not an `f32`
            // array. Keep the pointer so deletion still finds it.
            self.index.remove(&ptr);
            self.unindexed.insert(ptr);
        }

        scratch.clear();
        self.scratch = scratch;
    }

    /// Removes `ptr` from the index, returning whether it was present.
    pub fn delete(&mut self, ptr: RowPointer) -> bool {
        // A pointer is in at most one of the two, but check both: an update can move a row
        // between them.
        let removed = self.index.remove(&ptr);
        removed | self.unindexed.remove(&ptr)
    }

    /// Empties the index, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.index.clear();
        self.unindexed.clear();
    }

    /// Returns the `k` rows whose vectors are nearest to `query`, nearest first.
    ///
    /// `keep` filters candidates by row pointer, which is how a transaction hides rows it
    /// has deleted but which are still present in the committed index.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        keep: impl Fn(&RowPointer) -> bool,
    ) -> Result<Vec<Neighbor<RowPointer>>, VectorError> {
        self.index.search_filtered(query, k, keep)
    }
}

/// Reads `row_ref`'s `cols` column into `out`, returning whether it was an `f32` array.
///
/// `out` is left empty when this returns `false`.
fn read_vector_into(cols: &ColList, row_ref: RowRef<'_>, out: &mut Vec<f32>) -> bool {
    out.clear();
    // `project` is the safe path: it bounds-checks the column and decodes the value. An
    // index over a vector column is only ever created after schema validation has checked
    // the column's type, so the `else` branches below are unreachable in practice — but
    // being wrong here would corrupt search results rather than crash, so they are handled
    // rather than asserted.
    let Ok(value) = row_ref.project(cols) else {
        return false;
    };
    let AlgebraicValue::Array(ArrayValue::F32(components)) = value else {
        return false;
    };
    out.reserve(components.len());
    // `F32` is `decorum::Total<f32>`, a `repr(transparent)` wrapper. Copying component by
    // component keeps this free of transmutes; it is a single pass over contiguous memory.
    out.extend(components.iter().map(|c| c.into_inner()));
    true
}

impl MemoryUsage for VectorTableIndex {
    fn heap_usage(&self) -> usize {
        let Self {
            index,
            unindexed,
            dimension: _,
            metric: _,
            strategy: _,
            // Transient working space, not part of the index's contents.
            scratch: _,
        } = self;
        index.heap_usage() + unindexed.heap_usage()
    }
}

impl fmt::Debug for VectorTableIndex {
    /// Summarises rather than dumping every vector, which would be megabytes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.index.stats();
        f.debug_struct("VectorTableIndex")
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("approximate", &stats.approximate)
            .field("vectors", &stats.vectors)
            .field("unindexed", &self.unindexed.len())
            .finish()
    }
}

impl PartialEq for VectorTableIndex {
    /// Compares configuration and contents, ignoring how the contents are arranged.
    ///
    /// Two approximate indexes holding the same vectors are equal even if their graphs
    /// differ, which is the useful notion: the graph is an accelerator, not data.
    fn eq(&self, other: &Self) -> bool {
        if (self.dimension, self.metric, self.strategy) != (other.dimension, other.metric, other.strategy)
            || self.index.len() != other.index.len()
            || self.unindexed != other.unindexed
        {
            return false;
        }
        self.index.payloads().all(|ptr| {
            // Compare by bits: these are stored values being checked for identity, not
            // measurements being checked for closeness.
            match (self.vector_of(ptr), other.vector_of(ptr)) {
                (Some(a), Some(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                _ => false,
            }
        })
    }
}

impl Eq for VectorTableIndex {}

impl VectorTableIndex {
    /// The vector indexed under `ptr`, if any.
    fn vector_of(&self, ptr: RowPointer) -> Option<&[f32]> {
        match &self.index {
            VectorIndex::Exact(i) => i.get(&ptr),
            VectorIndex::Hnsw(i) => i.get(&ptr),
        }
    }
}
