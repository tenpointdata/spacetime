//! A worked example of using SpacetimeDB as a vector database.
//!
//! This module stores documents alongside their embeddings and answers "which documents
//! are most similar to this one?" without scanning the table. It is also the module driven
//! by the vector-search integration test in `crates/testing`.
//!
//! The embeddings here are four-dimensional and written by hand so the expected answers
//! are obvious. A real module would store the output of an embedding model — 768 or 1536
//! dimensions is typical — and the only thing that would change is the `dimension` on the
//! index.

use spacetimedb::{log, table, ReducerContext, Table};

/// A document and the embedding that represents its meaning.
///
/// The vector index is declared on `embedding`. `metric = cosine` is the usual choice for
/// text embeddings, which encode meaning in a vector's *direction* rather than its
/// magnitude: two documents about the same subject point the same way whether or not one
/// is longer than the other.
///
/// `dimension` is mandatory. A `Vec<f32>` column says nothing about its length, so the
/// index has to be told, and it enforces the declared length from then on.
#[table(
    accessor = document,
    public,
    index(accessor = by_embedding, vector(column = embedding, dimension = 4, metric = cosine))
)]
pub struct Document {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    /// The document's embedding. Every row's must have exactly 4 components.
    pub embedding: Vec<f32>,
    pub text: String,
}

/// The same data indexed for approximate search.
///
/// `hnsw` builds a proximity graph instead of scanning: on a large collection it turns a
/// linear pass into something closer to logarithmic, at the cost of occasionally missing a
/// true neighbour. Exact search — the default — is the right starting point; reach for
/// this once a linear pass stops fitting the latency budget.
///
/// `l2` (Euclidean distance) suits embeddings whose magnitude carries meaning.
#[table(
    accessor = point,
    public,
    index(accessor = by_position, vector(column = position, dimension = 2, metric = l2, hnsw(m = 16)))
)]
pub struct Point {
    #[primary_key]
    pub id: u64,
    pub position: Vec<f32>,
}

/// Stores a document and its embedding.
#[spacetimedb::reducer]
pub fn add_document(ctx: &ReducerContext, text: String, embedding: Vec<f32>) {
    ctx.db.document().insert(Document {
        id: 0,
        embedding,
        text,
    });
}

/// Logs the `k` documents most similar to `query`, most similar first.
#[spacetimedb::reducer]
pub fn search_documents(ctx: &ReducerContext, query: Vec<f32>, k: u32) {
    for doc in ctx.db.document().by_embedding().search(&query, k) {
        log::info!("{}", doc.text);
    }
}

/// Deletes a document by id, demonstrating that the index keeps up with writes.
#[spacetimedb::reducer]
pub fn delete_document(ctx: &ReducerContext, id: u64) {
    ctx.db.document().id().delete(&id);
}

#[spacetimedb::reducer]
pub fn add_point(ctx: &ReducerContext, id: u64, position: Vec<f32>) {
    ctx.db.point().insert(Point { id, position });
}

/// Logs the ids of the `k` points nearest to `query`, via the approximate index.
#[spacetimedb::reducer]
pub fn search_points(ctx: &ReducerContext, query: Vec<f32>, k: u32) {
    let ids: Vec<u64> = ctx
        .db
        .point()
        .by_position()
        .search(&query, k)
        .map(|point| point.id)
        .collect();
    log::info!("{ids:?}");
}

/// Searches from *within* the same transaction that wrote the rows, which must see them.
#[spacetimedb::reducer]
pub fn add_then_search(ctx: &ReducerContext, text: String, embedding: Vec<f32>, query: Vec<f32>) {
    ctx.db.document().insert(Document {
        id: 0,
        embedding,
        text,
    });
    for doc in ctx.db.document().by_embedding().search(&query, 1) {
        log::info!("nearest after insert: {}", doc.text);
    }
}
