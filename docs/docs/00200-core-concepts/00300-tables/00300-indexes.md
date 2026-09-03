---
title: Indexes
slug: /tables/indexes
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CppModuleVersionNotice } from "@site/src/components/CppModuleVersionNotice";


Indexes accelerate queries by maintaining sorted data structures alongside your tables. Without an index, finding rows that match a condition requires scanning every row. With an index, the database locates matching rows directly.

## When to Use Indexes

Add an index when you frequently query a column with equality or range conditions. Common scenarios include:

- **Filtering by foreign key**: A `player_id` column in an inventory table benefits from an index when you query items belonging to a specific player.
- **Range queries**: An `age` column benefits from an index when you query users within an age range.
- **Sorting**: Columns used in ORDER BY clauses benefit from indexes that maintain sort order.

Indexes consume additional memory and slow down inserts and updates, since the database must maintain the index structure. Add indexes based on your actual query patterns rather than speculatively.

Primary keys and unique constraints automatically create indexes. You do not need to add a separate index for columns that already have these constraints.

## Index Types

SpacetimeDB supports three index types:

| Type | Use Case | Key Types | Multi-Column |
|------|----------|-----------|--------------|
| B-tree | General purpose | Any | Yes |
| Direct | Dense integer sequences | `u8`, `u16`, `u32`, `u64` | No |
| Vector | Similarity search over embeddings | `Vec<f32>` | No |

B-tree and direct indexes answer *"which rows have this key?"*. A vector index answers a
different question — *"which rows are most similar to this one?"* — and is covered in
[Vector Indexes](#vector-indexes) below.

### Supported Column Types

Not all column types can be used as index keys. The following types are supported for B-tree indexes:

| Category | Types |
|----------|-------|
| Integers | `u8`, `u16`, `u32`, `u64`, `u128`, `u256`, `i8`, `i16`, `i32`, `i64`, `i128`, `i256` |
| Boolean | `bool` |
| Strings | `String` |
| Identifiers | `Identity`, `ConnectionId`, `Uuid`, `Hash` |
| Enums | No-payload (C-style) enums annotated with `#[derive(SpacetimeType)]` |

The following types are **not** supported as index keys:

| Type | Reason |
|------|--------|
| `f32`, `f64` | Floating-point values do not have a total ordering (`NaN` is not comparable) |
| `ScheduleAt`, `TimeDuration`, `Timestamp` | Not yet supported ([#2650](https://github.com/clockworklabs/SpacetimeDB/issues/2650)) |
| `Vec<T>`, arrays | Variable-length collections are not indexable as *keys*. `Vec<f32>` can be indexed for [similarity search](#vector-indexes). |
| Enums with payloads | Only no-payload (C-style) enums are supported |
| Nested structs | Product types cannot be used as index keys |

If you attempt to use an unsupported type as an index key, you will get a compile error. For multi-column indexes, every column in the index must use a supported type.

:::tip Workaround for floating-point data
If you need to index floating-point coordinates (for example, `x` and `y` positions), consider storing them as scaled integers. For instance, multiply by 1000 and store as `i32` to get three decimal places of precision while remaining indexable.
:::

Direct indexes have additional restrictions: only `u8`, `u16`, `u32`, `u64`, and no-payload enums are supported.

### B-tree Indexes

B-trees maintain data in sorted order, enabling both equality lookups (`x = 5`) and range queries (`x > 5`, `x BETWEEN 1 AND 10`). The sorted structure also supports prefix matching on multi-column indexes. B-tree is the default and most commonly used index type.

### Direct Indexes

Direct indexes use array indexing instead of tree traversal, providing O(1) lookups for unsigned integer keys. SpacetimeDB uses the key value directly as an array offset, eliminating the need to search through a tree structure.

Direct indexes perform well when:
- Keys are dense (few gaps between values)
- Keys start near zero
- Insert patterns are sequential rather than random

Direct indexes perform poorly when:
- Keys are sparse (large gaps between values)
- The first key inserted is a large number
- Insert patterns are highly random

Direct indexes only support single-column indexes on unsigned integer types. Use them for auto-increment primary keys or other dense sequential identifiers where you need maximum lookup performance.

:::note
Direct indexes are currently available in Rust and TypeScript. C# support is planned.
:::

<Tabs groupId="server-language" queryString>
<TabItem value="typescript" label="TypeScript">

```typescript
const position = table(
  { name: 'position', public: true },
  {
    id: t.u32().primaryKey().index('direct'),
    x: t.f32(),
    y: t.f32(),
    z: t.f32(),
  }
);
```

</TabItem>
<TabItem value="rust" label="Rust">

```rust
#[spacetimedb::table(accessor = position, public)]
pub struct Position {
    #[primary_key]
    #[index(direct)]
    id: u32,
    x: f32,
    y: f32,
    z: f32,
}
```

</TabItem>
</Tabs>

This example from the SpacetimeDB benchmarks uses direct indexes for a million entities with sequential IDs starting at 0, enabling O(1) lookups when joining position and velocity data by entity ID.

For most use cases, B-tree indexes provide good performance without these restrictions. Consider direct indexes only when profiling reveals that index lookups are a bottleneck and your key distribution matches the ideal pattern.

## Single-Column Indexes

A single-column index accelerates queries that filter on one column. You can define the index at the field level or the table level.

### Field-Level Syntax

The field-level syntax places the index declaration directly on the column:

<Tabs groupId="server-language" queryString>
<TabItem value="typescript" label="TypeScript">

```typescript
const user = table(
  { name: 'user', public: true },
  {
    id: t.u32().primaryKey(),
    name: t.string().index('btree'),
    age: t.u8().index('btree'),
  }
);
```

</TabItem>
<TabItem value="csharp" label="C#">

:::danger Use full namespace
Never use bare `Index` — it conflicts with `System.Index`. Always write `SpacetimeDB.Index.BTree`. For table-level indexes, use `Columns = new[] { nameof(Col) }` or `new[] { "Col1", "Col2" }`, not collection expressions like `[nameof(X)]`.
:::

```csharp
[SpacetimeDB.Table(Accessor = "User", Public = true)]
public partial struct User
{
    [SpacetimeDB.PrimaryKey]
    public uint Id;

    [SpacetimeDB.Index.BTree]
    public string Name;

    [SpacetimeDB.Index.BTree]
    public byte Age;
}
```

</TabItem>
<TabItem value="rust" label="Rust">

```rust
#[spacetimedb::table(accessor = user, public)]
pub struct User {
    #[primary_key]
    id: u32,
    #[index(btree)]
    name: String,
    #[index(btree)]
    age: u8,
}
```

</TabItem>
<TabItem value="cpp" label="C++">

<CppModuleVersionNotice />

```cpp
struct User {
  uint32_t id;
  std::string name;
  uint8_t age;
};
SPACETIMEDB_STRUCT(User, id, name, age)
SPACETIMEDB_TABLE(User, user, Public)
FIELD_PrimaryKey(user, id)
FIELD_Index(user, name)
FIELD_Index(user, age)
```

Use `FIELD_Index(table, field)` to create a B-tree index on individual columns.

</TabItem>
</Tabs>

### Table-Level Syntax

The table-level syntax defines indexes separately from columns. This approach allows you to name the index explicitly:

<Tabs groupId="server-language" queryString>
<TabItem value="typescript" label="TypeScript">

```typescript
const user = table(
  {
    name: 'user',
    public: true,
    indexes: [
      { accessor: 'idx_age', algorithm: 'btree', columns: ['age'] },
    ],
  },
  {
    id: t.u32().primaryKey(),
    name: t.string(),
    age: t.u8(),
  }
);
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
[SpacetimeDB.Table(Accessor = "User", Public = true)]
[SpacetimeDB.Index.BTree(Accessor = "idx_age", Columns = new[] { "Age" })]
public partial struct User
{
    [SpacetimeDB.PrimaryKey]
    public uint Id;

    public string Name;

    public byte Age;
}
```

</TabItem>
<TabItem value="rust" label="Rust">

```rust
#[spacetimedb::table(accessor = user, public, index(accessor = idx_age, btree(columns = [age])))]
pub struct User {
    #[primary_key]
    id: u32,
    name: String,
    age: u8,
}
```

</TabItem>
</Tabs>

## Multi-Column Indexes

A multi-column index (also called a composite index) spans multiple columns. The index maintains rows sorted by the first column, then by the second column within equal values of the first, and so on.

Multi-column indexes support:
- **Full match**: Queries that specify all indexed columns
- **Prefix match**: Queries that specify the leftmost columns in order
- **Range on trailing column**: A prefix of equality conditions followed by a range on the next column

A multi-column index on `(player_id, level)` accelerates these queries:
- `player_id = 123` (prefix match on first column)
- `player_id = 123 AND level = 5` (full match)
- `player_id = 123 AND level > 5` (prefix match with range)

The same index does not accelerate a query on `level` alone, since `level` is not a prefix of the index.

<Tabs groupId="server-language" queryString>
<TabItem value="typescript" label="TypeScript">

```typescript
const score = table(
  {
    name: 'score',
    public: true,
    indexes: [
      { accessor: 'by_player_and_level', algorithm: 'btree', columns: ['player_id', 'level'] },
    ],
  },
  {
    player_id: t.u32(),
    level: t.u32(),
    points: t.i64(),
  }
);
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
[SpacetimeDB.Table(Accessor = "Score", Public = true)]
[SpacetimeDB.Index.BTree(Accessor = "by_player_and_level", Columns = new[] { "PlayerId", "Level" })]
public partial struct Score
{
    public uint PlayerId;
    public uint Level;
    public long Points;
}
```

</TabItem>
<TabItem value="rust" label="Rust">

```rust
#[spacetimedb::table(accessor = score, public, index(accessor = by_player_and_level, btree(columns = [player_id, level])))]
pub struct Score {
    player_id: u32,
    level: u32,
    points: i64,
}
```

</TabItem>
<TabItem value="cpp" label="C++">

```cpp
struct Score {
  uint32_t player_id;
  uint32_t level;
  int64_t points;
};
SPACETIMEDB_STRUCT(Score, player_id, level, points)
SPACETIMEDB_TABLE(Score, score, Public)
FIELD_NamedMultiColumnIndex(score, by_player_and_level, player_id, level)
```

Use `FIELD_NamedMultiColumnIndex(table, index_name, field1, field2, ...)` to create a named multi-column B-tree index.

</TabItem>
</Tabs>

## Querying with Indexes

SpacetimeDB generates type-safe accessor methods for each index. These methods accept filter arguments and return matching rows.

### Equality Queries

Pass a single value to find rows where the indexed column equals that value:

<Tabs groupId="server-language" queryString>
<TabItem value="typescript" label="TypeScript">

```typescript
// Find users with a specific name
for (const user of ctx.db.user.name.filter('Alice')) {
  console.log(`Found user: ${user.id}`);
}
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
// Find users with a specific name
foreach (var user in ctx.Db.User.Name.Filter("Alice"))
{
    Log.Info($"Found user: {user.Id}");
}
```

</TabItem>
<TabItem value="rust" label="Rust">

```rust
// Find users with a specific name
for user in ctx.db.user().name().filter("Alice") {
    log::info!("Found user: {}", user.id);
}
```

</TabItem>
<TabItem value="cpp" label="C++">

```cpp
// Find users with a specific name
for (auto user : ctx.db[user_name].filter("Alice")) {
    LOG_INFO("Found user: " + user.name);
}
```

Use the index accessor `ctx.db[index_name]` created by `FIELD_Index` to perform filtered queries.

</TabItem>
</Tabs>

### Range Queries

Pass a `Range` object to find rows where the indexed column falls within bounds. The `Range` constructor accepts `from` and `to` bounds, each specified as `{ tag: 'included', value }`, `{ tag: 'excluded', value }`, or `{ tag: 'unbounded' }`:

<Tabs groupId="server-language" queryString>
<TabItem value="typescript" label="TypeScript">

```typescript
import { Range } from 'spacetimedb/server';

// Find users aged 18 to 65 (inclusive)
for (const user of ctx.db.user.age.filter(
  new Range({ tag: 'included', value: 18 }, { tag: 'included', value: 65 })
)) {
  console.log(`${user.name} is ${user.age}`);
}

// Find users aged 18 or older (from 18 inclusive, unbounded above)
for (const user of ctx.db.user.age.filter(
  new Range({ tag: 'included', value: 18 }, { tag: 'unbounded' })
)) {
  console.log(`${user.name} is an adult`);
}

// Find users younger than 18 (unbounded below, to 18 exclusive)
for (const user of ctx.db.user.age.filter(
  new Range({ tag: 'unbounded' }, { tag: 'excluded', value: 18 })
)) {
  console.log(`${user.name} is a minor`);
}
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
// Find users aged 18 to 65 (inclusive)
foreach (var user in ctx.Db.User.Age.Filter(new Bound<byte>(18, 65)))
{
    Log.Info($"{user.Name} is {user.Age}");
}

// Find users aged 18 or older (inclusive, unbounded above)
foreach (var user in ctx.Db.User.Age.Filter(new Bound<byte>(18, byte.MaxValue)))
{
    Log.Info($"{user.Name} is an adult");
}

// Find users younger than 18 (unbounded below, to 17 inclusive)
foreach (var user in ctx.Db.User.Age.Filter(new Bound<byte>(byte.MinValue, 17)))
{
    Log.Info($"{user.Name} is a minor");
}
```

You can also use the implicit tuple conversion, like `ctx.Db.User.Age.Filter((18, byte.MaxValue))`, which is functionally identical.

</TabItem>
<TabItem value="rust" label="Rust">

```rust
// Find users aged 18 to 65 (inclusive)
for user in ctx.db.user().age().filter(18..=65) {
    log::info!("{} is {}", user.name, user.age);
}

// Find users aged 18 or older
for user in ctx.db.user().age().filter(18..) {
    log::info!("{} is an adult", user.name);
}

// Find users younger than 18
for user in ctx.db.user().age().filter(..18) {
    log::info!("{} is a minor", user.name);
}
```

</TabItem>
<TabItem value="cpp" label="C++">

```cpp
// Find users aged 18 to 65 (inclusive)
for (auto user : ctx.db[user_age].filter(range_inclusive(uint8_t(18), uint8_t(65)))) {
    // Process user
}

// Find users aged 18 or older
for (auto user : ctx.db[user_age].filter(range_from(uint8_t(18)))) {
    // Process user
}

// Find users younger than 18
for (auto user : ctx.db[user_age].filter(range_to(uint8_t(18)))) {
    // Process user
}
```

Use range query functions: `range_inclusive()`, `range_from()`, `range_to()`, and `range_to_inclusive()`. Include `<spacetimedb/range_queries.h>` for full range query support.

</TabItem>
</Tabs>

### Multi-Column Queries

For multi-column indexes, pass a tuple of values. You can specify exact values for prefix columns and optionally a range for the trailing column:

<Tabs groupId="server-language" queryString>
<TabItem value="typescript" label="TypeScript">

```typescript
import { Range } from 'spacetimedb/server';

// Find all scores for player 123 (prefix match on first column)
for (const score of ctx.db.score.by_player_and_level.filter(123)) {
  console.log(`Level ${score.level}: ${score.points} points`);
}

// Find scores for player 123 at levels 1-10 (inclusive)
for (const score of ctx.db.score.by_player_and_level.filter([
  123,
  new Range({ tag: 'included', value: 1 }, { tag: 'included', value: 10 })
])) {
  console.log(`Level ${score.level}: ${score.points} points`);
}

// Find the exact score for player 123 at level 5
for (const score of ctx.db.score.by_player_and_level.filter([123, 5])) {
  console.log(`Points: ${score.points}`);
}
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
// Find all scores for player 123
foreach (var score in ctx.Db.Score.by_player_and_level.Filter(123u))
{
    Log.Info($"Level {score.Level}: {score.Points} points");
}
```

</TabItem>
<TabItem value="rust" label="Rust">

```rust
// Find all scores for player 123 (prefix match)
for score in ctx.db.score().by_player_and_level().filter(&123u32) {
    log::info!("Level {}: {} points", score.level, score.points);
}

// Find scores for player 123 at levels 1-10
for score in ctx.db.score().by_player_and_level().filter((123u32, 1u32..=10u32)) {
    log::info!("Level {}: {} points", score.level, score.points);
}

// Find the exact score for player 123 at level 5
for score in ctx.db.score().by_player_and_level().filter((123u32, 5u32)) {
    log::info!("Points: {}", score.points);
}
```

</TabItem>
<TabItem value="cpp" label="C++">

```cpp
// Find all scores for player 123 (prefix match)
for (auto score : ctx.db[score_by_player_and_level].filter(uint32_t(123))) {
    LOG_INFO("Level " + std::to_string(score.level) + ": " + std::to_string(score.points) + " points");
}

// Find scores for player 123 at levels 1-10
for (auto score : ctx.db[score_by_player_and_level].filter(
         std::make_tuple(uint32_t(123), range_inclusive(uint32_t(1), uint32_t(10))))) {
    LOG_INFO("Level " + std::to_string(score.level) + ": " + std::to_string(score.points) + " points");
}

// Find the exact score for player 123 at level 5
for (auto score : ctx.db[score_by_player_and_level].filter(
         std::make_tuple(uint32_t(123), uint32_t(5)))) {
    LOG_INFO("Points: " + std::to_string(score.points));
}
```

</TabItem>
</Tabs>

## Deleting with Indexes

Indexes also accelerate deletions. Instead of scanning the entire table to find rows to delete, you can delete directly by index value:

<Tabs groupId="server-language" queryString>
<TabItem value="typescript" label="TypeScript">

```typescript
import { Range } from 'spacetimedb/server';

// Delete all users named "Alice"
const deleted = ctx.db.user.name.delete('Alice');
console.log(`Deleted ${deleted} user(s)`);

// Delete users younger than 18
const deletedMinors = ctx.db.user.age.delete(
  new Range({ tag: 'unbounded' }, { tag: 'excluded', value: 18 })
);
console.log(`Deleted ${deletedMinors} minor(s)`);
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
// Delete all users named "Alice"
var deleted = ctx.Db.User.Name.Delete("Alice");
Log.Info($"Deleted {deleted} user(s)");
```

</TabItem>
<TabItem value="rust" label="Rust">

```rust
// Delete all users named "Alice"
let deleted = ctx.db.user().name().delete("Alice");
log::info!("Deleted {} user(s)", deleted);

// Delete users in an age range
let deleted = ctx.db.user().age().delete(..18);
log::info!("Deleted {} minor(s)", deleted);
```

</TabItem>
</Tabs>

## Vector Indexes

A vector index turns a table of embeddings into a vector database: given a query vector, it
returns the `k` rows whose vectors are most similar, without scanning the table.

This is what powers semantic search, retrieval-augmented generation, recommendations, and
deduplication. You store the output of an embedding model in a `Vec<f32>` column, and
SpacetimeDB finds the nearest ones.

### Defining a Vector Index

<Tabs groupId="module-language">
<TabItem value="rust" label="Rust" default>

```rust
#[spacetimedb::table(
    accessor = document,
    public,
    index(accessor = by_embedding, vector(column = embedding, dimension = 768, metric = cosine))
)]
pub struct Document {
    #[primary_key]
    #[auto_inc]
    id: u64,
    embedding: Vec<f32>,
    text: String,
}
```

</TabItem>
</Tabs>

The indexed column must have type `Vec<f32>`. Three parameters configure the index:

| Parameter | Required | Meaning |
|-----------|----------|---------|
| `column` | yes | The `Vec<f32>` column to index. Exactly one; there is no meaningful way to combine similarity across several columns. |
| `dimension` | yes | How many components every vector in the column has. |
| `metric` | no | How similarity is measured. Defaults to `l2`. |

`dimension` is required because a `Vec<f32>` column cannot express its own length — the
type system has no fixed-size array — so the index has to be told, and enforces it from
then on.

:::note
Vector indexes are currently available in Rust modules only.
:::

### Choosing a Metric

Every metric is expressed as a *distance*: smaller means more similar.

| Metric | Formula | Use it when |
|--------|---------|-------------|
| `l2` (default) | `sqrt(sum((a - b)^2))` | The magnitude of an embedding carries meaning. Also called Euclidean distance. |
| `cosine` | `1 - cos(a, b)` | Comparing by direction rather than magnitude. **The usual choice for text embeddings.** |
| `dot_product` | `-(a . b)` | The model was trained with a dot-product objective. Ranks by *largest* inner product. |
| `l1` | `sum(\|a - b\|)` | Manhattan distance. |

Use the metric your embedding model was trained for. Most text embedding models (OpenAI,
Cohere, sentence-transformers) are trained for cosine similarity.

A zero vector has no direction, so its cosine distance to anything is defined as `1.0`
(orthogonal) rather than `NaN`.

### Searching

<Tabs groupId="module-language">
<TabItem value="rust" label="Rust" default>

```rust
#[spacetimedb::reducer]
pub fn find_similar(ctx: &ReducerContext, query: Vec<f32>) {
    // The 10 most similar documents, most similar first.
    for doc in ctx.db.document().by_embedding().search(&query, 10) {
        log::info!("{}", doc.text);
    }
}
```

</TabItem>
</Tabs>

`search` returns at most `k` rows, ordered nearest first, and fewer if the table holds
fewer. Rows written earlier in the same transaction are included; rows deleted in it are
not.

The query vector must have the index's declared dimension and contain only finite numbers.
A `NaN` or an infinity has no defined distance to anything, so `search` panics rather than
returning a meaningless ranking.

### Exact and Approximate Search

By default a vector index is **exact**: it compares the query against every indexed vector
and returns the true nearest neighbours, every time.

That is faster than it sounds. The vectors are stored end to end in a single allocation, so
a search is one sequential pass over contiguous memory rather than a walk over rows — a few
hundred thousand embeddings are handled in single-digit milliseconds. Start here.

When a linear pass stops fitting your latency budget, add `hnsw` to switch that index to an
approximate graph search:

<Tabs groupId="module-language">
<TabItem value="rust" label="Rust" default>

```rust
#[spacetimedb::table(
    accessor = document,
    index(accessor = by_embedding,
          vector(column = embedding, dimension = 768, metric = cosine,
                 hnsw(m = 16, ef_construction = 200, ef_search = 64)))
)]
pub struct Document { /* ... */ }
```

</TabItem>
</Tabs>

HNSW ("Hierarchical Navigable Small World") builds a layered proximity graph, turning a
linear scan into something closer to logarithmic. All three parameters are optional and
default to the values shown:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `m` | 16 | Edges kept per node per layer. Higher means better recall and more memory — roughly `2 * m * 4` bytes per vector. |
| `ef_construction` | 200 | Search width while inserting. Higher builds a better graph, more slowly. |
| `ef_search` | 64 | Search width while querying. Higher means better recall and slower queries. |

The trade-off is real: an approximate search occasionally misses a true neighbour. Typical
recall at the default settings is above 95%.

:::caution
An HNSW graph is shaped by the order rows were inserted in. Replicas replaying the same
commitlog stay in lockstep, but a graph rebuilt from a snapshot may differ from the one it
replaced, and can return a slightly different set of neighbours. Exact search has no such
caveat — its results depend only on which rows are in the table. If a reducer writes rows
derived from search results, prefer exact search.
:::

### What a Vector Index Cannot Do

A vector index has no keys, so it does not answer the queries the other index types do:

- It cannot be used for equality or range filters. An index on the same column of another
  type handles those; you can declare both.
- It cannot back a `#[unique]` constraint or a primary key.
- It is not used by SQL. Nearest-neighbour search is reached through the generated
  `search` accessor from a module.

### Rows That Cannot Be Indexed

Because a `Vec<f32>` column cannot constrain its own length, a row can be inserted carrying
a vector of the wrong dimension, or one containing `NaN`. Such a row is stored normally but
is left out of search results — a vector of a different dimension has no defined distance
to the query, so it is not a neighbour of anything.

If you want a hard failure instead, check the length before inserting.

## Index Design Guidelines

**Choose columns based on query patterns.** Index the columns that appear in your WHERE clauses and JOIN conditions. An unused index wastes memory.

**Consider column order in multi-column indexes.** Place the most selective column (the one that narrows results most) first, followed by columns used in range conditions. An index on `(country, city)` works for queries on `country` alone or `country AND city`, but not for queries on `city` alone.

**Avoid redundant indexes.** A multi-column index on `(a, b)` makes a separate index on `(a)` redundant, since the multi-column index handles prefix queries. However, an index on `(b)` is not redundant if you query `b` independently.

**Balance read and write performance.** Each index speeds up reads but slows down writes. Tables with high write volume and few reads may benefit from fewer indexes.

**Budget memory for vector indexes.** A vector index holds a copy of every embedding:
`dimension * 4` bytes per row, plus roughly `2 * m * 4` more per row if it is an HNSW
index. At 768 dimensions that is about 3 KB per row before the graph.

## Next Steps

- Learn about [Constraints](./00240-constraints.md) for primary keys and unique indexes
- See [Access Permissions](./00400-access-permissions.md) for querying tables from reducers
