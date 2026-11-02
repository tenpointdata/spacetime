# Engineering guide — spacetime

Upstream SpacetimeDB fork with five local commits adding vector search; 139 upstream-only commits at comparison.

## Fork and upstream ownership

- Reviewed source: `1100f098f1fb79bdd0bc9cb51b705a0871c7a1b1` on `master`.
- Upstream: [clockworklabs/SpacetimeDB](https://github.com/clockworklabs/SpacetimeDB).
- At the review snapshot, the fork had 5 fork-only and 139 upstream-only commits relative to `bb6eb379faeaa0a45e666042eab218097cc9e46b`. [Read the pinned comparison context](https://github.com/tenpointdata/spacetime/compare/master...clockworklabs:master). These counts are historical evidence, not a claim about today's latest release.
- Document fork-only vector contracts separately from upstream product docs. Pin upstream/fork SHAs and distinguish source tests present from tests executed. Preserve generated binding ownership and vector client-codegen exclusions.

## Setup and native verification

Rust toolchain/Cargo.lock plus root pnpm@10.16.0 and TypeScript workspace; use the exact checked-in toolchain rather than an assumed globally installed version.

| Command | Owning source | Review execution |
| --- | --- | --- |
| cargo test -p spacetimedb-vector | crates/vector/Cargo.toml and source tests | Not run |
| cargo test -p spacetimedb-codegen --test codegen vector_indexes_are_not_exposed_to_clients | crates/codegen/tests/codegen.rs:98 | Not run |
| cargo test -p spacetimedb-testing --test standalone_integration_test test_vector_search | crates/testing/tests/standalone_integration_test.rs:517 | Not run |
| pnpm lint && pnpm test && pnpm test:typecheck && pnpm build (from crates/bindings-typescript) | crates/bindings-typescript/package.json | Not run |

Native toolchains and inherited workflow enablement remain unchanged. Source-rule inspection in this review was local only; the PR does not distribute additional inspection code or introduce validation dependencies. Full inventory debt remains open. Native type/contract tests, targeted source-size review and the linked remediation issues remain the validation path; missing or incomplete checks are not a pass.

## Architecture and trust boundaries

Rust database/runtime workspace, typed SDK/codegen surfaces, WASM modules and docs; fork adds independent spacetimedb-vector engine plus storage, transaction, ABI and bindings adapters.

Enumerated auth/permissions source and test surfaces; reviewed vector adapter invalid-vector boundary and client-codegen exclusion contract. No runtime/authenticated security acceptance or vulnerability-free claim. Deeper review should preserve authorization around candidate retrieval and module/transaction execution.

## Operations and release evidence

This review did not deploy, pair nodes, install engines, publish packages, change credentials, run live data queries or enable repository Actions. Preserve existing workflow restrictions. A release needs its own immutable source/artifact receipt, passed native contracts, real acceptance gates and rollback target. Upstream release text is not proof that this fork shipped that release.

## Roadmap

Use [the dated review](REVIEW-2026-11-02.md) for the consolidated issue map and inspected paths. GitHub issues own current state. Record fork patches, upstream imports and intentional divergences with their compatibility evidence instead of duplicating inconsistent checklists.
