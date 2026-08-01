# Changelog

All notable changes follow Keep a Changelog conventions.

## [Unreleased]

### Added

- Typed synchronous and asynchronous structured-output streaming for Ollama,
  with draft-only chunks, schema-validated completion events, usage metadata,
  explicit interrupted-stream errors, and root-level `think` control.

## [0.3.0] - 2026-07-29

### Added

- Named project model profiles, selectable globally or per request, with exact
  provider/model routes by task.
- Pre-execution installed-model validation, enabled by default, raising
  `ModelNotFoundError` before a provider request is made.
- Optional typed `think` control for structured-output requests.

### Fixed

- Ollama structured requests now send `think` at the `/api/generate` payload
  root, allowing Qwen3 models to return schema-constrained JSON in `response`
  when reasoning is disabled.
- Example model profiles now use the exact installed
  `nomic-embed-text:latest` tag.

## [0.2.0] - 2026-07-27

### Added

- Opt-in bounded web-page retrieval with public-address and host-allowlist
  validation, redirect revalidation, visible-text/table parsing, and
  source-attributed structured extraction through a configured model.
- Deterministic synthetic CSV fixtures for revenue, population, weather, and
  stock-price regression scenarios, including independently verified SMA50 and
  SMA200 values.
- End-to-end local demo with generated CSV data, provider model discovery,
  Ollama `qwen2.5-coder` analysis, and a generic ComfyUI workflow.
- Reusable ComfyUI workflow catalogs discovered from versioned local manifests,
  with facade and CLI listing plus execution by workflow name.
- Typed synchronous/asynchronous ComfyUI progress callbacks covering queue,
  execution, node progress, downloads, completion, and errors.
- Independent mathematical verification for structured AI claims, using
  source-bound operands, whitelisted operations, Decimal arithmetic, configurable
  tolerances, and optional strict rejection.
- In-memory dictionary records for text and numeric data extracted by a website
  ingestion layer, without granting the analysis provider network access.

### Fixed

- Reuse one event loop across sequential synchronous facade calls so persistent
  HTTP clients remain valid for health checks, model listing, and execution.
- Fall back to the deterministic baseline when an LLM analysis plan references
  missing columns or otherwise fails safe execution validation.

## [0.1.0] - 2026-07-27

### Added

- Typed facade, schemas, registry, routing, configuration, and exception hierarchy.
- Direct Ollama and ComfyUI providers.
- Safe deterministic Pandas, optional Polars, and optional DuckDB analysis.
- CLI, documentation, tests, CI, and repository templates.
