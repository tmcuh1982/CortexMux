# Changelog

All notable changes follow Keep a Changelog conventions.

## [Unreleased]

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
