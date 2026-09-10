# Changelog

All notable changes follow Keep a Changelog conventions.

## [Unreleased]

## [0.6.0rc1] - 2026-09-10

### Added

- Opt-in `codex` subscription provider using local App Server stdio, with
  managed ChatGPT/browser or device login, sanitized account state and limits.
- Version-pinned CLI 0.140.0 protocol, bounded process lifecycle, correlated
  notifications, cancellation, fresh contexts and explicit persistent threads.
- Codex text/chat/JSON Schema generation, local JSON validation, streaming and
  exact model/effort discovery without API-key fallback or inferred costs.
- Generic sync/async facade streams and structured completion correlation metadata.
- Offline fake-process regression coverage, opt-in live subscription test,
  isolation documentation and release-only UniversRobot example.

### Security

- Dedicated auth directory, filtered process environment, restrictive settings,
  no automatic approvals, no raw server diagnostics or token propagation.
- Strict no-tool guarantee explicitly unsupported: requests require it by default
  and fail closed unless the caller explicitly accepts sandbox limitations.

### Fixed

- Router streaming closes provider iterators when consumers stop early.

## [0.5.2] - 2026-09-07

### Added

- Typed, preflight-validated OpenAI `service_tier`, reasoning effort, verbosity,
  and output-token options, including explicit Standard/default processing.
- Bounded OpenAI transient-error retries with exponential full-jitter backoff,
  configurable limits, attempt metadata, provider request IDs, effective model
  and tier metadata, and total provider-call duration.
- Migration notes and precise documentation of the locally enforced JSON Schema
  subset.

### Changed

- GPT-6 Astra defaults to `reasoning.effort: low` and accepts only `low`,
  `medium`, `high`, `xhigh`, and `max`.
- Per-request timeouts override the configured OpenAI provider timeout for each
  attempt.

### Fixed

- Text and structured system instructions are sent through the Responses API
  `instructions` field instead of being omitted.
- JSON booleans no longer validate as `integer` or `number`, and unsupported
  JSON Schema validation keywords are no longer silently ignored.

## [0.5.1] - 2026-08-10

### Changed

- Structured model qualification now reports JSON syntax, schema conformance,
  and expected-value accuracy separately; the example suite uses three
  repetitions and an explicit deterministic trend definition.

## [0.5.0] - 2026-08-10

### Added

- Opt-in, provider-independent CapitalForge MCP stdio client with MCP 2025
  negotiation, bounded structured results, a five-tool read-only allowlist,
  child-process cleanup, and native Ollama `qwen3:4b` tool-loop support.
- Per-task model-profile options, with explicit request options taking
  precedence, plus a memory-bounded `qwen3.6:27b` quality-profile example.
- Provider-neutral model qualification with deterministic text/JSON cases,
  machine and memory metadata, fast/balanced/quality scoring, and atomic
  JSON or optional YAML recommendation manifests.
- Explicit opt-in OpenAI Responses API provider with authenticated model
  discovery and text, chat, and structured-output normalization.
- `cortexmux models qualify` command and a mixed Ollama/GPT-5.6 example suite.

### Changed

- Example fast and vision routes now use `qwen3:4b` and
  `ministral-3:8b`, removing the retired Llama 3.2 and Gemma 3 defaults.

### Fixed

- The `cortexmux chat` command now uses the facade's chat-message
  normalization instead of constructing an invalid raw chat request.

## [0.4.0] - 2026-08-01

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
