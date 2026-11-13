# CortexMux 1.2.0rc1 — corrected review candidate

Prepared for review on 13 November 2026. **Not published or activated.**
Base: `a407fb86a5648c62455cffad15b0e067814d8527` on
`codex/release-v1.1.0`; `main` remains on 0.3.0. This focused branch excludes the
unrelated uncommitted vision/Ollama/data experiments in the original checkout.
No UniversRobot files, dependencies or routes were modified.

The candidate adds versioned portable MLX/DeepSeek behavior contracts through the
existing facade and provider clients. Exact bindings, default live qualification
gates, explicit offline probe mode, common JSON/schema/finish checks, native
reasoning intent, bounded opt-in private evidence and typed public diagnostics
replace per-task technical adapters. No retry or semantic repair is introduced.

See [the requirement-by-requirement review](behavior-review-1.2.0rc1.md) and
[migration/requalification guide](behavior-profiles.md). The review corrected
qualification bypass, independent DeepSeek handling, gzip double decoding,
unbounded metadata reading, lost partial/normalized evidence, ambiguous bindings,
unqualified sampling and missing reasoning-budget checks.

## Manifest artifacts

Both bundled profiles are **offline candidates**, revision **2** (rules/fixtures
changed from the initial local revision 1), format version 1. They target the exact
requested MLX-VLM 0.7.11, 8-bit model identities. Model revision and output-cap scope
remain unknown. No live model/server identity was inferred.

| Profile ID | Canonical manifest SHA-256 | Rules and fixtures SHA-256 |
| --- | --- | --- |
| `mlx-vlm-0.7.11-magistral-8bit` | `7788cc7765b017c951250770e91fca0bbf36a76ffd570a2070372ad99f5eef9a` | `ca3957a9ad1a323e32ce54d4e9f4fb06923380fe2152a169b01b0560f5dc861d` |
| `mlx-vlm-0.7.11-qwen-8bit` | `78ad61198615ee87bbf90612ce55b983fc193ff172ad42afb410497f8009b48c` | `bee273cca05649515980d423f4c90f8ef6b2913a63f2b31cf083addd5e6c93b2` |

Canonical manifest hashes exclude the trailing file newline. The wheel packages
both manifests. Build-file SHA-256 checksums belong in the PR validation record,
not inside the source archive whose hash they describe. Recompute them after any
source change; a release maintainer must build from the reviewed commit.

## Validation

- Ruff check and formatting check: passed.
- Mypy: passed for all 97 source files.
- Full focused offline suite: **469 passed, 8 skipped**, with `MPLBACKEND=Agg`.
  The earlier candidate's 549-test count included unrelated local tests absent
  from HEAD; no tracked test was deleted in this PR.
- `git diff --check`: passed.
- Offline public example: success plus an audited schema failure, no HTTP patch.
- Distribution build, Twine validation and wheel manifest import are recorded
  with exact artifact hashes in the PR description after final packaging checks.

## Live trials and release gates

**Not run:** real Magistral/Qwen calls, server token-ceiling enforcement,
reasoning toggles/budget enforcement, DeepSeek live qualification, vision or
TracedCortexMux integration. Five explicit behavior probes plus three pre-existing
integration tests are skipped without approved configuration. No missing counter
is counted as a successful cap measurement. No live attestation is fabricated.

By default, these offline manifests fail before generation until adequate live
qualification is supplied. `qualification_policy="offline"` is an explicit probe
mode, visible in diagnostics. DeepSeek uses the shared registry and native
translation but needs an honest exact deployment identity and reviewed evidence;
no server version or numeric reasoning-budget support is invented.

Review, live deployment qualification, publication, and UniversRobot adoption are
separate actions. This PR creates no release, tag, automatic activation or DSPy
runtime, and does not bypass repository review.
