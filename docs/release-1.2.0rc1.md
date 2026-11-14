# CortexMux 1.2.0rc1 — corrected review candidate

Prepared for review on 13 November 2026; updated 14 November 2026.
**Published as a GitHub prerelease; no profile activated.**
[Release v1.2.0rc1](https://github.com/tmcuh1982/CortexMux/releases/tag/v1.2.0rc1)
points to `d6d8d22551cb84bfb30544ac954ca360e96cd0e2`. PR #5 remains open.
This documentation follow-up does not move that tag or replace its artifacts;
the documents bundled in those original archives predate the live-results update.
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
requested MLX-VLM 0.7.11, 8-bit model identities. Model revisions and weight
quantization remain unverified. The manifests retain their original unknown
output-cap scope; live/source investigation now establishes total-generation scope
for the tested MLX-VLM path, as detailed in the [live report](behavior-live-1.2.0rc1.md).
No manifest or attestation was changed by that investigation.

| Profile ID | Canonical manifest SHA-256 | Rules and fixtures SHA-256 |
| --- | --- | --- |
| `mlx-vlm-0.7.11-magistral-8bit` | `7788cc7765b017c951250770e91fca0bbf36a76ffd570a2070372ad99f5eef9a` | `ca3957a9ad1a323e32ce54d4e9f4fb06923380fe2152a169b01b0560f5dc861d` |
| `mlx-vlm-0.7.11-qwen-8bit` | `78ad61198615ee87bbf90612ce55b983fc193ff172ad42afb410497f8009b48c` | `bee273cca05649515980d423f4c90f8ef6b2913a63f2b31cf083addd5e6c93b2` |

Canonical manifest hashes exclude the trailing file newline. The wheel packages
both manifests. Published distribution hashes, verified against GitHub asset digests:

```text
5243bbdeb951a9f870ae42743407a21f6f0929d843710ebc321bc8c9680a64f5  cortexmux-1.2.0rc1-py3-none-any.whl
8c46ca73e13be0ab33270bd8955691607f291b5500a07c083eaffa80b5ac1756  cortexmux-1.2.0rc1.tar.gz
```

These hashes identify the original candidate distributions, not a future build of
this documentation follow-up. No PyPI publication was performed.

## Validation of the published candidate

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

[Detailed live results](behavior-live-1.2.0rc1.md): 14 public-facade calls produced
eight verified successes and six expected truncation rejections. Seven additional
client-level logprobs diagnostics measured Qwen reasoning segments and confirmed
reasoning-only truncation at a total cap of 64. OpenAPI reported 0.7.11 and model
selectors matched on all calls. The eight skipped tests in the offline suite are
not a claim that these separate live campaigns were absent.

Remaining: Magistral reasoning-budget enforcement, independently verified model
revisions/quantization, DeepSeek live qualification, vision and TracedCortexMux
integration. No complete live attestation or new public reasoning metric was added.

By default, these offline manifests fail before generation until adequate live
qualification is supplied. `qualification_policy="offline"` is an explicit probe
mode, visible in diagnostics. DeepSeek uses the shared registry and native
translation but needs an honest exact deployment identity and reviewed evidence;
no server version or numeric reasoning-budget support is invented.

Review, live deployment qualification, publication, and UniversRobot adoption are
separate actions. The separately authorized GitHub prerelease does not merge PR #5,
activate profiles, add a DSPy runtime or update UniversRobot.
