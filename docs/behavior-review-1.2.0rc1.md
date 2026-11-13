# Review of 1.2.0rc1 against the initial request

Review scope: local candidate built over HEAD
`a407fb86a5648c62455cffad15b0e067814d8527`. PR base is the verified remote
`codex/release-v1.1.0` at that commit; remote `main` still points to release 0.3.0.
The review is isolated from the original dirty checkout. No UniversRobot changes,
publication, tags, route activation or live service calls are included.

## HEAD audit and reconciliation

In released HEAD, `CortexMux.astructured` builds `StructuredOutputRequest`, then
`Router.select` resolves explicit provider/model before routing profiles and
`Router._selected_request` merges route options only for profile-selected routes.
`MLXProvider.execute` accepts max_tokens/temperature/top_p, rejects think, and
parses structured output without portable behavior identity or private audit.
`DeepSeekProvider.execute` also rejects think. Thus the UniversRobot 1.1.0
observations are accurate for HEAD, not evidence of the subsequent dirty-tree
candidate. The local candidate had already added a Magistral parser/private audit,
but left the gaps listed below. This PR reuses the existing facade, router,
provider clients and schema validator; it does not build a parallel transport.

## Findings and corrections

| Priority | Candidate gap | Correction / evidence |
| --- | --- | --- |
| P1 | The original manifest identity bypassed qualification, including absent or failed live evidence. | Default live policy now requires sufficient passing checks for the exact identity; explicit offline policy runs fixtures and cannot bypass failed live evidence. `test_candidate_requires_explicit_offline_policy`, `test_incomplete_or_failed_live_attestations_cannot_be_used`. |
| P1 | DeepSeek `think` translation bypassed the portable registry, common diagnostics and strict JSON checks. | Both providers use `execute_behavior` around their existing clients. DeepSeek has native declarative translation and identical diagnostics. `test_controls_and_diagnostics_share_the_same_pipeline`, `test_provider_failures_have_common_private_diagnostics`, `test_schema_errors_share_safe_diagnostics`. |
| P1 | Reconstructing an HTTP response with already-decoded bytes and Content-Encoding could decode gzip twice. | Shared bounded reader returns decoded bytes once. `test_gzip_response_is_decoded_once` covers both clients. |
| P1 | OpenAPI reading was unbounded; interrupted bodies lost private evidence. | Streaming bounds also cover metadata; partial reads are retained on errors. `test_openapi_is_bounded_before_completion`, `test_partial_body_survives_transport_failure`. |
| P2 | Alias collisions and conflicting rules for the same ID/revision were not rejected. | Registry construction validates keys, providers, alias ownership and revision/rule consistency. |
| P2 | Sampling and measured reasoning overruns could bypass a profile's supported controls. | Explicit sampling whitelist; reported reasoning counts above requested budgets fail. `test_sampling_override_cannot_enable_unsupported_control`, `test_reported_reasoning_budget_violation_is_rejected`. |
| P2 | Syntax failures after envelope removal lost normalized text and transformations. | Envelope and parser phases are separate; private normalized text and public transformation names survive parse failures. `test_transformed_invalid_json_remains_in_private_audit`. |
| P2 | Exported fixtures checked acceptance only, not preserved expected values. | Optional expected JSON values are included in hashes and qualification; all positive bundled fixtures provide them. `test_semantic_fixture_expectation_is_checked`. |
| P2 | The initial candidate included unrelated uncommitted vision/Ollama work and inferred MLX vision support from provider name. | This focused PR excludes that work, keeps released generic capabilities and advertises only the bound profile's tasks. Original working files are preserved. `test_metadata_and_capabilities_do_not_infer_vision`. |
| P2 | Valid responses with missing counters could be counted as live budget qualification. | Separate live probes skip unavailable measurements and do not generate attestations or activate routes. |

Rules/fixtures changed, so bundled profiles advance to revision 2 instead of
reusing immutable revision 1. Public data uses Pydantic manifests, options and
diagnostics. Router selection remains unchanged. The schema is injected once and
no semantic postprocessing, implicit retry, demonstration or executable manifest
extension is introduced.

## Initial acceptance-test traceability

Numbers below refer to the 39 tests requested in the original brief. Offline
coverage proves library behavior, not server/model enforcement.

| Requested tests | Coverage / status |
| --- | --- |
| 1–5: exact identities, aliases, unknown versions, ambiguity, revisions/quantization | `test_identity_mismatch`, `test_alias_and_strict_normalization`, `test_unknown_server_version_refuses_bound_profile`, registry collision checks, and default qualification-gate regressions. |
| 6–8: requalification, no fallback, generic compatibility | `test_requalification_reuses_rules_without_implicit_inheritance`, existing MLX/DeepSeek provider tests, preflight no-I/O tests. No router/provider fallback added. |
| 9–14: explicit/routed parity, two tasks, budgets, hashes, sync/async, mandatory precedence | `test_public_explicit_and_routed_calls_keep_task_budgets`, `test_sync_public_parity`, public example, preflight controls. |
| 15–20: wire contract, omitted format, reasoning, rejection, no effort equivalence, unknown counters | `test_qwen_native_controls_without_response_format`, `test_controls_and_diagnostics_share_the_same_pipeline`, `test_preflight_never_sends_unsupported_options`. |
| 21: actual ceiling and truncation | **Not executed live.** Opt-in live cap/truncation probes provided; unknown counters remain unqualified. |
| 22–28: Magistral examples, literals, duplicates/nonfinite JSON, syntax/schema, length, no semantics or array selection | Embedded expected-value fixtures; `test_portable_profiles_and_fixtures`, failure diagnostics, relation/literal preservation, common provider failure/schema tests. |
| 29–33: diagnostics, private access, safe messages, limits, no retry | Common typed diagnostics, private-audit tests, gzip/partial/size tests, one-call assertions, response-model validation audit. |
| 34–37: round-trip, portable exports, reusable rules, non-regression | Profile round-trip/fingerprint tests, requalification tests and full offline repository suite against the focused branch. |
| 38: native qualified DeepSeek control | Registry-backed native mapping is tested offline through the public facade and existing client; **no live DeepSeek identity or qualification is claimed**. |
| 39: public consumer example | `examples/behavior_profiles.py`: two schemas/budgets, one success and one privately audited failure without monkeypatches. |

## Remaining external qualification and release gates

No approved `CORTEXMUX_BEHAVIOR_LIVE_CONFIG` or new live evidence was supplied.
MLX-VLM 0.7.11 Magistral/Qwen reasoning-budget enforcement, output-cap scope,
actual model revisions, and DeepSeek live deployment identity remain to qualify.
The runtime now refuses to treat those missing attestations as live qualification.
Attestations are reviewer/application assertions referencing private evidence,
not signatures or automatically verified external evidence.

The live probes are opt-in and synthetic. Missing counters explicitly prevent the
relevant measurement from qualifying. Model/server changes require exact new
identity plus reviewed evidence; unchanged rules keep their hash. DSPy remains
outside transport, validation, permissions and activation. UniversRobot adaptation
and TracedCortexMux live integration require a separately requested release adoption.

The initial local 549-test run included uncommitted legacy/vision/data test files
not present in HEAD. The focused PR does not delete any tracked tests; its count
is different because unrelated local experiments are not copied into the PR.
See the release note for the final focused-suite count and artifact verification.
