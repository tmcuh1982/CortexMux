# MLX-VLM 0.7.11 live results and remaining limits

Updated 14 November 2026 for PR #5. These are synthetic qualification probes of
candidate `d6d8d22551cb84bfb30544ac954ca360e96cd0e2`, not production business data.
No UniversRobot files, dependencies, services or routes were changed. No profile
was activated or complete live attestation added. Raw replies, reasoning text,
private selectors, host addresses and credentials are not included here.

## Evidence and provenance

- [Initial 14-call report and response hashes](https://github.com/tmcuh1982/CortexMux/pull/5#issuecomment-6864672771).
- [Official 0.7.11 source inspection](https://github.com/tmcuh1982/CortexMux/pull/5#issuecomment-6875460563),
  upstream commit `527c513a0768245644a70266a22126d880767b18`.
- [Seven logprobs probes and response hashes](https://github.com/tmcuh1982/CortexMux/pull/5#issuecomment-6880046895).

The initial campaign ran on 13 November; the complementary probes ran on
14 November after the owner supplied the server's changed address. Earlier failed
connection attempts executed no generations and are not included in these counts.
A temporary SSH tunnel was closed after testing. OpenAPI reported **0.7.11**;
returned selectors matched the explicitly requested selectors on every call.
That confirms neither the installed source SHA nor the actual model weight
revision/quantization. The 8-bit identities remain deployment declarations.

## Initial public-facade campaign: 14 calls

Magistral and Qwen each passed four structured probes (base structured output,
thinking disabled, thinking enabled, requested thinking budget 128). All eight
outputs passed schema and independent value verification. Each model also reached
requested output caps **1, 64 and 512** exactly on truncation probes; all six
truncations were correctly rejected through the public facade.

| Model | Structured | Thinking off | Thinking on | Budget 128 |
| --- | ---: | ---: | ---: | ---: |
| Magistral | 14 | 16 | 16 | 17 |
| Qwen | 58 | 9 | 83 | 142 |

Numbers are reported total completion tokens, not separate reasoning counts.
Qwen's budget-128 probe exposed reasoning text but no reasoning-token count.
Magistral exposed no separate reasoning text; its JSON fences were normalized.
These probes do not establish Magistral's reasoning toggle or budget enforcement.

## Complementary protocol diagnostics: seven calls

These use the existing MLX client, non-streaming `logprobs=true`, no demonstrations
and no `response_format`. They are protocol diagnostics, not additional public
BehaviorProfile attestations. Synthetic task: sum primes strictly below 100 and
return JSON; independently calculated expected answer **1060**.

| Model | Thinking budget | Output cap | Generated tokens | Logprob entries | Entries before closing marker | Finish | Answer verified |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen | 8 | 512 | 21 | 20 | 10 | stop | yes |
| Qwen | 32 | 512 | 45 | 44 | 34 | stop | yes |
| Qwen | 128 | 512 | 141 | 140 | 130 | stop | yes |
| Qwen | disabled | 512 | 10 | 9 | none | stop | yes |
| Magistral | 8 | 512 | 17 | 16 | none | stop | yes |
| Qwen | 128 | 64 | 64 | 64 | none | length | no, truncated |
| Magistral | 128 | 64 | 17 | 16 | none | stop | yes |

For Qwen budgets 8/32/128, the generated sequence has no opening marker and its
first `</think>` occurs at zero-based index 10/34/130. Concatenating the preceding
entries matches returned `reasoning_content` after trimming edge whitespace.
These counts include the last newline entry and exclude the closing delimiter.
Excluding that newline yields **9/33/129** entries. This is consistent with the
upstream `count > budget` transition followed by forced newline and closure,
not a universal guarantee that all templates/backends add exactly two tokens.

For the six natural stops, logprob entries equal generated tokens minus one,
consistent with exclusion of the terminal stop token. For truncated Qwen, all
64 entries match the returned reasoning text; there is no closing marker or final
answer. Magistral's cap-64 call finished naturally and is not a positive truncation
test; its positive truncation tests belong to the initial campaign.

## Budget semantics and measurement limits

`max_tokens` caps the entire generated sequence, including reasoning, final answer
and generated control tokens; input tokens are separate. `thinking_budget` triggers
transition out of recognized reasoning after its counter exceeds the budget. It
adds no extra allowance above the global cap. A cap of 64 can therefore terminate
reasoning before a requested 128-token transition or any final answer.

The source inspection explains this behavior; the live probes confirm the cases
above. The internal metric named `reasoning_tokens` on the inspected Chat
Completions path derives from total-generation versus completion adjustments and
is not a reliable reasoning-body count. In these probes, `completion_tokens`
equals `timings.predicted_n` despite Qwen's nonempty reasoning. Their difference
cannot measure it. Do not interpret an absent count as zero.

Logprobs allow a defined reasoning-segment measurement on the tested Qwen path.
They do not yet provide a public CortexMux metric, automatic attestation or
cross-backend guarantee. Retokenizing returned text is a reconstructed text count,
not necessarily the original generated-token count. Server timing/cache details
remain in private evidence where not normalized by public diagnostics.

The values 512/128 are test settings, not hardware/model maxima. Choose a total cap
that covers expected reasoning, closure and final output with margin, within model
context and resource limits. Local budgets bound compute, duration and memory;
they do not reserve a separate final-answer allowance.

## Remaining qualification and adoption work

- Magistral reasoning budget and effective template/delimiters: unqualified; no
  visible reasoning block proves neither enforcement nor absence of internal work.
- Actual model weight revisions/quantization and installed server source SHA:
  not independently verified.
- DeepSeek live identity/control qualification, vision and TracedCortexMux live
  integration: not executed. Offline DeepSeek mapping tests are separate evidence.
- One sample per complementary configuration: no statistical quality or performance
  guarantee. No server instrumentation or template changes were made.
- Bundled revision-2 manifests remain offline candidates with unchanged hashes and
  unknown output-cap scope in their stored declarations. Reviewed evidence and
  appropriate identity/metric semantics are required before live attestation.
- UniversRobot adoption and removal of its shims require a separate request and
  integration qualification; release publication does not perform that migration.

The [release record](release-1.2.0rc1.md) identifies the separately published
GitHub prerelease and original artifact hashes. This PR documentation follow-up
does not move the published tag or replace its assets. The [review matrix](behavior-review-1.2.0rc1.md#nine-requested-deliverables)
accounts for all nine requested deliverables without claiming full live certification.
