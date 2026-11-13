# Behavior profiles — 1.2.0rc1 review candidate

A routing `ModelProfile` chooses a provider/model. A `BehaviorProfile` specifies how
that exact deployment is called and validated **after routing**. Existing
`structured` / `astructured` interfaces, the shared synchronous Runner and async
cleanup remain in place. No business engine calls a new transport.

The registry is instance-scoped in `providers.mlx[_lm|_vlm].behavior_profiles` or
`providers.deepseek.behavior_profiles`, keyed by canonical model ID. Both explicit
and routing-profile requests use the same provider-owned client and common behavior
pipeline. Task budgets remain request options. Profiles currently qualify structured
output; generic text/chat and other provider APIs keep their 1.1.0 behavior.
MLX vision is not introduced by this PR or inferred from a server/model name.

## Activation and qualification

Bundled `magistral` and `qwen` manifests target the exact supplied model IDs,
MLX-VLM 0.7.11 and 8-bit quantization. They are **offline candidates**, not live
certifications. Revision 2 replaces the initial local candidate's revision 1 after
rules/fixtures changed during review. Model revisions and output-cap scope are
unknown (`null` / `unknown`); no version is invented.

Bindings require adequate live evidence by default. A live attestation must name
the exact deployment identity and current rules/fixture hash, pass, and cover
`identity`, `structured_output`, `max_tokens`, plus `reasoning` / `reasoning_budget`
when enabled by the rules. Missing, failed, incomplete or contradictory live
attestations fail before generation. An unknown configured server version fails.

For unit tests or explicitly approved qualification probes only, choose
`qualification_policy="offline"`. This requires the original manifest identity,
passing embedded syntax/value fixtures and no failed/incomplete live evidence.
It never silently authorizes a new server/model revision. Runtime diagnostics
expose `qualification_kind="offline"` so applications can reject such results.
Importing a manifest alone never activates it or contacts a server.

```python
from cortexmux import CortexMux
from cortexmux.behavior import BehaviorBinding, bundled_profile
from cortexmux.core.config import CortexMuxConfig

profile = bundled_profile("magistral")  # or "qwen"
binding = BehaviorBinding(
    profile=profile,
    identity=profile.identity,
    qualification_policy="offline",  # explicit probe mode; never a live attestation
)
config = CortexMuxConfig.model_validate(
    {
        "providers": {
            "ollama": {"enabled": False},
            "comfyui": {"enabled": False},
            "mlx_vlm": {
                "enabled": True,
                "base_url": "http://127.0.0.1:8080/v1",
                "server_version": "0.7.11",  # configuration declaration
                "check_openapi_version": True,
                "behavior_profiles": {profile.identity.model: binding},
            },
        },
    }
)
with CortexMux(config) as mux:
    result = mux.structured(
        "Return a JSON object with answer='ok'.",
        provider="mlx_vlm",
        model=profile.identity.model,
        behavior_profile=f"{profile.profile_id}@{profile.revision}",
        json_schema={"type": "object", "required": ["answer"]},
        max_tokens=512,
        think=True,
        thinking_budget=128,
        timeout=120,
        required_capabilities=["structured_output", "reasoning_budget"],
        normalization="profile",
        capture_private=True,
    )
```

Existing endpoint approval rules still apply. No remote host is enabled by a
profile. `selector` and `reported_selectors` are private, explicit binding aliases;
they never appear in portable manifests or public diagnostics. Conflicting aliases,
wrong registry keys/providers, or different rules sharing one profile ID/revision
are rejected. Snapshot-content verification remains the deployment owner's job.

`identity_source` records the application's declared/observed identity provenance;
its default is `declared`. Missing model revision or quantization remains unknown.
Optional OpenAPI checking records an actually observed version separately and
rejects a contradiction. Unknown observation stays `None`; a configured declaration
is not rewritten as a detected version. Exact reported aliases are private; public
reported model identity is canonical with an explicit alias flag.

## Controls and syntax contract

Request options override routing defaults, then mandatory profile constraints apply.
`BehaviorOptions` is the typed whitelist. `max_tokens` is required. MLX `think`
maps to `enable_thinking`; `thinking_budget` is forwarded only if rules support it
and `think=True`. `reasoning_effort` is never converted to a numeric budget.
Unsupported required capabilities, sampling controls or parameters fail before I/O.
Sampling must be explicitly listed in `rules.sampling`; bundled candidates list none.

A missing counter is unknown, not zero or proof of enforcement. Known completion
counts beyond the requested ceiling and reasoning counts beyond the requested
budget are rejected. Both bundled output-cap scopes are `unknown`; live tests must
establish whether the ceiling includes reasoning before that rule is revised.

`normalization="profile"` permits only the declared complete outer THINK prefix
and/or one JSON fence. `normalization="strict"` disables both but never relaxes
finish, object-root, JSON or schema validation. JSON strings, citations and values
are not rewritten. Duplicate keys, nonfinite numbers, incomplete/nested prefixes,
multiple objects, non-object roots and non-`stop` finishes fail. `length` is refused
even with a closed valid object. Relations arrays remain arrays; no `[0]` selection
or business interpretation occurs. Each structured request receives one schema
instruction, no demonstration and no `response_format` from this pipeline.

## DeepSeek without per-task shims

DeepSeek uses the **same registry, typed controls, strict parser and diagnostics**.
Create a reviewed `BehaviorProfile(provider="deepseek", identity=..., rules=...)`
with `BehaviorRules(reasoning="deepseek", reasoning_budget=False)` and bind it in
`providers.deepseek.behavior_profiles`. Single-component model IDs such as
`deepseek-chat` are supported. The server version must be supplied by an honest
qualified deployment declaration/observation; CortexMux does not claim the public
API exposes the underlying server version. There is no bundled live DeepSeek
attestation because no such deployment identity/evidence was supplied.

Then public `think=True/False` maps to `thinking.type=enabled/disabled`, following
the [DeepSeek API contract](https://api-docs.deepseek.com/guides/thinking_mode/).
Numeric reasoning budgets are rejected for DeepSeek. Thinking combined with
sampling overrides is rejected. No per-task provider registration or HTTP patch
is needed: configure the binding once. Unbound calls preserve 1.1.0 behavior,
including rejection of unsupported reasoning controls; they do not silently gain
a profile or a qualification.

## Public diagnostics and private audit

Read `response.raw_metadata["behavior"]` on success and
`error.context["behavior"]` on profile pipeline failures. Values validate as
`BehaviorDiagnostic` in `cortexmux.core.behavior_runtime`. They include profile ID,
revision, both hashes, canonical identities/provenance, qualification kind,
requested and transmitted controls, finish, normalized usage, reasoning count,
transformations, local elapsed time and failure category. Unavailable counters and
server latency remain unknown. Errors before provider execution retain the router's
existing safe diagnostics; routing metadata on success retains original/selected
provider/model. Unsupported/unvalidated arbitrary options are not echoed.

`capture_private=True` enables `response.private_audit` / `error.private_audit`.
This contains raw decoded HTTP bytes, status, relevant version headers, exact
reported model and normalized text when envelope normalization succeeds, including
JSON syntax/schema failures. Authorization headers are never retained. Raw and
normalized content are excluded from public diagnostic objects and exception text.
The response itself still contains its valid structured business result.

Bodies are bounded during streaming to 2,000,000 decoded bytes, including OpenAPI
metadata. Oversize replies fail; partial bodies survive transport errors. Compressed
responses are decoded once. A finish reason unavailable in an oversized or partial
body remains unknown. Private content belongs in application-owned secure storage
with application retention rules; CortexMux neither persists it nor logs it.

The [runnable offline example](../examples/behavior_profiles.py) shows two schemas,
two budgets, a success and an audited schema error through public facade calls,
without monkeypatches. TracedCortexMux can keep ownership of persistence, suspensions
and execution windows while consuming these diagnostics.

## Portability and migration

`BehaviorProfile.export_json()` / `.import_json(document)` preserve format version,
identity, revision, attestations, synthetic fixtures and hashes. Import is bounded
and strict, rejects duplicate keys and cannot execute code. Endpoint URLs, API keys,
headers and local paths have no manifest fields. Review fixture prose as synthetic
and non-sensitive before export; a schema cannot detect all secrets inside prose.

`rules_fingerprint` hashes rules plus fixtures, including expected semantic values.
`fingerprint` hashes canonical full-manifest JSON including attestations. Frozen
Pydantic records prevent ordinary mutation; never bypass validation using unchecked
`model_copy(update=...)` for imported data. Use a new revision for changed rules or
fixtures. An updated attestation changes the manifest hash but not the rules hash.

The initial preparation manifest is not accepted verbatim: map `match` to `identity`,
retain its profile ID if externally pinned, map its THINK/fence permissions to rules,
and attach synthetic fixtures. The endpoint/method/required ceiling/omitted format
are implemented by the adapter, not downloaded executable transport instructions.
The bundled IDs are explicit shorter IDs and must not silently replace a pinned ID.
The uncommitted legacy `MLXProfileBinding`/vision experiment from the original local
tree is not part of released 1.1.0 or this PR; migrate that experiment explicitly to
`BehaviorBinding` and the reviewed manifest. Other uncommitted local work is untouched.

After a separately reviewed release adoption, remove MLX parameter injection,
DeepSeek per-task adapters and technical format guides. Preserve UniversRobot's
names, units, citations, product attribution, relation cardinality/IDs, payload
codes, gallery advice, human validation and merge decisions. Those are business
contracts, not model-profile rules. No UniversRobot route changes accompany this PR.

## Requalification

1. Record the exact new server/model/revision/quantization and identity provenance.
   Never reuse a qualification just because a model name looks similar.
2. Run `qualify_syntax(profile)` and offline transport/schema regression tests.
   This produces only an offline syntax/value attestation.
3. For approved real probes, explicitly bind the candidate's exact identity in
   offline mode and point `CORTEXMUX_BEHAVIOR_LIVE_CONFIG` to its config. Run
   `pytest -q tests/integration/test_behavior_profiles_live.py` for each deployment.
   Changed identities need a new candidate manifest identity, preserving rules/hash;
   the production binding is not silently updated.
4. Inspect private evidence for exact identity, finish, JSON/schema, maximum output,
   reasoning enabled/disabled and the 128-token budget where applicable. Missing
   completion/reasoning counters skip the relevant measurement and do not attest it.
   Determine output-cap scope separately; accepted HTTP parameters alone are not proof.
5. Review and append a `Qualification(kind="live", passed=True, checks=(...),
   identity=new_identity, rules_fingerprint=..., evidence="reviewed-evidence-id")`.
   Rebuild using `BehaviorProfile.model_validate(...)`, export, and bind the new
   identity with the default live policy. Unchanged rules/fixtures retain their hash;
   changed rules require a new revision. Failed evidence cannot be bypassed by
   selecting offline mode.

Attestations are explicit reviewer/application assertions, not cryptographic
signatures or automatically verified external reports. No test publishes or
activates a profile. DSPy remains a separate benchmark tool: pin program, dataset,
server/model/profile identities and separate raw scores from Python-validated
scores. No runtime DSPy dependency or route authority is introduced.
