# Model qualification

CortexMux can benchmark complete provider/model/options configurations and
write a portable JSON or YAML recommendation manifest for one machine. The
qualifier is a service above the provider registry, not a provider that routes
or fetches data itself. This preserves the router/provider architecture and
allows local and explicitly enabled remote providers to be compared by the
same deterministic cases.

```bash
PYTHONPATH=src .venv/bin/cortexmux models qualify \
  --config configs/cortexmux.example.toml \
  --suite configs/model-qualification.example.json \
  --output outputs/model-qualification.json
```

The suite may list the same model more than once with different options. For
example, a local model can be tested at 8K and 16K context, while an OpenAI
model can be tested with different `reasoning_effort` values. Recommendations
are made for `fast`, `balanced`, and `quality` per tested task. A candidate
must meet `minimum_quality`; latency, estimated request cost, and local memory
fit then contribute according to the profile weights.

The output contains both the full evidence and `routing_profiles`, a compact
provider/model/options mapping that can be copied into a project's routing
configuration without reinterpreting scores.

Only `exact_text` and `exact_json` validators are supported. Structured results
record `syntax_valid`, `schema_valid`, and `expected_match` independently, so a
semantically wrong value is not reported as malformed JSON. The overall
`passed` field remains the strict conjunction used for scoring. Repetitions are
scored independently; the example runs each case three times and defines its
business terms explicitly to reduce ambiguous or one-off verdicts.

Models never grade other models, generated code is never executed, failures
remain explicit, and tests run sequentially to avoid loading multiple large
local models together. Each result keeps only a bounded output excerpt for
diagnosis; suites should still contain synthetic or otherwise approved
benchmark inputs.

## OpenAI opt-in

The OpenAI provider is disabled by default. Enabling it requires all three:

```toml
[core]
allow_remote_hosts = false
approved_hosts = ["api.openai.com"]

[providers.openai]
enabled = true
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
```

The key is read from the named environment variable and is never written to a
manifest. CortexMux first calls the Models API; a configured candidate that is
not accessible to that account is recorded as unavailable and cannot be
recommended. Qualification sends only the synthetic prompts committed in the
suite. Do not put project data, secrets, or production samples in a suite.

The example uses the documented GPT-5.6 roles: `gpt-5.6-luna` for efficient,
high-volume work, `gpt-5.6-terra` for balance, and `gpt-5.6-sol` for maximum
capability. Prices in a suite are inputs to the scoring calculation, not a
live billing feed; verify them against the official model catalog before a
production run: <https://developers.openai.com/api/docs/models>.

## YAML

JSON works with the base installation. YAML input and output are available
through the optional dependency:

```bash
pip install "cortexmux[yaml]"
```

Manifests are written atomically and existing files are not replaced unless
`--overwrite` is explicit.
