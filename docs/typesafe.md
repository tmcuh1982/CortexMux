# TypeSafe Jev decisions

CortexMux exposes TypeSafe's System One API as an optional decision provider.
It sends an application-supplied state and typed questions, then returns typed
answers, probabilities, model identity, usage and routing metadata. CortexMux
does not decide what a probability means or trigger a second model.

Jev is a hosted service. Access to the TypeSafe API and a `TYPESAFE_API_KEY` are
required for live calls; unit tests use `httpx.MockTransport` without an account
or network connection. The provider is disabled by default and requires
explicit approval of `api.typesafe.ai` under the provider network policy.

```toml
[core]
approved_hosts = ["api.typesafe.ai"]

[providers.typesafe]
enabled = true
default_model = "jev-latest"
```

The application supplies the key in the `TYPESAFE_API_KEY` environment variable.
`CORTEXMUX_TYPESAFE_ENABLED=true` is also supported. A key alone does not enable
the provider. TypeSafe publishes `jev-latest` as an alias that can move; the
response retains the versioned model ID that answered.
For calibrated thresholds, the application can pass a versioned ID such as
`jev-1.13.0`. TypeSafe may omit these IDs from `GET /v1/models`; CortexMux
passes a versioned Jev ID to the API for validation instead of rejecting it
from the alias list.

```python
from cortexmux import CortexMux
from cortexmux.schemas import NoulQuestion

questions = {
    "relevant": NoulQuestion(
        instructions="Does this exact page describe a specific robotics product?"
    ),
    "extractable": NoulQuestion(
        instructions="Does this page contain concrete product facts that can be extracted?"
    ),
}

with CortexMux.from_env(config_path="config.toml") as mux:
    response = mux.decide(
        {
            "exact_url": "https://example.org/product",
            "title": "Example robot",
            "observations": ["An application-supplied, bounded page excerpt."],
        },
        questions,
    )
    relevant = response.answers["relevant"]
    extractable = response.answers["extractable"]
    assert relevant.type == extractable.type == "noul"
    print(relevant.noul, extractable.noul)
```

An application such as UniversRobot compares both values to its own threshold
and calls another AI for validation when both pass. The threshold, choice of
validator, evidence policy and final approval remain in the application.
Input `state` must be a bounded JSON string, object or array. Questions may be
`NoulQuestion`, `ChoiceQuestion` or `ScoreQuestion`. Choice and Score retain the
complete probability distribution and TypeSafe confidence; Noul returns the
probability of yes. These are model estimates, so calibrate thresholds using
the application's own labeled pages before automatic routing.

The TypeSafe provider never fetches a page. Pass content obtained by the
application or by CortexMux's opt-in web fetcher. Preserve the exact URL and
page provenance at the application boundary. Avoid sending sensitive page data
unless remote transmission is explicitly intended.

## Live probe with reviewable output

For the technical integration contract, run:

```bash
PYTHONPATH=src .venv/bin/python examples/typesafe_contract_probe.py
```

This probe makes three calls with object, string and array states and exercises
Noul, Choice and Score questions. It checks the normalized answer IDs and
types, probability bounds and distributions, the versioned model, token usage,
request IDs and routing metadata. The prompt and response logs are written to
an ignored, private directory under `outputs/typesafe-contract/`. It applies no
domain threshold and does not judge whether the model's decisions are correct.
To verify a saved run without an API key or network access, use:

```bash
PYTHONPATH=src .venv/bin/python examples/typesafe_contract_probe.py \
  --verify-run outputs/typesafe-contract/<run-id>
```

The scenario probe below exercises the application's example decision rule;
its expected labels do not form part of the CortexMux contract.

Run the synthetic probe without network access first:

```bash
PYTHONPATH=src .venv/bin/python examples/typesafe_live_probe.py --dry-run
```

Then run it live with an API key in `TYPESAFE_API_KEY`, or let the script ask
for the key without echoing it in an interactive terminal:

```bash
PYTHONPATH=src .venv/bin/python examples/typesafe_live_probe.py
```

Each run creates a unique directory under the ignored `outputs/typesafe-probe/`.
`prompt.json` contains the exact state, model and two Noul questions sent for
four synthetic pages. The probe pins `jev-1.13.0` by default so repeated runs
can be compared against the same model version; `--model` overrides it. It also
contains expected labels for review; these are
not sent to TypeSafe. `log.json` contains the normalized responses, scores,
model IDs, usage, routing metadata, errors and the example's strict `> 0.75`
decision about whether to request a second validation. Each result also records
whether this final route matches its synthetic expected label. `status: ok`
means the API calls succeeded; inspect `matches_expected_route` separately to
assess the decision. The API key is never
written to either file. Keep logs with real page content private if you later
adapt the script for non-synthetic examples. Share `prompt.json` and `log.json`
for analysis of the returned probabilities.

See the [TypeSafe API reference](https://docs.typesafe.ai/api) and
[model limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).
