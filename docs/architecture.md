# Architecture

```text
Application
    ↓
CortexMux facade
    ↓
Typed request validation
    ↓
Deterministic router
    ↓
Instance provider registry
    ↓
Selected provider
    ↓
Provider-specific client
    ↓
Normalized response + routing metadata
```

The router knows only the provider contract. It honors an explicit
provider/model, an explicit provider with a configured model, a model supported
by exactly one provider, or a task default—in that order. Ambiguity is an error.

Data analysis is separate:

```text
Validate local source → profile → optional structured plan → validate whitelist
→ deterministic engine → bounded structured interpretation → independently
verify every AI mathematical claim → structured response
```

Providers and registries are instance scoped. Network clients support async
cleanup and the facade supplies safe synchronous wrappers.

Model qualification is also separate from routing:

```text
Portable suite + machine profile → provider model discovery
→ bounded deterministic cases, executed sequentially
→ exact validation + latency/cost/memory evidence
→ per-task fast/balanced/quality ranking
→ auditable JSON/YAML manifest + adaptable routing profiles
```

The qualifier consumes registered providers but is not itself a provider. It
never changes router state, downloads models, executes model-generated code, or
enables a remote provider. Candidate configurations are explicit so embedding,
image, audio, or other specialized models are not benchmarked with an invalid
text workload merely because a provider lists them.
