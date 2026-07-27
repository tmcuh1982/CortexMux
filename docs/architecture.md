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
