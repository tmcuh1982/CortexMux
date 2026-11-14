# Synthetic behavior-profile fixtures

`behavior-profile-fixtures.json` contains only synthetic strings; every row and
the document are marked `is_synthetic=true`. The portable manifests embed these
fixtures so imports remain self-contained. Tests require exact equality between
the source and packaged copies.

Stable invariants: only one complete permitted THINK prefix and/or one JSON fence
may be removed; JSON values, citations and nested relations are preserved. Duplicate
keys, nonfinite numbers, extra objects and malformed envelopes fail. Positive
fixtures include the exact expected JSON values. These are syntax/value fixtures,
not model/server performance or real-market measurements. Finish and token-budget
checks use separate MockTransport tests; live enforcement is never inferred here.
