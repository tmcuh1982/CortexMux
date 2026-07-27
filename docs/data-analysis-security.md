# Data-analysis security

Models receive no complete dataset. CortexMux sends only schema, numeric
summaries, missing counts, bounded operation results, and at most the configured
sample size. Columns matching password, token, secret, API key, or private key
patterns are redacted before sampling. Prompt characters, steps, output rows,
categories, and chart count are capped.

Analysis plans contain enums and values, not source code. Engines never use
`eval`, `exec`, dynamic imports, user-controlled subprocesses, arbitrary SQL,
or model-initiated filesystem/network access. DuckDB uses fixed reader APIs.

AI mathematical claims also contain enums and values, never expressions or
code. Operands are either finite decimal literals or JSON Pointers restricted
to the bounded `/profile` and `/results` payload. CortexMux recalculates them
with a fixed operation whitelist and `Decimal` arithmetic, then applies
configured absolute and relative tolerances. Missing paths, division by zero,
non-numeric values, NaN, and infinity are marked unverifiable.
If any declared claim fails, the model text is withheld from the trusted
`interpretation` field. Sensitive columns are removed not only from samples but
also from deterministic result rows before those rows are sent to a model.

Website retrieval and HTML parsing are deliberately outside this module.
Applications should enforce their own URL allowlist and pass only extracted
records to CortexMux, avoiding server-side request forgery and accidental
transmission of full pages.
