# Data-analysis security

Models receive no complete dataset. CortexMux sends only schema, numeric
summaries, missing counts, bounded operation results, and at most the configured
sample size. Columns matching password, token, secret, API key, or private key
patterns are redacted before sampling. Prompt characters, steps, output rows,
categories, and chart count are capped.

Analysis plans contain enums and values, not source code. Engines never use
`eval`, `exec`, dynamic imports, user-controlled subprocesses, arbitrary SQL,
or model-initiated filesystem/network access. DuckDB uses fixed reader APIs.

