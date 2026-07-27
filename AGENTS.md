# CortexMux Agent Rules

- Keep provider-specific behavior outside the router.
- Use typed Pydantic request and response models.
- Never use arbitrary `eval`, `exec`, or LLM-generated executable code or SQL.
- Do not allow silent remote data transmission; local endpoints are the default.
- Mark every network test as an integration test. Unit tests must not require Ollama or ComfyUI.
- Lazily import optional dependencies and raise `OptionalDependencyError` with the required extra.
- Public functions and classes require docstrings.
- Public API changes require tests and a changelog entry.
- Run Ruff, mypy, and pytest before considering work complete.
- Never log secrets, full binary data, base64 images, or complete sensitive datasets.
- Preserve backward compatibility within the `0.1.x` line unless fixing a security problem.
- Do not add large orchestration frameworks.

