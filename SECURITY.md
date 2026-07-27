# Security

Report vulnerabilities privately through the repository's security advisory
feature. Do not include live credentials or sensitive datasets in reports.

CortexMux permits only loopback/localhost provider URLs by default. To use a
remote provider, set `allow_remote_hosts = true` or explicitly approve a host
and understand that prompts, images, or bounded data summaries may leave the
machine. Authorization header values use secret types and are not logged.

Data plans are Pydantic-validated enums. CortexMux does not execute
model-generated Python, SQL, shell commands, imports, filesystem operations, or
network actions. Sensitive columns are excluded from samples using configurable
patterns, and prompts/results are capped.

Files are resolved, format- and size-checked, outputs are constrained beneath
their root, and existing files are not overwritten.

