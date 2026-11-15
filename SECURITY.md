# Security

## CapitalForge MCP

The CapitalForge MCP integration is opt-in and stdio-only. CortexMux launches
the configured executable without a shell, gives it no inherited secrets, and
never logs stdout because it carries JSON-RPC protocol data. Only five
read-only `capitalforge_` tools are accepted; order, import, YAML-write, and
candidate-activation capabilities do not exist in CortexMux. See
[docs/capitalforge-mcp.md](docs/capitalforge-mcp.md) for disabling and limits.

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

Optional web-page retrieval is disabled by default and has an independent
policy from provider endpoints. Initial URLs and redirects are checked against
an optional exact host allowlist, allowed ports, and resolved public IP
addresses. URL credentials, private/non-public targets, oversized responses,
and non-text content are rejected. Models receive bounded cleaned content, not
network access. For hostile multi-tenant deployments, also enforce outbound
network restrictions because DNS validation and connection establishment are
separate operating-system operations.
