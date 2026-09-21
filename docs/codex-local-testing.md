# Private offline Codex testing

Use `codex-lab` to store synthetic or already obtained Codex responses, validate
them, and replay them without starting Codex, using a subscription, or making a
network request. This is deterministic response replay, not a local AI model or
an emulation of App Server authentication, tools, timing, or protocol behavior.
The existing fake-process unit tests cover the App Server transport separately.

## Where data is stored

The default is `<platform application data>/cortexmux/codex-lab`, outside the
CortexMux checkout. On macOS this is normally
`~/Library/Application Support/cortexmux/codex-lab`.

```sh
cortexmux codex-lab init
```

The command prints the actual directory. Each case is a separate
`<id>.codex-case.json` file. Each validation saves a content-free
`<id>.codex-validation.json` report. Listing and validating do not print the
stored prompts, responses, or expected outputs. Explicit Python replay returns
the output to the consuming application, which controls its own display/logging.

Storage is refused inside any detected Git repository, including conventional
worktrees. This is checked again before reads and writes, so initializing Git in
the storage directory later stops lab operations. A nonempty unrelated directory
cannot be used as a store. Case files cannot overwrite existing cases, follow a
file symlink, or escape through a supplied case identifier. Limits: 4 MB per file,
1 million characters per response/expected response, 1000 cases per listing.
Files use mode `0600` and the store `0700` on POSIX systems; Windows applications
should use their normal per-user filesystem access controls.

The store has an ignore-all `.gitignore`. The project also ignores
`*.codex-case.json`, `*.codex-validation.json`, and `.cortexmux-private/`, and
explicitly excludes them from wheel/source packages. No case data is copied into
the repository, sent to GitHub, or uploaded by lab operations. These protections
do not prevent someone manually copying/renaming data, forcing a Git add, or
uploading it through another application. Keep private cases out of cloud-synced
folders if they must remain on the device. Files are local plaintext, not encrypted.

## Import and validate

Prepare a request JSON file **outside your repository** using the existing
CortexMux request format, for example:

```json
{
  "task": "structured_output",
  "provider": "codex",
  "model": "your-recorded-model-id",
  "prompt": "Your supplied text and analysis request",
  "json_schema": {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": false
  }
}
```

Save only the response's text in another private file. For structured output,
this text is the JSON that Codex returned, not the whole App Server envelope.

```sh
cortexmux codex-lab add --request /private/request.json \
  --response /private/response.txt --origin recorded
cortexmux codex-lab list
cortexmux codex-lab validate CASE_ID
cortexmux codex-lab validate
```

`add` prints the new opaque case identifier. Replace `CASE_ID` with it. Omit the
identifier to validate every local case. Every command accepts `--root` for a
custom dedicated directory outside Git. The lab never initiates real generation
or authentication. `--origin recorded` marks imported data; it is not proof that
the data came from Codex.

For deliberately invalid fixtures, use `--expect-invalid`. Use
`--status interrupted` or `--status failed` to test non-completed responses.
Provide `--expected /private/expected.txt` for an additional **exact text** check
(including JSON whitespace). A negative test can pass because the invalid output
was correctly rejected; the report keeps `response_valid=false` separate from
`expectation_met=true`.

Validation checks completion, nonempty text, JSON syntax and the same supported
JSON Schema subset as CortexMux. Without an expected output, schema validation
does not establish factual correctness, calculation accuracy, or business-rule
correctness. Add application assertions for those properties. `expected_content`
compares exact strings, not semantic JSON equality.

Exit codes: `0` when every expectation is met; `1` for an unmet expectation;
`2` for invalid inputs/storage or an empty suite. Reports contain no sensitive
content. Validation failures never execute any instruction from returned JSON.

## Python recording and replay

```python
from cortexmux.testing import CodexCaseStore, CodexReplayProvider
from cortexmux import CortexMux

store = CodexCaseStore()
# Explicitly record an already obtained response. This method makes no live call.
# request and response must have matching task, model, and request_id.
# path = store.record(request, response)

case = store.load("your-32-character-case-id")
with CortexMux(register_builtin_providers=False) as mux:
    mux.register_provider(CodexReplayProvider(case))
    replay_request = case.request.model_copy(update={"provider": "codex-replay"})
    result = mux.run(replay_request)
    assert result.raw_metadata["simulation"] is True
```

`record` stores only the typed request and final text. Request metadata, response
metadata, tokens, account state, login URLs and raw transport logs are not
captured. Text intentionally supplied in the prompt or response is retained,
so redact any credentials embedded in that text before storing it.
For invalid raw responses, construct `CodexTestCase` or use `codex-lab add` instead
of `record`, which accepts already normalized responses.

Replay requires the fixture's model, task and exact inputs/schema/options. A
mismatch fails instead of returning unrelated data or making a live fallback.
Provider name, request identifier, timeout, metadata, routing profile and streaming
flag are ignored for matching. Repeated runs are deterministic and stateless;
there is no simulated conversation memory. The provider exposes only the fixture's
model with `simulation=true`. Its healthcheck is fixture health, not account health.

`mux.stream(replay_request)` / `mux.astream(replay_request)` emit deterministic
256-character chunks after validation and a final marked completion. They do not
reproduce original stream timing, partial invalid chunks, or tool notifications.
Replay never opens a process or network connection. Selecting another provider
elsewhere in your application remains outside the lab's control.

A runnable example creates synthetic valid/invalid cases in private storage:

```sh
python examples/codex_local_lab.py
```

Only this example's source code belongs in Git; its generated cases and reports
go to the private store. No live account, API key, or Codex executable is needed.
