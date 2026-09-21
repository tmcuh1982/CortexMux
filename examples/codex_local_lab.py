"""Create synthetic private cases and validate them without contacting Codex."""

import argparse
from pathlib import Path

from cortexmux import CortexMux
from cortexmux.schemas.requests import StructuredOutputRequest
from cortexmux.testing import CodexCaseStore, CodexReplayProvider, CodexTestCase


def main() -> None:
    """Store synthetic valid/invalid outputs outside Git and exercise offline replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="Dedicated private directory outside Git")
    args = parser.parse_args()
    store = CodexCaseStore(args.root)
    request = StructuredOutputRequest(
        provider="codex",
        model="synthetic-test-model",
        prompt="Synthetic fixture: return the number of two synthetic records.",
        json_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}, "is_synthetic": {"const": True}},
            "required": ["count", "is_synthetic"],
            "additionalProperties": False,
        },
    )
    valid = CodexTestCase(
        request=request,
        content='{"count":2,"is_synthetic":true}',
        expected_content='{"count":2,"is_synthetic":true}',
    )
    invalid = CodexTestCase(
        request=request,
        content='{"count":"wrong-type","is_synthetic":true}',
        expect_valid=False,
    )
    for case in (valid, invalid):
        store.save(case)
        print(store.validate(case.case_id).model_dump_json())
    with CortexMux(register_builtin_providers=False) as mux:
        mux.register_provider(CodexReplayProvider(store.load(valid.case_id)))
        response = mux.run(request.model_copy(update={"provider": "codex-replay"}))
        assert response.parsed == {"count": 2, "is_synthetic": True}
        print("Offline replay validated. Private store:", store.root)


if __name__ == "__main__":
    main()
