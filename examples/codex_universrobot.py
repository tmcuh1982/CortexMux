"""Run from an approved installed release, never a UniversRobot checkout dependency."""

import argparse
from pathlib import Path

from pydantic import BaseModel

from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig
from cortexmux.providers.codex import CodexConfig


class Analysis(BaseModel):
    """Validated application response."""

    summary: str


def main() -> None:
    """Connect, select an explicit model, analyze supplied text, and read limits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Exact identifier from Codex model/list")
    parser.add_argument("--auth-directory", type=Path, required=True)
    parser.add_argument("--accept-sandbox-limitations", action="store_true")
    args = parser.parse_args()
    config = CortexMuxConfig()
    config.providers.ollama.enabled = False
    config.providers.comfyui.enabled = False
    config.providers.codex = CodexConfig(enabled=True, auth_directory=args.auth_directory)
    with CortexMux(config) as mux:
        if not mux.codex_account().connected:
            login = mux.codex_login_start()
            # A GUI application presents this URL without logging it.
            print("Open this login URL:", login.auth_url)
            if not mux.codex_login_result(login.login_id).success:
                raise RuntimeError("ChatGPT login did not complete.")
        if args.model not in {model.name for model in mux.list_models("codex")}:
            raise ValueError("Choose an available Codex model.")
        response = mux.structured(
            "Summarize this supplied text: The prototype processed 42 local records.",
            response_model=Analysis,
            provider="codex",
            model=args.model,
            require_no_tools=not args.accept_sandbox_limitations,
        )
        print(response.parsed.model_dump_json())
        print(mux.codex_rate_limits().model_dump_json())


if __name__ == "__main__":
    main()
