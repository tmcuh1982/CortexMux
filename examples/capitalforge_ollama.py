"""Ask local Ollama about read-only CapitalForge data through MCP stdio."""

from cortexmux import CortexMux


def main() -> None:
    """Run an opt-in French CapitalForge analysis with a pinned local model."""
    with CortexMux.from_env(config_path="configs/cortexmux.example.toml") as mux:
        response = mux.capitalforge_chat(
            "Fais une analyse générale du portefeuille et compare seulement les signaux utiles.",
            model="qwen3:4b",
            temperature=0.1,
        )
    print(response.content)


if __name__ == "__main__":
    main()
