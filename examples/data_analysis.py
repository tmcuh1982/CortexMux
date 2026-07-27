"""Run deterministic local analysis."""

from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    print(mux.analyze_data("sales.csv", instruction="Summarize sales.").report_markdown)
