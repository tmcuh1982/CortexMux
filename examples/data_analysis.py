"""Run deterministic local analysis."""

from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    response = mux.analyze_data("sales.csv", instruction="Summarize sales.")
    print(response.results)
    print(response.calculation_verifications)
