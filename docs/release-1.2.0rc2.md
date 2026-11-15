# CortexMux 1.2.0rc2

Prepared 14 November 2026 as the documentation follow-up to
[1.2.0rc1](release-1.2.0rc1.md). Version metadata advances to 1.2.0rc2;
runtime behavior, profile rules, fixtures and qualification gates are unchanged.

## Included changes

- [Nine-deliverable review matrix](behavior-review-1.2.0rc1.md#nine-requested-deliverables).
- [Live report](behavior-live-1.2.0rc1.md): 14 public-facade calls and seven separate
  client-level logprobs probes, with measurement conventions and remaining limits.
- [Migration and requalification guide](behavior-profiles.md): output-cap scope,
  reasoning transition semantics and distinction between diagnostics and attestation.
- Corrected historical publication status and immutable rc1 artifact references.

The rc1-named documents identify the candidate used for the live campaigns. Their
measurements are carried forward as evidence of unchanged runtime behavior; no new
live campaign is claimed for rc2. The rc1 tag and distributions remain intact.

## Remaining limits

Qualification remains partial. Magistral reasoning-budget enforcement, actual
weight revisions/quantization, installed server source SHA, DeepSeek live behavior,
vision and TracedCortexMux integration remain unqualified or untested as described
in the live report. No complete live attestation, profile activation or automatic
consumer migration is included. UniversRobot is unchanged.

## Validation and distribution record

Ruff lint/format, mypy (97 source files), the offline suite (**469 passed,
8 skipped**) and diff checks passed. The skipped tests require explicit live
configuration; no new live run is claimed. Wheel/source build, Twine validation
and exact SHA-256 hashes are recorded in the GitHub prerelease notes.
The rc1 hashes in the historical release record do not identify rc2 artifacts.
Publication is to GitHub Releases only; no PyPI upload is included.
