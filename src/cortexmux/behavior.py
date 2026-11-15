"""Portable, declarative behavior contracts; deployment secrets stay in bindings."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cortexmux.core.exceptions import InvalidRequestError


class ContractModel(BaseModel):
    """Immutable, closed manifest record."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ModelIdentity(ContractModel):
    """Exact portable identity; absent revisions are explicitly unknown."""

    server: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    server_version: str = Field(pattern=r"^[a-zA-Z0-9_.+-]+$")
    model: str = Field(pattern=r"^[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_.-]+)?$")
    model_revision: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]+$")
    quantization: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]+$")


class BehaviorRules(ContractModel):
    """Whitelisted syntax and transport rules, never executable instructions."""

    tasks: tuple[Literal["structured_output"], ...] = ("structured_output",)
    require_max_tokens: Literal[True] = True
    finish_reason: Literal["stop"] = "stop"
    json_root: Literal["object"] = "object"
    think_prefix: bool = False
    json_fence: bool = False
    reasoning: Literal["unsupported", "mlx", "deepseek"] = "unsupported"
    reasoning_budget: bool = False
    sampling: tuple[Literal["temperature", "top_p"], ...] = ()
    output_cap_scope: Literal["unknown", "answer", "reasoning_and_answer"] = "unknown"

    @model_validator(mode="after")
    def coherent_budget(self) -> BehaviorRules:
        """Reject budget support without a supported reasoning translation."""
        if self.reasoning_budget and self.reasoning != "mlx":
            raise ValueError("Reasoning budget requires a reasoning translation")
        return self


class SyntaxFixture(ContractModel):
    """Synthetic syntax example included in the portable fingerprint."""

    is_synthetic: Literal[True] = True
    raw: str = Field(max_length=16000, repr=False)
    accepted: bool
    expected_json: str | None = Field(default=None, max_length=16000, repr=False)


class Qualification(ContractModel):
    """Attestation for an exact identity and immutable rules/fixture fingerprint."""

    identity: ModelIdentity
    rules_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    kind: Literal["offline", "live"]
    passed: bool
    checks: tuple[
        Literal[
            "syntax", "identity", "structured_output", "max_tokens", "reasoning", "reasoning_budget"
        ],
        ...,
    ] = ()
    evidence: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$", max_length=128)


class BehaviorProfile(ContractModel):
    """Versioned portable contract, importable without executing downloaded code."""

    format_version: Literal[1] = 1
    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]+$", max_length=128)
    revision: int = Field(gt=0, strict=True)
    provider: Literal["mlx", "mlx_lm", "mlx_vlm", "deepseek"]
    identity: ModelIdentity
    rules: BehaviorRules
    fixtures: tuple[SyntaxFixture, ...] = Field(default=(), max_length=128)
    qualifications: tuple[Qualification, ...] = ()

    @property
    def rules_fingerprint(self) -> str:
        """Hash rules and fixtures independently of compatible server versions."""
        return _digest(
            {
                "rules": self.rules.model_dump(),
                "fixtures": [fixture.model_dump() for fixture in self.fixtures],
            }
        )

    @property
    def fingerprint(self) -> str:
        """Hash the entire canonical manifest, including attestations."""
        return _digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def attestations_match(self) -> BehaviorProfile:
        """Do not accept an attestation for different rules."""
        if any(q.rules_fingerprint != self.rules_fingerprint for q in self.qualifications):
            raise ValueError("Qualification fingerprint mismatch")
        return self

    def export_json(self) -> str:
        """Export canonical portable JSON without deployment bindings."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @classmethod
    def import_json(cls, document: str) -> BehaviorProfile:
        """Validate a bounded manifest; importing never activates it."""
        if len(document) > 2_000_000:
            raise ValueError("Manifest exceeds size limit")
        from cortexmux.core.normalization import normalize_json_object

        return cls.model_validate(normalize_json_object(document)[1])


class BehaviorBinding(ContractModel):
    """Local, explicit identity/alias assertion; never part of an export."""

    profile: BehaviorProfile
    identity: ModelIdentity
    identity_source: Literal["declared", "observed"] = "declared"
    qualification_policy: Literal["live", "offline"] = "live"
    selector: str | None = Field(default=None, min_length=1, repr=False)
    reported_selectors: tuple[str, ...] = Field(default=(), repr=False)

    def validate_target(
        self, provider: str, model: str, version: str | None
    ) -> Literal["offline", "live"]:
        """Require exact identity and sufficient attestation; offline use is explicit."""
        p = self.profile
        required = {"identity", "structured_output", "max_tokens"}
        if p.rules.reasoning != "unsupported":
            required.add("reasoning")
        if p.rules.reasoning_budget:
            required.add("reasoning_budget")
        matching = [q for q in p.qualifications if q.identity == self.identity and q.kind == "live"]
        live = (
            bool(matching)
            and all(q.passed for q in matching)
            and any(required <= set(q.checks) for q in matching)
        )
        offline = (
            self.qualification_policy == "offline"
            and self.identity == p.identity
            and not matching
            and qualify_syntax(p).passed
        )
        translation = "deepseek" if provider == "deepseek" else "mlx"
        if (
            provider != p.provider
            or model != self.identity.model
            or version != self.identity.server_version
            or p.rules.reasoning not in {"unsupported", translation}
            or not (live or offline)
        ):
            raise InvalidRequestError(
                "Behavior profile identity or qualification is incompatible.",
                category="compatibility",
            )
        return "live" if live else "offline"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def qualify_syntax(profile: BehaviorProfile) -> Qualification:
    """Run embedded synthetic fixtures only; this never attests server behavior."""
    from cortexmux.core.normalization import normalize_json_object

    passed = bool(profile.fixtures)
    for fixture in profile.fixtures:
        try:
            _, parsed, _ = normalize_json_object(
                fixture.raw,
                think_prefix=profile.rules.think_prefix,
                json_fence=profile.rules.json_fence,
            )
            accepted = True
            if fixture.expected_json is not None:
                expected = normalize_json_object(fixture.expected_json)[1]
                if parsed != expected:
                    passed = False
        except (ValueError, RecursionError):
            accepted = False
        passed = passed and accepted == fixture.accepted
    return Qualification(
        identity=profile.identity,
        rules_fingerprint=profile.rules_fingerprint,
        kind="offline",
        checks=("syntax",),
        passed=passed,
        evidence="embedded-syntax-fixtures",
    )


def bundled_profile(name: Literal["magistral", "qwen"]) -> BehaviorProfile:
    """Load a bundled candidate manifest without activating or live-qualifying it."""
    from importlib.resources import files

    if name not in {"magistral", "qwen"}:
        raise ValueError("Unknown bundled profile")
    return BehaviorProfile.import_json(
        files("cortexmux").joinpath("behavior_profiles", f"{name}.json").read_text()
    )
