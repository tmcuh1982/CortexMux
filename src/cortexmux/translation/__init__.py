"""Typed requests and results for the CortexMux translation harness."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Name = Annotated[str, StringConstraints(min_length=1, max_length=256, strip_whitespace=True)]
Fragment = Annotated[str, StringConstraints(min_length=1, max_length=8192)]


class TranslationEngine(BaseModel):
    """Explicit provider, model and optional behavior profile for one role."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    provider: Name
    model: Name
    model_profile: Name | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    behavior_profile: Name | None = None


class TranslationRequest(BaseModel):
    """Translate one literal field from a Japanese or Chinese citation into French."""

    model_config = ConfigDict(extra="forbid")
    text: Fragment
    citation: Fragment
    source_language: Literal["zh", "ja"]
    translators: tuple[TranslationEngine, TranslationEngine]
    arbiter: TranslationEngine
    think: bool = True
    max_passes: Literal[1, 2] = 2
    glossary: dict[Name, Name] = Field(
        default_factory=lambda: {
            "喷涂": "pulvérisation",
            "コストパフォーマンス": "bon rapport qualité-prix",
            "分子公司": "filiales et succursales",
        },
        max_length=256,
    )
    preserve: list[Name] = Field(default_factory=list, max_length=256)

    references: list[Name] = Field(default_factory=list, max_length=256)
    decisive_span: Name | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        """Require a literal source span and two distinct models."""
        if not self.text.strip() or self.text not in self.citation:
            raise ValueError("text must be a nonblank literal span of citation")
        if self.decisive_span and self.decisive_span not in self.text:
            raise ValueError("decisive_span must occur in the field")
        if any(ref not in self.text for ref in self.references):
            raise ValueError("each declared reference must occur in the field")
        a, b = self.translators
        if (a.provider, a.model) == (b.provider, b.model):
            raise ValueError("translators must specify two distinct engines")
        return self


class TranslationResult(BaseModel):
    """One field decision, with both candidates in the original engine order."""

    status: Literal["accepted", "review"]
    translation: str | None = None
    selected_engine: TranslationEngine | None = None
    candidates: tuple[str, str]
    source_quote: str | None = None
    reason: str
    passes_used: Literal[1, 2] = 1
    first_review_reason: str | None = None
    quantities: dict[str, int] = Field(default_factory=dict)
