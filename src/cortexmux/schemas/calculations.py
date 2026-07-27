"""Schemas for deterministic verification of AI mathematical claims."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, WithJsonSchema, field_validator, model_validator

DecimalNumber = Annotated[Decimal, WithJsonSchema({"type": "number"})]


class MathOperation(StrEnum):
    """Whitelisted mathematical operations accepted from an AI claim."""

    SUM = "sum"
    SUBTRACT = "subtract"
    PRODUCT = "product"
    DIVIDE = "divide"
    MEAN = "mean"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    PERCENTAGE = "percentage"
    PERCENTAGE_CHANGE = "percentage_change"


class VerificationStatus(StrEnum):
    """Outcome of deterministic mathematical verification."""

    VERIFIED = "verified"
    INCORRECT = "incorrect"
    UNVERIFIABLE = "unverifiable"


class CalculationOperand(BaseModel):
    """A numeric literal or JSON Pointer into deterministic analysis results."""

    source_path: str | None = Field(
        default=None,
        max_length=500,
    )
    literal: DecimalNumber | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> CalculationOperand:
        """Require one and only one source for an operand."""
        if (self.source_path is None) == (self.literal is None):
            raise ValueError("operand requires exactly one of source_path or literal")
        return self

    @field_validator("source_path")
    @classmethod
    def source_path_is_bounded(cls, value: str | None) -> str | None:
        """Allow JSON Pointers only below deterministic context roots."""
        if value is not None and not (
            value == "/profile"
            or value.startswith("/profile/")
            or value == "/results"
            or value.startswith("/results/")
        ):
            raise ValueError("source_path must start with /profile or /results")
        return value

    @field_validator("literal")
    @classmethod
    def literal_is_finite(cls, value: Decimal | None) -> Decimal | None:
        """Reject NaN and infinity."""
        if value is not None and not value.is_finite():
            raise ValueError("operand literal must be finite")
        return value


class CalculationClaim(BaseModel):
    """A structured numerical result claimed by a language model."""

    label: str = Field(min_length=1, max_length=200)
    operation: MathOperation
    operands: list[CalculationOperand] = Field(min_length=1, max_length=100)
    claimed_result: DecimalNumber
    rounding_digits: int | None = Field(default=None, ge=0, le=12)

    @field_validator("claimed_result")
    @classmethod
    def result_is_finite(cls, value: Decimal) -> Decimal:
        """Reject NaN and infinity."""
        if not value.is_finite():
            raise ValueError("claimed result must be finite")
        return value


class InterpretationWithCalculations(BaseModel):
    """Structured AI interpretation with auditable mathematical claims."""

    content: str = Field(min_length=1)
    calculations: list[CalculationClaim] = Field(default_factory=list, max_length=100)


class CalculationVerification(BaseModel):
    """Independent deterministic verification of one AI calculation claim."""

    label: str
    operation: MathOperation
    status: VerificationStatus
    claimed_result: DecimalNumber
    expected_result: DecimalNumber | None = None
    absolute_error: DecimalNumber | None = None
    tolerance: DecimalNumber | None = None
    resolved_operands: list[DecimalNumber] = Field(default_factory=list)
    reason: str | None = None
