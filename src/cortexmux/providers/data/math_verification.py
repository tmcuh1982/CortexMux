"""Deterministic verification for structured mathematical claims."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any

from cortexmux.schemas.calculations import (
    CalculationClaim,
    CalculationOperand,
    CalculationVerification,
    MathOperation,
    VerificationStatus,
)


class MathVerifier:
    """Recalculate AI claims using a small operation whitelist and Decimal arithmetic."""

    def __init__(
        self,
        *,
        absolute_tolerance: Decimal = Decimal("1e-9"),
        relative_tolerance: Decimal = Decimal("1e-6"),
    ) -> None:
        if absolute_tolerance < 0 or relative_tolerance < 0:
            raise ValueError("verification tolerances cannot be negative")
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance

    def verify(
        self,
        claim: CalculationClaim,
        context: dict[str, Any],
    ) -> CalculationVerification:
        """Resolve operands, recalculate the claim, and compare within tolerance."""
        try:
            operands = [self._resolve_operand(operand, context) for operand in claim.operands]
            expected = self._calculate(claim.operation, operands)
            if claim.rounding_digits is not None:
                quantum = Decimal(1).scaleb(-claim.rounding_digits)
                expected = expected.quantize(quantum, rounding=ROUND_HALF_UP)
            error = abs(expected - claim.claimed_result)
            tolerance = max(
                self.absolute_tolerance,
                self.relative_tolerance * abs(expected),
            )
            status = (
                VerificationStatus.VERIFIED if error <= tolerance else VerificationStatus.INCORRECT
            )
            return CalculationVerification(
                label=claim.label,
                operation=claim.operation,
                status=status,
                claimed_result=claim.claimed_result,
                expected_result=expected,
                absolute_error=error,
                tolerance=tolerance,
                resolved_operands=operands,
                reason=None if status is VerificationStatus.VERIFIED else "Result mismatch.",
            )
        except (InvalidOperation, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return CalculationVerification(
                label=claim.label,
                operation=claim.operation,
                status=VerificationStatus.UNVERIFIABLE,
                claimed_result=claim.claimed_result,
                reason=str(exc),
            )

    def _resolve_operand(
        self,
        operand: CalculationOperand,
        context: dict[str, Any],
    ) -> Decimal:
        if operand.literal is not None:
            return operand.literal
        if operand.source_path is None:
            raise ValueError("operand has no source")
        value: Any = context
        for raw_part in operand.source_path.removeprefix("/").split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict):
                if part not in value:
                    raise KeyError(f"source path does not exist: {operand.source_path}")
                value = value[part]
            elif isinstance(value, list):
                if not part.isdigit():
                    raise KeyError(f"source path list index is invalid: {operand.source_path}")
                index = int(part)
                if index >= len(value):
                    raise KeyError(f"source path does not exist: {operand.source_path}")
                value = value[index]
            else:
                raise KeyError(f"source path is not numeric: {operand.source_path}")
        if isinstance(value, bool) or not isinstance(value, int | float | str | Decimal):
            raise TypeError(f"source path is not numeric: {operand.source_path}")
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise TypeError(f"source path is not numeric: {operand.source_path}") from exc
        if not number.is_finite():
            raise ValueError(f"source path is not finite: {operand.source_path}")
        return number

    @staticmethod
    def _calculate(operation: MathOperation, operands: list[Decimal]) -> Decimal:
        with localcontext() as context:
            context.prec = 50
            if operation is MathOperation.SUM:
                return sum(operands, start=Decimal(0))
            if operation is MathOperation.PRODUCT:
                result = Decimal(1)
                for operand in operands:
                    result *= operand
                return result
            if operation is MathOperation.MEAN:
                return sum(operands, start=Decimal(0)) / Decimal(len(operands))
            if operation is MathOperation.MINIMUM:
                return min(operands)
            if operation is MathOperation.MAXIMUM:
                return max(operands)
            if len(operands) != 2:
                raise ValueError(f"{operation.value} requires exactly two operands")
            first, second = operands
            if operation is MathOperation.SUBTRACT:
                return first - second
            if operation is MathOperation.DIVIDE:
                return first / second
            if operation is MathOperation.PERCENTAGE:
                return first / second * Decimal(100)
            if operation is MathOperation.PERCENTAGE_CHANGE:
                return (first - second) / abs(second) * Decimal(100)
        raise ValueError(f"unsupported mathematical operation: {operation.value}")
