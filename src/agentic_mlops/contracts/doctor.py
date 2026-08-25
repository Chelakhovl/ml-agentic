"""Pydantic contracts for the `agentic-mlops doctor` self-check command."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CheckStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class DoctorCheck(BaseModel):
    name: str
    category: str  # "packages" | "configs" | "directories" | "tools"
    status: CheckStatus
    message: str
    detail: str | None = None


class DoctorReport(BaseModel):
    checks: list[DoctorCheck] = Field(default_factory=list)
    overall_status: CheckStatus = CheckStatus.OK
    num_ok: int = 0
    num_warnings: int = 0
    num_errors: int = 0

    @classmethod
    def from_checks(cls, checks: list[DoctorCheck]) -> DoctorReport:
        num_ok = sum(1 for c in checks if c.status == CheckStatus.OK)
        num_warnings = sum(1 for c in checks if c.status == CheckStatus.WARNING)
        num_errors = sum(1 for c in checks if c.status == CheckStatus.ERROR)
        if num_errors:
            overall = CheckStatus.ERROR
        elif num_warnings:
            overall = CheckStatus.WARNING
        else:
            overall = CheckStatus.OK
        return cls(
            checks=checks,
            overall_status=overall,
            num_ok=num_ok,
            num_warnings=num_warnings,
            num_errors=num_errors,
        )
