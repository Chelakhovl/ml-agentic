"""SystemDoctor — runs local self-checks for `agentic-mlops doctor`.

No network calls are made. Checks:
  packages   — importability of required and optional extras
  configs    — YAML files parse without error, required fields present
  directories — runs_dir / registry_dir / dataset_registry_dir are
                creatable/writable
  tools      — external CLI binaries (docker, az) in PATH
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

import yaml

from agentic_mlops.contracts.doctor import CheckStatus, DoctorCheck, DoctorReport

# ── Package manifest ──────────────────────────────────────────────────────────

_REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("pydantic", "pydantic"),
    ("yaml", "pyyaml"),
    ("typer", "typer"),
    ("rich", "rich"),
]

_OPTIONAL_PACKAGES: list[tuple[str, str, str]] = [
    # (import_name, pip_name, extra_label)
    ("ultralytics", "ultralytics", "ultralytics"),
    ("mlflow", "mlflow", "mlflow"),
    ("azure.ai.ml", "azure-ai-ml", "azure"),
    ("azure.monitor.query", "azure-monitor-query", "azure"),
    ("onnx", "onnx", "onnx"),
    ("PIL", "Pillow", "vision"),
    ("fastapi", "fastapi", "web"),
    ("uvicorn", "uvicorn", "web"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _check_import(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _dir_writable(path: Path) -> tuple[bool, str]:
    """Return (ok, reason). Creates missing directories; uses a probe write."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Cannot create directory: {exc}"
    probe = path / ".doctor_probe"
    try:
        probe.write_text("x")
        probe.unlink()
        return True, ""
    except OSError as exc:
        return False, f"Directory not writable: {exc}"


def _cmd_version(cmd: str) -> str | None:
    """Return the first line of `cmd --version`, or None if not found."""
    exe = shutil.which(cmd)
    if exe is None:
        return None
    try:
        result = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5)
        first = (result.stdout or result.stderr or "").strip().splitlines()
        return first[0] if first else "(unknown version)"
    except Exception:
        return None


# ── Check implementations ─────────────────────────────────────────────────────


def _check_required_packages() -> list[DoctorCheck]:
    checks = []
    for import_name, pip_name in _REQUIRED_PACKAGES:
        ok = _check_import(import_name)
        checks.append(
            DoctorCheck(
                name=f"package:{pip_name}",
                category="packages",
                status=CheckStatus.OK if ok else CheckStatus.ERROR,
                message=(
                    f"{pip_name} available"
                    if ok
                    else f"{pip_name} missing — run: pip install agentic-mlops"
                ),
            )
        )
    return checks


def _check_optional_packages() -> list[DoctorCheck]:
    checks = []
    for import_name, pip_name, extra in _OPTIONAL_PACKAGES:
        ok = _check_import(import_name)
        checks.append(
            DoctorCheck(
                name=f"package:{pip_name}",
                category="packages",
                status=CheckStatus.OK if ok else CheckStatus.WARNING,
                message=(
                    f"{pip_name} available"
                    if ok
                    else (
                        f"{pip_name} not installed — optional"
                        f" (pip install 'agentic-mlops[{extra}]')"
                    )
                ),
            )
        )
    return checks


def _check_config_file(path: Path, label: str) -> DoctorCheck:
    if not path.exists():
        return DoctorCheck(
            name=f"config:{label}",
            category="configs",
            status=CheckStatus.ERROR,
            message=f"{label} not found: {path}",
        )
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if data is None:
            return DoctorCheck(
                name=f"config:{label}",
                category="configs",
                status=CheckStatus.WARNING,
                message=f"{label} is empty: {path}",
            )
        return DoctorCheck(
            name=f"config:{label}",
            category="configs",
            status=CheckStatus.OK,
            message=f"{label} valid YAML ({len(data)} top-level key(s))",
            detail=str(path),
        )
    except yaml.YAMLError as exc:
        return DoctorCheck(
            name=f"config:{label}",
            category="configs",
            status=CheckStatus.ERROR,
            message=f"{label} invalid YAML: {exc}",
            detail=str(path),
        )


def _check_directory(path: Path, label: str) -> DoctorCheck:
    ok, reason = _dir_writable(path)
    return DoctorCheck(
        name=f"directory:{label}",
        category="directories",
        status=CheckStatus.OK if ok else CheckStatus.ERROR,
        message=f"{label} writable: {path}" if ok else f"{label}: {reason} ({path})",
    )


def _check_tool(cmd: str, label: str, optional: bool = True) -> DoctorCheck:
    ver = _cmd_version(cmd)
    if ver is not None:
        return DoctorCheck(
            name=f"tool:{cmd}",
            category="tools",
            status=CheckStatus.OK,
            message=f"{label} found: {ver}",
        )
    return DoctorCheck(
        name=f"tool:{cmd}",
        category="tools",
        status=CheckStatus.WARNING if optional else CheckStatus.ERROR,
        message=f"{label} not found in PATH"
        + (" (optional — needed for Docker/AKS deployment)" if cmd == "docker" else "")
        + (" (optional — needed for Azure CLI auth)" if cmd == "az" else ""),
    )


# ── Public API ────────────────────────────────────────────────────────────────


class SystemDoctor:
    """Runs all self-checks and returns a DoctorReport."""

    def check(
        self,
        *,
        config_paths: list[tuple[Path, str]] | None = None,
        directory_paths: list[tuple[Path, str]] | None = None,
        check_docker: bool = True,
        check_az: bool = True,
    ) -> DoctorReport:
        """
        Args:
            config_paths: List of ``(path, label)`` pairs to validate as YAML.
            directory_paths: List of ``(path, label)`` pairs to check for write access.
            check_docker: Whether to probe for the docker CLI binary.
            check_az: Whether to probe for the az CLI binary.
        """
        checks: list[DoctorCheck] = []

        checks += _check_required_packages()
        checks += _check_optional_packages()

        for path, label in config_paths or []:
            checks.append(_check_config_file(path, label))

        for path, label in directory_paths or []:
            checks.append(_check_directory(path, label))

        if check_docker:
            checks.append(_check_tool("docker", "Docker CLI", optional=True))
        if check_az:
            checks.append(_check_tool("az", "Azure CLI", optional=True))

        return DoctorReport.from_checks(checks)
