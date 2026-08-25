"""ConfigLinter — validates an orchestrator YAML config without running anything.

Checks:
  steps        — all step names are valid; canonical order is enforced
  dependencies — each step's required upstream steps are present (or have
                 a valid standalone override, e.g. dataset_path for evaluation
                 without training)
  fields       — required fields for each configured step are non-empty
  files        — explicitly-set file paths exist on disk
  azure        — Azure backend/runner consistency (all azure things need
                 azure_config_path; azure_config_path must parse if present)
"""

from __future__ import annotations

from pathlib import Path

from agentic_mlops.contracts.doctor import CheckStatus, DoctorCheck, DoctorReport
from agentic_mlops.contracts.orchestrator import DEFAULT_STEPS, PIPELINE_STEPS, OrchestratorInput


class ConfigLinter:
    """Validates an orchestrator YAML and returns a DoctorReport."""

    def lint(self, config_path: str | Path) -> DoctorReport:
        checks: list[DoctorCheck] = []
        config_path = Path(config_path)

        # ── 1. File exists and parses ─────────────────────────────────────────
        if not config_path.exists():
            self._err(checks, "config_file", "Config", f"Config file not found: {config_path}")
            return DoctorReport.from_checks(checks)

        try:
            inp = OrchestratorInput.from_yaml(config_path, workflow_id="__lint__")
        except Exception as exc:
            self._err(checks, "config_parse", "Config", f"Cannot parse config: {exc}")
            return DoctorReport.from_checks(checks)

        self._ok(checks, "config_parse", "Config", "Config YAML parsed successfully")

        # ── 2. Steps valid + canonical order ─────────────────────────────────
        raw_steps: list[str] = list(inp.steps) if inp.steps else list(DEFAULT_STEPS)
        unknown = [s for s in raw_steps if s not in PIPELINE_STEPS]
        if unknown:
            self._err(
                checks,
                "steps_valid",
                "Steps",
                f"Unknown step(s): {unknown}",
                detail=f"Valid steps: {list(PIPELINE_STEPS)}",
            )
            return DoctorReport.from_checks(checks)

        steps: list[str] = [s for s in PIPELINE_STEPS if s in raw_steps]
        steps_set: set[str] = set(steps)
        self._ok(checks, "steps_valid", "Steps", f"Steps: {', '.join(steps)}")

        # ── 3. Step dependency checks ─────────────────────────────────────────
        self._check_step_deps(inp, steps, steps_set, checks)

        # ── 4. Required fields per step ───────────────────────────────────────
        self._check_required_fields(inp, steps, steps_set, checks)

        # ── 5. File existence (only for explicitly-set paths) ─────────────────
        self._check_file_paths(inp, steps_set, checks)

        # ── 6. Azure / backend consistency ───────────────────────────────────
        self._check_azure(inp, steps_set, checks)

        return DoctorReport.from_checks(checks)

    # ── Step dependency checks ────────────────────────────────────────────────

    def _check_step_deps(
        self,
        inp: OrchestratorInput,
        steps: list[str],
        steps_set: set[str],
        checks: list[DoctorCheck],
    ) -> None:
        if "training_approval" in steps_set and "dataset_validation" not in steps_set:
            self._err(
                checks,
                "dep_training_approval",
                "Step dependencies",
                "training_approval requires dataset_validation to precede it",
            )

        if "evaluation" in steps_set and "training" not in steps_set:
            if inp.dataset_path:
                self._ok(
                    checks,
                    "dep_evaluation",
                    "Step dependencies",
                    "evaluation without training: standalone mode (dataset_path is set)",
                )
            else:
                self._warn(
                    checks,
                    "dep_evaluation",
                    "Step dependencies",
                    "evaluation without training: dataset_path should be set for "
                    "standalone evaluation",
                )

        if "model_decision" in steps_set and "evaluation" not in steps_set:
            self._err(
                checks,
                "dep_model_decision",
                "Step dependencies",
                "model_decision requires evaluation to precede it",
            )

        if "approval" in steps_set and "evaluation" not in steps_set:
            self._err(
                checks,
                "dep_approval",
                "Step dependencies",
                "approval requires evaluation to precede it",
            )

        if "model_registry" in steps_set and "approval" not in steps_set:
            self._warn(
                checks,
                "dep_model_registry",
                "Step dependencies",
                "model_registry without approval: H5 gate checks run inside "
                "ModelRegistryAgent, but no human approval step in this workflow",
                detail="Valid if an approval decision JSON was created in a prior run.",
            )

        if (
            "deployment" in steps_set
            and "model_registry" not in steps_set
            and "training" not in steps_set
        ):
            self._warn(
                checks,
                "dep_deployment",
                "Step dependencies",
                "deployment without model_registry or training: model weights must "
                "be available from a prior run",
            )

    # ── Required-field checks ─────────────────────────────────────────────────

    def _check_required_fields(
        self,
        inp: OrchestratorInput,
        steps: list[str],
        steps_set: set[str],
        checks: list[DoctorCheck],
    ) -> None:
        structured = "dataset_structuring" in steps_set

        if "data_intake" in steps_set:
            if inp.raw_data_path:
                self._ok(
                    checks,
                    "req_raw_data_path",
                    "Required fields",
                    f"raw_data_path: {inp.raw_data_path}",
                )
            else:
                self._err(
                    checks,
                    "req_raw_data_path",
                    "Required fields",
                    "data_intake requires raw_data_path",
                )

        if "dataset_structuring" in steps_set:
            ok_raw = bool(inp.raw_data_path)
            ok_cls = bool(inp.classes)
            if not ok_raw:
                self._err(
                    checks,
                    "req_structuring_raw",
                    "Required fields",
                    "dataset_structuring requires raw_data_path",
                )
            if not ok_cls:
                self._err(
                    checks,
                    "req_structuring_classes",
                    "Required fields",
                    "dataset_structuring requires classes (non-empty list)",
                )
            if ok_raw and ok_cls:
                self._ok(
                    checks,
                    "req_structuring",
                    "Required fields",
                    f"dataset_structuring inputs ok ({len(inp.classes)} class(es))",
                )

        if "dataset_validation" in steps_set and not structured:
            if inp.dataset_path:
                self._ok(
                    checks,
                    "req_dataset_path",
                    "Required fields",
                    f"dataset_path: {inp.dataset_path}",
                )
            else:
                self._err(
                    checks,
                    "req_dataset_path",
                    "Required fields",
                    "dataset_validation requires dataset_path " "(or dataset_structuring step)",
                )
            if inp.data_yaml_path:
                self._ok(
                    checks,
                    "req_data_yaml",
                    "Required fields",
                    f"data_yaml_path: {inp.data_yaml_path}",
                )
            else:
                self._err(
                    checks,
                    "req_data_yaml",
                    "Required fields",
                    "dataset_validation requires data_yaml_path " "(or dataset_structuring step)",
                )

        if "training" in steps_set:
            if inp.training_config_path:
                self._ok(
                    checks,
                    "req_training_config",
                    "Required fields",
                    f"training_config_path: {inp.training_config_path}",
                )
            else:
                self._err(
                    checks,
                    "req_training_config",
                    "Required fields",
                    "training requires training_config_path",
                )

        if "evaluation" in steps_set or "model_decision" in steps_set:
            if inp.promotion_policy_path:
                self._ok(
                    checks,
                    "req_promotion_policy",
                    "Required fields",
                    f"promotion_policy_path: {inp.promotion_policy_path}",
                )
            else:
                self._warn(
                    checks,
                    "req_promotion_policy",
                    "Required fields",
                    "promotion_policy_path not set: evaluation will use built-in "
                    "defaults (no hard mAP/precision/recall thresholds)",
                    detail="Set promotion_policy_path to enforce per-class thresholds.",
                )

        if "deployment" in steps_set:
            from agentic_mlops.contracts.deployment import DeploymentTarget  # noqa: PLC0415

            if inp.deployment_target == DeploymentTarget.PRODUCTION:
                if inp.production_approval_path:
                    self._ok(
                        checks,
                        "req_prod_approval",
                        "Required fields",
                        f"production_approval_path: {inp.production_approval_path}",
                    )
                else:
                    self._err(
                        checks,
                        "req_prod_approval",
                        "Required fields",
                        "deployment_target=production requires production_approval_path "
                        "(H6 gate)",
                    )
                if inp.rollback_plan:
                    self._ok(checks, "req_rollback_plan", "Required fields", "rollback_plan is set")
                else:
                    self._err(
                        checks,
                        "req_rollback_plan",
                        "Required fields",
                        "deployment_target=production requires rollback_plan",
                    )

    # ── File-existence checks ─────────────────────────────────────────────────

    def _check_file_paths(
        self,
        inp: OrchestratorInput,
        steps_set: set[str],
        checks: list[DoctorCheck],
    ) -> None:
        structured = "dataset_structuring" in steps_set

        # (check_name, path_value, label, error_if_missing)
        candidates: list[tuple[str, str | None, str, bool]] = [
            (
                "file_training_config",
                inp.training_config_path if "training" in steps_set else None,
                "training_config_path",
                True,
            ),
            # dataset_path / data_yaml_path only checked when structuring won't create them
            (
                "file_dataset_path",
                inp.dataset_path if not structured else None,
                "dataset_path",
                False,
            ),
            (
                "file_data_yaml",
                inp.data_yaml_path if not structured else None,
                "data_yaml_path",
                False,
            ),
            ("file_promotion_policy", inp.promotion_policy_path, "promotion_policy_path", False),
            ("file_evaluation_config", inp.evaluation_config_path, "evaluation_config_path", False),
            ("file_azure_config", inp.azure_config_path, "azure_config_path", True),
            ("file_baseline_report", inp.baseline_report_path, "baseline_report_path", False),
            ("file_prod_approval", inp.production_approval_path, "production_approval_path", False),
        ]

        for name, value, label, is_error in candidates:
            if not value:
                continue
            p = Path(value)
            if p.exists():
                self._ok(checks, name, "File paths", f"{label} found: {value}")
            else:
                (self._err if is_error else self._warn)(
                    checks, name, "File paths", f"{label} not found on disk: {value}"
                )

    # ── Azure / backend consistency ───────────────────────────────────────────

    def _check_azure(
        self,
        inp: OrchestratorInput,
        steps_set: set[str],
        checks: list[DoctorCheck],
    ) -> None:
        azure_runners = (
            inp.training_runner in ("azure-ml", "azure-ml-pipeline")
            or inp.evaluation_runner == "azure-ml"
        )
        azure_backends = (
            str(inp.registry_backend) == "azure_ml"
            or str(inp.dataset_registry_backend) == "azure_ml"
            or str(inp.deployment_backend) == "azure_ml"
        )
        needs_azure = azure_runners or azure_backends

        if needs_azure and not inp.azure_config_path:
            self._err(
                checks,
                "azure_config_required",
                "Azure",
                "azure_config_path is required for the configured Azure backends/runners",
                detail=(
                    f"training_runner={inp.training_runner}, "
                    f"evaluation_runner={inp.evaluation_runner}, "
                    f"registry_backend={inp.registry_backend}, "
                    f"deployment_backend={inp.deployment_backend}"
                ),
            )
        elif needs_azure and inp.azure_config_path:
            p = Path(inp.azure_config_path)
            if not p.exists():
                pass  # already caught by file_paths check
            else:
                try:
                    from agentic_mlops.contracts.azure_ml import AzureMLConfig  # noqa: PLC0415

                    AzureMLConfig.from_yaml(inp.azure_config_path)
                    self._ok(
                        checks,
                        "azure_config_valid",
                        "Azure",
                        f"azure_config_path valid: {inp.azure_config_path}",
                    )
                except Exception as exc:
                    self._err(
                        checks, "azure_config_valid", "Azure", f"azure_config_path invalid: {exc}"
                    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ok(self, checks: list, name: str, category: str, message: str) -> None:
        checks.append(
            DoctorCheck(name=name, category=category, status=CheckStatus.OK, message=message)
        )

    def _warn(
        self, checks: list, name: str, category: str, message: str, detail: str | None = None
    ) -> None:
        checks.append(
            DoctorCheck(
                name=name,
                category=category,
                status=CheckStatus.WARNING,
                message=message,
                detail=detail,
            )
        )

    def _err(
        self, checks: list, name: str, category: str, message: str, detail: str | None = None
    ) -> None:
        checks.append(
            DoctorCheck(
                name=name,
                category=category,
                status=CheckStatus.ERROR,
                message=message,
                detail=detail,
            )
        )
