"""DocsLinter — checks that spec docs in agentic_mlops_workflow_docs/ are
kept in sync with the codebase and carry a living-spec header.

Checks performed:
  header       — every spec file has a living-spec banner
  paths        — Python file paths mentioned in specs exist under src/
  classes      — agent/workflow class names in specs exist in Python source
  claude_sync  — each agent spec (01–12) is mentioned in CLAUDE.md
"""

from __future__ import annotations

import re
from pathlib import Path

from agentic_mlops.contracts.doctor import CheckStatus, DoctorCheck, DoctorReport

_LIVING_SPEC_MARKER = "**Living Spec**"

# Pattern matching Python paths relative to src/agentic_mlops
_PATH_RE = re.compile(r'\b(agents|tools|contracts|integrations|workflows|cli|web)/[\w/]+\.py\b')

# Pattern matching likely class names (PascalCase with >=2 words, ends in known suffixes)
_CLASS_RE = re.compile(
    r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+(?:Agent|Workflow|Runner|Client|Store|Linter|Checker|Scanner|Ingester|Deployer|Registry|Comparator|Scorer|Resolver|Tracker|Monitor|Labeler|Structurer|Validator|Versioning|Approval|Decision|Evaluation|Annotation))\b'
)

# CLAUDE.md sections to skip (they're not agent-specific)
_IGNORED_DOCS = {"13_backlog.md", "00_architecture_overview.md"}


class DocsLinter:
    """Validates agentic_mlops_workflow_docs/ spec files."""

    def lint(
        self,
        docs_dir: str | Path,
        src_dir: str | Path,
        claude_md_path: str | Path | None = None,
        strict: bool = False,
    ) -> DoctorReport:
        """Run all checks.

        Args:
            docs_dir: Root of ``agentic_mlops_workflow_docs/`` (contains
                ``agents/`` and ``docs/`` subdirs).
            src_dir: Root of the Python source tree (typically ``src/``).
            claude_md_path: Path to ``CLAUDE.md`` for cross-reference checks.
                When ``None`` the check is skipped.
            strict: When ``True``, warnings count as errors for the
                ``overall_status`` calculation.
        """
        docs_dir = Path(docs_dir)
        src_dir = Path(src_dir)
        checks: list[DoctorCheck] = []

        spec_files = sorted(
            p
            for p in [*docs_dir.glob("agents/*.md"), *docs_dir.glob("docs/*.md")]
            if p.name not in _IGNORED_DOCS
        )

        if not spec_files:
            self._err(checks, "docs_found", "Structure", f"No spec files found in {docs_dir}")
            return DoctorReport.from_checks(checks)

        self._ok(checks, "docs_found", "Structure", f"{len(spec_files)} spec file(s) found")

        # ── 1. Living-spec header ─────────────────────────────────────────────
        missing_header: list[str] = []
        for spec in spec_files:
            text = spec.read_text(encoding="utf-8", errors="replace")
            if _LIVING_SPEC_MARKER not in text:
                missing_header.append(spec.name)
        if missing_header:
            self._warn(
                checks,
                "header",
                "Header",
                f"{len(missing_header)} spec file(s) missing living-spec banner",
                detail=", ".join(missing_header),
            )
        else:
            self._ok(checks, "header", "Header", "All spec files carry a living-spec banner")

        # ── 2. Python path references ─────────────────────────────────────────
        src_pkg = src_dir / "agentic_mlops"
        bad_paths: list[str] = []
        for spec in spec_files:
            text = spec.read_text(encoding="utf-8", errors="replace")
            for m in _PATH_RE.finditer(text):
                rel = m.group(0)
                full = src_pkg / rel
                if not full.exists():
                    bad_paths.append(f"{spec.name}: {rel}")
        if bad_paths:
            self._warn(
                checks,
                "paths",
                "Paths",
                f"{len(bad_paths)} referenced path(s) not found in src/",
                detail="\n".join(bad_paths[:20]),
            )
        else:
            self._ok(checks, "paths", "Paths", "All referenced Python paths exist in src/")

        # ── 3. Class name references ──────────────────────────────────────────
        missing_classes: list[str] = []
        for spec in spec_files:
            text = spec.read_text(encoding="utf-8", errors="replace")
            for m in _CLASS_RE.finditer(text):
                class_name = m.group(1)
                found = _grep_class(class_name, src_pkg)
                if not found:
                    missing_classes.append(f"{spec.name}: {class_name}")
        if missing_classes:
            self._warn(
                checks,
                "classes",
                "Classes",
                f"{len(missing_classes)} class reference(s) not found in src/",
                detail="\n".join(missing_classes[:20]),
            )
        else:
            self._ok(
                checks, "classes", "Classes", "All class references resolved in src/"
            )

        # ── 4. CLAUDE.md cross-reference ──────────────────────────────────────
        if claude_md_path is not None:
            claude_md = Path(claude_md_path)
            if not claude_md.exists():
                self._warn(
                    checks,
                    "claude_sync",
                    "CLAUDE.md",
                    f"CLAUDE.md not found at {claude_md}",
                )
            else:
                claude_text = claude_md.read_text(encoding="utf-8", errors="replace")
                agent_specs = [p for p in spec_files if p.parent.name == "agents"]
                unmentioned: list[str] = []
                for spec in agent_specs:
                    # e.g. 01_data_intake_agent.md → DataIntakeAgent or data_intake
                    stem = spec.stem  # "01_data_intake_agent"
                    # Extract the meaningful part after the two-digit prefix
                    bare = re.sub(r"^\d+_", "", stem)  # "data_intake_agent"
                    # Check for any reasonable reference: the bare name, or the
                    # PascalCase form, or the Python filename without extension
                    pascal = "".join(w.capitalize() for w in bare.split("_"))
                    if bare not in claude_text and pascal not in claude_text:
                        unmentioned.append(spec.name)
                if unmentioned:
                    self._warn(
                        checks,
                        "claude_sync",
                        "CLAUDE.md",
                        f"{len(unmentioned)} agent spec(s) not mentioned in CLAUDE.md",
                        detail=", ".join(unmentioned),
                    )
                else:
                    self._ok(
                        checks,
                        "claude_sync",
                        "CLAUDE.md",
                        "All agent specs are referenced in CLAUDE.md",
                    )

        report = DoctorReport.from_checks(checks)
        if strict and report.overall_status == CheckStatus.WARNING:
            # In strict mode, degrade overall to ERROR so the CLI exits 1.
            object.__setattr__(report, "overall_status", CheckStatus.ERROR)
        return report

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _ok(checks: list, name: str, cat: str, msg: str, detail: str | None = None) -> None:
        checks.append(
            DoctorCheck(name=name, category=cat, status=CheckStatus.OK, message=msg, detail=detail)
        )

    @staticmethod
    def _warn(checks: list, name: str, cat: str, msg: str, detail: str | None = None) -> None:
        checks.append(
            DoctorCheck(
                name=name, category=cat, status=CheckStatus.WARNING, message=msg, detail=detail
            )
        )

    @staticmethod
    def _err(checks: list, name: str, cat: str, msg: str, detail: str | None = None) -> None:
        checks.append(
            DoctorCheck(
                name=name, category=cat, status=CheckStatus.ERROR, message=msg, detail=detail
            )
        )


def _grep_class(class_name: str, src_root: Path) -> bool:
    """Return True if ``class_name`` appears in any .py file under src_root."""
    pattern = f"class {class_name}"
    for py_file in src_root.rglob("*.py"):
        try:
            if pattern in py_file.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            pass
    return False
