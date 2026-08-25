"""Unit tests for SystemDoctor and the `doctor` CLI command."""

from __future__ import annotations

from agentic_mlops.contracts.doctor import CheckStatus
from agentic_mlops.tools.doctor import SystemDoctor


class TestDoctorReport:
    def test_overall_ok_when_all_checks_pass(self, tmp_path):
        report = SystemDoctor().check(
            config_paths=[],
            directory_paths=[(tmp_path, "test_dir")],
            check_docker=False,
            check_az=False,
        )
        # required packages always present; optional may warn — overall is ok/warning, not error
        assert report.overall_status in (CheckStatus.OK, CheckStatus.WARNING)
        assert report.num_errors == 0

    def test_error_for_missing_config_file(self, tmp_path):
        report = SystemDoctor().check(
            config_paths=[(tmp_path / "nonexistent.yaml", "test config")],
            directory_paths=[],
            check_docker=False,
            check_az=False,
        )
        errors = [
            c for c in report.checks if c.status == CheckStatus.ERROR and c.category == "configs"
        ]
        assert len(errors) == 1
        assert "not found" in errors[0].message

    def test_ok_for_valid_yaml_config(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("workflow_id: test\ndry_run: true\n")
        report = SystemDoctor().check(
            config_paths=[(cfg, "my config")],
            directory_paths=[],
            check_docker=False,
            check_az=False,
        )
        ok_cfg = [
            c for c in report.checks if c.category == "configs" and c.status == CheckStatus.OK
        ]
        assert len(ok_cfg) == 1
        assert "2 top-level key(s)" in ok_cfg[0].message

    def test_warning_for_empty_yaml_config(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")
        report = SystemDoctor().check(
            config_paths=[(cfg, "empty config")],
            directory_paths=[],
            check_docker=False,
            check_az=False,
        )
        warns = [
            c for c in report.checks if c.category == "configs" and c.status == CheckStatus.WARNING
        ]
        assert len(warns) == 1
        assert "empty" in warns[0].message

    def test_error_for_invalid_yaml_config(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("key: [unclosed bracket\n")
        report = SystemDoctor().check(
            config_paths=[(cfg, "bad config")],
            directory_paths=[],
            check_docker=False,
            check_az=False,
        )
        errors = [
            c for c in report.checks if c.category == "configs" and c.status == CheckStatus.ERROR
        ]
        assert len(errors) == 1

    def test_ok_for_writable_directory(self, tmp_path):
        report = SystemDoctor().check(
            directory_paths=[(tmp_path / "newdir", "mydir")],
            check_docker=False,
            check_az=False,
        )
        dir_checks = [c for c in report.checks if c.category == "directories"]
        assert any(c.status == CheckStatus.OK and "mydir" in c.message for c in dir_checks)

    def test_required_packages_all_present(self, tmp_path):
        report = SystemDoctor().check(check_docker=False, check_az=False)
        req = [
            c
            for c in report.checks
            if c.category == "packages"
            and c.name in ("package:pydantic", "package:pyyaml", "package:typer", "package:rich")
        ]
        bad = [c for c in req if c.status != CheckStatus.OK]
        assert all(c.status == CheckStatus.OK for c in req), bad

    def test_no_tool_checks_when_disabled(self, tmp_path):
        report = SystemDoctor().check(check_docker=False, check_az=False)
        tool_checks = [c for c in report.checks if c.category == "tools"]
        assert tool_checks == []

    def test_tool_checks_included_when_enabled(self, tmp_path):
        report = SystemDoctor().check(check_docker=True, check_az=True)
        tool_checks = [c for c in report.checks if c.category == "tools"]
        names = {c.name for c in tool_checks}
        assert "tool:docker" in names
        assert "tool:az" in names
        # Missing tools should warn (not error) since they're optional
        for c in tool_checks:
            assert c.status in (CheckStatus.OK, CheckStatus.WARNING)

    def test_from_checks_overall_error_when_any_error(self):
        from agentic_mlops.contracts.doctor import DoctorCheck, DoctorReport

        checks = [
            DoctorCheck(name="a", category="packages", status=CheckStatus.OK, message="ok"),
            DoctorCheck(name="b", category="packages", status=CheckStatus.ERROR, message="bad"),
        ]
        report = DoctorReport.from_checks(checks)
        assert report.overall_status == CheckStatus.ERROR
        assert report.num_ok == 1
        assert report.num_errors == 1

    def test_from_checks_overall_warning_when_no_error(self):
        from agentic_mlops.contracts.doctor import DoctorCheck, DoctorReport

        checks = [
            DoctorCheck(name="a", category="packages", status=CheckStatus.OK, message="ok"),
            DoctorCheck(name="b", category="packages", status=CheckStatus.WARNING, message="warn"),
        ]
        report = DoctorReport.from_checks(checks)
        assert report.overall_status == CheckStatus.WARNING


class TestDoctorCLI:
    def test_exits_0_on_success(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(
            app,
            [
                "doctor",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--registry-dir",
                str(tmp_path / "registry"),
                "--dataset-registry-dir",
                str(tmp_path / "dataset_registry"),
                "--no-tools",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "doctor" in result.output.lower()

    def test_exits_1_on_missing_config(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(
            app,
            [
                "doctor",
                "--config",
                str(tmp_path / "no_such_config.yaml"),
                "--no-tools",
            ],
        )
        assert result.exit_code == 1

    def test_validates_passed_config(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        cfg = tmp_path / "orch.yaml"
        cfg.write_text("dry_run: true\nmodel_name: my-model\n")

        result = CliRunner().invoke(
            app,
            [
                "doctor",
                "--config",
                str(cfg),
                "--no-tools",
            ],
        )
        assert "orchestrator config" in result.output
        assert "2 top-level key(s)" in result.output

    def test_output_contains_section_headers(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(app, ["doctor", "--no-tools"])
        assert "Python Packages" in result.output
        assert "Directories" in result.output

    def test_summary_line_present(self, tmp_path):
        from typer.testing import CliRunner

        from agentic_mlops.cli.main import app

        result = CliRunner().invoke(app, ["doctor", "--no-tools"])
        assert "Summary:" in result.output
        assert "ok" in result.output
