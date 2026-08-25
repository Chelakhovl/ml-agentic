"""Docker image build + push client.

Two implementations:
  - DockerClient  — real subprocess calls to the ``docker`` CLI
  - FakeDockerClient — in-memory test double; records calls in .built / .pushed

DockerClient.build() and .push() raise subprocess.CalledProcessError on failure
(check=True). The caller (ModelDeployer._deploy_docker) catches them and converts
them to a DeploymentOutput(success=False).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)


class DockerClient:
    """Wraps docker build + docker push via subprocess."""

    def build(self, context_dir: Path, tag: str) -> None:
        logger.info("docker build", extra={"tag": tag, "context": str(context_dir)})
        subprocess.run(
            ["docker", "build", "-t", tag, "."],
            cwd=str(context_dir),
            check=True,
        )

    def push(self, tag: str) -> None:
        logger.info("docker push", extra={"tag": tag})
        subprocess.run(["docker", "push", tag], check=True)


class FakeDockerClient:
    """In-memory test double for DockerClient."""

    def __init__(self) -> None:
        self.built: list[tuple[Path, str]] = []
        self.pushed: list[str] = []

    def build(self, context_dir: Path, tag: str) -> None:
        self.built.append((context_dir, tag))

    def push(self, tag: str) -> None:
        self.pushed.append(tag)
