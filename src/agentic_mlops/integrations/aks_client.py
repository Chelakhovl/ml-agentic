"""kubectl-based AKS deployer client.

Two implementations:
  - AksClient      — real subprocess calls to ``kubectl``
  - FakeAksClient  — in-memory test double; records manifests applied

AksClient.apply() raises subprocess.CalledProcessError on failure (check=True).
The caller (ModelDeployer._deploy_aks) catches it and converts to
DeploymentOutput(success=False).
"""

from __future__ import annotations

import json
import subprocess

from agentic_mlops.contracts.docker import AksConfig
from agentic_mlops.observability.logging import get_logger

logger = get_logger(__name__)


class AksClient:
    """Wraps kubectl apply and kubectl get service via subprocess."""

    def __init__(self, config: AksConfig) -> None:
        self._config = config

    def _base_cmd(self) -> list[str]:
        cmd = ["kubectl"]
        if self._config.kubeconfig_path:
            cmd += ["--kubeconfig", self._config.kubeconfig_path]
        if self._config.context:
            cmd += ["--context", self._config.context]
        return cmd

    def apply(self, manifest_yaml: str) -> None:
        logger.info("kubectl apply manifest")
        subprocess.run(
            self._base_cmd() + ["apply", "-f", "-"],
            input=manifest_yaml.encode(),
            check=True,
        )

    def get_service_url(self, namespace: str, service_name: str) -> str | None:
        """Return the external URL for a LoadBalancer service, or None."""
        try:
            result = subprocess.run(
                self._base_cmd() + ["get", "service", service_name, "-n", namespace, "-o", "json"],
                capture_output=True,
                check=True,
            )
            data = json.loads(result.stdout)
            ingress = data.get("status", {}).get("loadBalancer", {}).get("ingress", [])
            if ingress:
                host = ingress[0].get("ip") or ingress[0].get("hostname")
                port = self._config.port
                if host:
                    return f"http://{host}:{port}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not retrieve service URL", extra={"error": str(exc)})
        return None


class FakeAksClient:
    """In-memory test double for AksClient."""

    def __init__(self, service_url: str = "http://fake-aks.mlops.svc:8080") -> None:
        self.applied: list[str] = []
        self._service_url = service_url

    def apply(self, manifest_yaml: str) -> None:
        self.applied.append(manifest_yaml)

    def get_service_url(self, namespace: str, service_name: str) -> str | None:
        return self._service_url
