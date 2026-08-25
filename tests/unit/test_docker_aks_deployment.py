"""Unit tests for Docker image build and AKS deployment backends."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_mlops.contracts.deployment import (
    DeploymentBackend,
    DeploymentInput,
    DeploymentStatus,
)
from agentic_mlops.contracts.docker import AksConfig, DockerConfig
from agentic_mlops.integrations.aks_client import AksClient, FakeAksClient
from agentic_mlops.integrations.docker_client import DockerClient, FakeDockerClient
from agentic_mlops.tools.deployer import (
    ModelDeployer,
    _generate_dockerfile,
    _generate_k8s_manifest,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _minimal_pt(tmp_path: Path, name: str = "best.pt") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x80\x02}q\x00.")  # minimal non-empty bytes
    return p


def _docker_cfg(**kwargs) -> DockerConfig:
    defaults = dict(registry="myacr.azurecr.io", base_image="python:3.11-slim")
    defaults.update(kwargs)
    return DockerConfig(**defaults)


def _aks_cfg(**kwargs) -> AksConfig:
    defaults: dict = {}
    defaults.update(kwargs)
    return AksConfig(**defaults)


def _inp(tmp_path: Path, backend: str = "docker", **kwargs) -> DeploymentInput:
    model_path = _minimal_pt(tmp_path)
    base = dict(
        model_path=str(model_path),
        model_name="my-yolo",
        model_version=3,
        backend=DeploymentBackend(backend),
        export_format="pt",
    )
    base.update(kwargs)
    return DeploymentInput(**base)


def _deployer(
    tmp_path: Path,
    fake_docker: FakeDockerClient,
    fake_aks: FakeAksClient | None = None,
    aks_cfg: AksConfig | None = None,
) -> ModelDeployer:
    return ModelDeployer(
        docker_client=fake_docker,
        docker_config=_docker_cfg(),
        aks_client=fake_aks,
        aks_config=aks_cfg or (AksConfig() if fake_aks else None),
    )


# ── DockerConfig helpers ──────────────────────────────────────────────────────


class TestDockerConfig:
    def test_image_tag_default_name(self):
        cfg = DockerConfig(registry="myacr.azurecr.io")
        assert cfg.image_tag("my-yolo", "v3") == "myacr.azurecr.io/my-yolo:v3"

    def test_image_tag_custom_name(self):
        cfg = DockerConfig(registry="myacr.azurecr.io", image_name="model-server")
        assert cfg.image_tag("my-yolo", "v3") == "myacr.azurecr.io/model-server:v3"

    def test_image_tag_prefix(self):
        cfg = DockerConfig(registry="myacr.azurecr.io", image_prefix="mlops/")
        assert cfg.image_tag("yolo", "latest") == "myacr.azurecr.io/mlops/yolo:latest"

    def test_image_tag_underscore_replaced(self):
        cfg = DockerConfig(registry="myacr.azurecr.io")
        tag = cfg.image_tag("my_yolo_model", "v1")
        assert "my-yolo-model" in tag

    def test_from_yaml(self, tmp_path):
        p = tmp_path / "docker.yaml"
        p.write_text("registry: myacr.azurecr.io\nport: 9000\n", encoding="utf-8")
        cfg = DockerConfig.from_yaml(p)
        assert cfg.registry == "myacr.azurecr.io"
        assert cfg.port == 9000


# ── AksConfig helpers ─────────────────────────────────────────────────────────


class TestAksConfig:
    def test_from_yaml(self, tmp_path):
        p = tmp_path / "aks.yaml"
        p.write_text("namespace: production\nreplicas: 3\n", encoding="utf-8")
        cfg = AksConfig.from_yaml(p)
        assert cfg.namespace == "production"
        assert cfg.replicas == 3


# ── FakeDockerClient ──────────────────────────────────────────────────────────


class TestFakeDockerClient:
    def test_build_records(self, tmp_path):
        fake = FakeDockerClient()
        fake.build(tmp_path, "myacr.azurecr.io/model:v1")
        assert len(fake.built) == 1
        assert fake.built[0] == (tmp_path, "myacr.azurecr.io/model:v1")

    def test_push_records(self):
        fake = FakeDockerClient()
        fake.push("myacr.azurecr.io/model:v1")
        assert fake.pushed == ["myacr.azurecr.io/model:v1"]


# ── FakeAksClient ─────────────────────────────────────────────────────────────


class TestFakeAksClient:
    def test_apply_records_manifest(self):
        fake = FakeAksClient()
        fake.apply("kind: Deployment\n")
        assert len(fake.applied) == 1
        assert "Deployment" in fake.applied[0]

    def test_get_service_url(self):
        fake = FakeAksClient(service_url="http://10.0.0.1:8080")
        url = fake.get_service_url("mlops", "my-model")
        assert url == "http://10.0.0.1:8080"


# ── DockerClient subprocess ───────────────────────────────────────────────────


class TestDockerClient:
    def test_build_calls_subprocess(self, tmp_path):
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0)
            DockerClient().build(tmp_path, "myacr.azurecr.io/model:v1")
        mock.assert_called_once_with(
            ["docker", "build", "-t", "myacr.azurecr.io/model:v1", "."],
            cwd=str(tmp_path),
            check=True,
        )

    def test_push_calls_subprocess(self):
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0)
            DockerClient().push("myacr.azurecr.io/model:v1")
        mock.assert_called_once_with(["docker", "push", "myacr.azurecr.io/model:v1"], check=True)

    def test_build_raises_on_failure(self, tmp_path):
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "docker")):
            with pytest.raises(subprocess.CalledProcessError):
                DockerClient().build(tmp_path, "tag:v1")


# ── AksClient subprocess ──────────────────────────────────────────────────────


class TestAksClient:
    def _client(self, **kwargs) -> AksClient:
        return AksClient(AksConfig(**kwargs))

    def test_apply_pipes_manifest_to_stdin(self):
        manifest = "kind: Deployment\n"
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0)
            self._client().apply(manifest)
        mock.assert_called_once_with(
            ["kubectl", "apply", "-f", "-"],
            input=manifest.encode(),
            check=True,
        )

    def test_apply_includes_kubeconfig(self):
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0)
            AksClient(AksConfig(kubeconfig_path="/home/.kube/config")).apply("x")
        cmd = mock.call_args[0][0]
        assert "--kubeconfig" in cmd
        assert "/home/.kube/config" in cmd

    def test_apply_includes_context(self):
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0)
            AksClient(AksConfig(context="prod-ctx")).apply("x")
        cmd = mock.call_args[0][0]
        assert "--context" in cmd
        assert "prod-ctx" in cmd

    def test_get_service_url_loadbalancer(self):
        response = json.dumps({"status": {"loadBalancer": {"ingress": [{"ip": "1.2.3.4"}]}}})
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0, stdout=response.encode())
            url = self._client(service_type="LoadBalancer").get_service_url("mlops", "svc")
        assert url == "http://1.2.3.4:8080"

    def test_get_service_url_no_ingress_returns_none(self):
        response = json.dumps({"status": {"loadBalancer": {}}})
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0, stdout=response.encode())
            url = self._client().get_service_url("mlops", "svc")
        assert url is None

    def test_get_service_url_exception_returns_none(self):
        with patch("subprocess.run", side_effect=Exception("kubectl not found")):
            url = self._client().get_service_url("mlops", "svc")
        assert url is None


# ── Template generators ───────────────────────────────────────────────────────


class TestGenerators:
    def test_dockerfile_contains_base_image(self):
        df = _generate_dockerfile("python:3.11-slim", [], 8080)
        assert "FROM python:3.11-slim" in df

    def test_dockerfile_extra_requirements(self):
        df = _generate_dockerfile("python:3.11-slim", ["torch==2.0", "onnx"], 8080)
        assert "torch==2.0" in df
        assert "onnx" in df

    def test_dockerfile_exposes_port(self):
        df = _generate_dockerfile("python:3.11-slim", [], 9090)
        assert "EXPOSE 9090" in df

    def test_k8s_manifest_has_deployment_and_service(self):
        m = _generate_k8s_manifest(
            deployment_name="my-yolo-staging",
            namespace="mlops",
            image="myacr.azurecr.io/my-yolo:v3",
            replicas=2,
            port=8080,
            service_type="LoadBalancer",
            cpu_request="500m",
            memory_request="1Gi",
            cpu_limit="2000m",
            memory_limit="2Gi",
        )
        assert "kind: Deployment" in m
        assert "kind: Service" in m
        assert "my-yolo-staging" in m
        assert "mlops" in m
        assert "myacr.azurecr.io/my-yolo:v3" in m
        assert "LoadBalancer" in m
        assert "replicas: 2" in m


# ── ModelDeployer.docker backend ──────────────────────────────────────────────


class TestModelDeployerDocker:
    def test_docker_deploy_calls_build_and_push(self, tmp_path):
        fake = FakeDockerClient()
        result = _deployer(tmp_path, fake).deploy(
            _inp(tmp_path, backend="docker"), artifacts_dir=tmp_path / "out"
        )
        assert result.success
        assert len(fake.built) == 1
        assert len(fake.pushed) == 1
        assert fake.built[0][1] == fake.pushed[0]

    def test_image_tag_uses_model_version(self, tmp_path):
        fake = FakeDockerClient()
        _deployer(tmp_path, fake).deploy(
            _inp(tmp_path, backend="docker", model_version=7),
            artifacts_dir=tmp_path / "out",
        )
        assert "v7" in fake.pushed[0]

    def test_image_tag_latest_when_no_version(self, tmp_path):
        fake = FakeDockerClient()
        _deployer(tmp_path, fake).deploy(
            _inp(tmp_path, backend="docker", model_version=None),
            artifacts_dir=tmp_path / "out",
        )
        assert "latest" in fake.pushed[0]

    def test_output_has_image_tag(self, tmp_path):
        fake = FakeDockerClient()
        result = _deployer(tmp_path, fake).deploy(
            _inp(tmp_path, backend="docker"), artifacts_dir=tmp_path / "out"
        )
        assert result.image_tag is not None
        assert "myacr.azurecr.io" in result.image_tag

    def test_manifest_written_to_artifacts_dir(self, tmp_path):
        fake = FakeDockerClient()
        out = tmp_path / "out"
        _deployer(tmp_path, fake).deploy(_inp(tmp_path, backend="docker"), artifacts_dir=out)
        assert (out / "docker_deployment_manifest.json").exists()

    def test_manifest_contains_image_tag(self, tmp_path):
        fake = FakeDockerClient()
        out = tmp_path / "out"
        _deployer(tmp_path, fake).deploy(_inp(tmp_path, backend="docker"), artifacts_dir=out)
        data = json.loads((out / "docker_deployment_manifest.json").read_text())
        assert data["image_tag"] == fake.pushed[0]

    def test_dockerfile_written_to_build_context(self, tmp_path):
        fake = FakeDockerClient()
        out = tmp_path / "out"
        _deployer(tmp_path, fake).deploy(_inp(tmp_path, backend="docker"), artifacts_dir=out)
        assert (out / "docker_build" / "Dockerfile").exists()

    def test_score_py_written_to_build_context(self, tmp_path):
        fake = FakeDockerClient()
        out = tmp_path / "out"
        _deployer(tmp_path, fake).deploy(_inp(tmp_path, backend="docker"), artifacts_dir=out)
        assert (out / "docker_build" / "score.py").exists()

    def test_status_staging(self, tmp_path):
        fake = FakeDockerClient()
        result = _deployer(tmp_path, fake).deploy(
            _inp(tmp_path, backend="docker", target="staging"),
            artifacts_dir=tmp_path / "out",
        )
        assert result.status == DeploymentStatus.DEPLOYED_TO_STAGING

    def test_missing_docker_client_returns_failed(self, tmp_path):
        deployer = ModelDeployer()  # no docker_client injected
        result = deployer.deploy(_inp(tmp_path, backend="docker"), artifacts_dir=tmp_path / "out")
        assert not result.success
        assert result.status == DeploymentStatus.FAILED

    def test_missing_model_path_returns_failed(self, tmp_path):
        fake = FakeDockerClient()
        deployer = ModelDeployer(docker_client=fake, docker_config=_docker_cfg())
        result = deployer.deploy(
            DeploymentInput(
                model_name="yolo",
                backend=DeploymentBackend.DOCKER,
                export_format="pt",
            ),
            artifacts_dir=tmp_path / "out",
        )
        assert not result.success

    def test_docker_build_failure_returns_failed(self, tmp_path):
        class FailingDocker:
            def build(self, context_dir, tag):
                raise subprocess.CalledProcessError(1, "docker build")

            def push(self, tag):
                pass

        deployer = ModelDeployer(docker_client=FailingDocker(), docker_config=_docker_cfg())
        result = deployer.deploy(_inp(tmp_path, backend="docker"), artifacts_dir=tmp_path / "out")
        assert not result.success
        assert "Docker" in result.message

    def test_production_gate_blocks_without_approval(self, tmp_path):
        fake = FakeDockerClient()
        result = _deployer(tmp_path, fake).deploy(
            _inp(tmp_path, backend="docker", target="production"),
            artifacts_dir=tmp_path / "out",
        )
        assert result.status == DeploymentStatus.BLOCKED

    def test_production_gate_passes_with_approval(self, tmp_path):
        approval = tmp_path / "approval.json"
        approval.write_text(json.dumps({"status": "approved"}), encoding="utf-8")
        fake = FakeDockerClient()
        result = _deployer(tmp_path, fake).deploy(
            _inp(
                tmp_path,
                backend="docker",
                target="production",
                production_approval_path=str(approval),
                rollback_plan="revert to v2",
            ),
            artifacts_dir=tmp_path / "out",
        )
        assert result.success
        assert result.status == DeploymentStatus.DEPLOYED_TO_PRODUCTION


# ── ModelDeployer.aks backend ─────────────────────────────────────────────────


class TestModelDeployerAks:
    def test_aks_deploy_calls_apply(self, tmp_path):
        fake_docker = FakeDockerClient()
        fake_aks = FakeAksClient()
        result = _deployer(tmp_path, fake_docker, fake_aks).deploy(
            _inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out"
        )
        assert result.success
        assert len(fake_aks.applied) == 1

    def test_aks_manifest_contains_deployment_and_service(self, tmp_path):
        fake_docker = FakeDockerClient()
        fake_aks = FakeAksClient()
        _deployer(tmp_path, fake_docker, fake_aks).deploy(
            _inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out"
        )
        manifest = fake_aks.applied[0]
        assert "kind: Deployment" in manifest
        assert "kind: Service" in manifest

    def test_aks_also_builds_and_pushes_docker(self, tmp_path):
        fake_docker = FakeDockerClient()
        fake_aks = FakeAksClient()
        _deployer(tmp_path, fake_docker, fake_aks).deploy(
            _inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out"
        )
        assert len(fake_docker.built) == 1
        assert len(fake_docker.pushed) == 1

    def test_aks_scoring_uri_in_output(self, tmp_path):
        fake_docker = FakeDockerClient()
        fake_aks = FakeAksClient(service_url="http://10.0.0.5:8080")
        result = _deployer(tmp_path, fake_docker, fake_aks).deploy(
            _inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out"
        )
        assert result.scoring_uri == "http://10.0.0.5:8080"

    def test_aks_k8s_manifest_written_to_disk(self, tmp_path):
        fake_docker = FakeDockerClient()
        fake_aks = FakeAksClient()
        out = tmp_path / "out"
        _deployer(tmp_path, fake_docker, fake_aks).deploy(
            _inp(tmp_path, backend="aks"), artifacts_dir=out
        )
        assert (out / "k8s_manifest.yaml").exists()

    def test_aks_missing_aks_client_returns_failed(self, tmp_path):
        deployer = ModelDeployer(
            docker_client=FakeDockerClient(), docker_config=_docker_cfg()
        )  # no aks_client
        result = deployer.deploy(_inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out")
        assert not result.success
        assert "aks_client" in result.message

    def test_aks_apply_failure_returns_failed(self, tmp_path):
        class FailingAks:
            def apply(self, manifest):
                raise subprocess.CalledProcessError(1, "kubectl")

            def get_service_url(self, namespace, name):
                return None

        deployer = ModelDeployer(
            docker_client=FakeDockerClient(),
            docker_config=_docker_cfg(),
            aks_client=FailingAks(),
            aks_config=AksConfig(),
        )
        result = deployer.deploy(_inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out")
        assert not result.success
        assert "AKS" in result.message

    def test_aks_uses_custom_namespace(self, tmp_path):
        fake_docker = FakeDockerClient()
        fake_aks = FakeAksClient()
        deployer = ModelDeployer(
            docker_client=fake_docker,
            docker_config=_docker_cfg(),
            aks_client=fake_aks,
            aks_config=AksConfig(namespace="prod-ns"),
        )
        deployer.deploy(_inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out")
        assert "prod-ns" in fake_aks.applied[0]

    def test_aks_manifest_contains_image_tag(self, tmp_path):
        fake_docker = FakeDockerClient()
        fake_aks = FakeAksClient()
        _deployer(tmp_path, fake_docker, fake_aks).deploy(
            _inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out"
        )
        assert "myacr.azurecr.io" in fake_aks.applied[0]

    def test_aks_k8s_deployment_name_in_output(self, tmp_path):
        fake_docker = FakeDockerClient()
        fake_aks = FakeAksClient()
        result = _deployer(tmp_path, fake_docker, fake_aks).deploy(
            _inp(tmp_path, backend="aks"), artifacts_dir=tmp_path / "out"
        )
        assert result.k8s_deployment_name is not None
