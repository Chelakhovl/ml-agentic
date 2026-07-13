# Клиентская инфраструктура проекта — текущее состояние

> Составлено 2026-07-11 по факту чтения исходного кода (не по памяти/докам —
> все пути к файлам и имена классов проверены `grep`/`Read` непосредственно
> перед записью этого документа). Все 489 unit-тестов проходят на момент
> написания. Обновлено 2026-07-13: (1) исправлена найденная в этом же
> документе нерабочая `azure_integration` pytest-команда — см. раздел 10;
> (2) реализован Azure ML Data Asset backend для Dataset Registry (был
> единственным реестром без Azure-варианта) — см. раздел 8. 500/500
> unit-тестов на текущий момент.

Документ описывает: (1) все "клиенты" — обёртки над внешними системами
(Azure ML, MLflow, локальные реестры), (2) что из них реально работает, а
что — заглушка/не реализовано, (3) что делать дальше, (4) как всё это
протестировать прямо сейчас, включая то, что для тестирования нужны
реальные учётные данные.

---

## 1. Карта всех клиентов

| # | Клиент | Файл | Статус |
|---|--------|------|--------|
| 1 | Azure ML SDK-клиент (низкий уровень) | `integrations/azure_ml_client.py` | ✅ реальный + Fake |
| 2 | Azure ML Training Runner | `tools/training_runner.py` | ✅ реальный (local + azure) |
| 3 | Azure ML Evaluation Runner | `tools/evaluation_runner.py` | ✅ реальный (local + azure) |
| 4 | Azure ML Online Endpoint (serving) | `integrations/azure_ml_online_endpoint.py` | ✅ реальный |
| 5 | MLflow клиент | `integrations/mlflow_client.py` | ✅ реальный (NoOp/Local/Fake) |
| 6 | Model Registry | `integrations/model_registry.py` | ✅ реальный (Local/MLflow/AzureML) |
| 7 | Dataset Registry | `integrations/dataset_registry.py` | ✅ реальный (Local + Azure ML) |
| 8 | Workflow State Store | `integrations/workflow_state_store.py` | ✅ реальный (чисто локальный, JSON) |

Общий паттерн для всех "внешних" клиентов (1, 2, 3, 4, 6-частично):
**Factory-инъекция**. Реальный класс делает настоящие вызовы SDK; для
тестов вместо него подставляется `Fake*`-класс с идентичным интерфейсом —
поэтому тесты гоняют **тот же самый код**, который выполнился бы в
проде, просто без сети.

---

## 2. Azure ML SDK-клиент (`integrations/azure_ml_client.py`)

Это самый низкоуровневый слой — просто фабрика, которая создаёт
`azure.ai.ml.MLClient`. Сам по себе ничего не делает (не отправляет jobs,
не регистрирует модели) — это делают клиенты уровня выше (п. 3, 4, 6, п.
"Online Endpoint").

- **`DefaultAzureMLClientFactory`** — РЕАЛЬНЫЙ. Создаёт `MLClient` через
  `DefaultAzureCredential()`. Требует `az login` или service-principal
  переменные окружения (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`,
  `AZURE_CLIENT_SECRET`) и `pip install agentic-mlops-yolo[azure]`
  (пакеты `azure-ai-ml>=1.12`, `azure-identity>=1.15`).
- **`FakeAzureMLClientFactory` + `FakeMLClient`** — тестовый дублёр,
  ничего никуда не отправляет. `FakeMLClient` содержит:
  - `jobs` (`FakeJobsOperations`) — create_or_update/stream/get/download/cancel
  - `models` (`FakeModelsOperations`) — create_or_update (регистрация модели)
  - `online_endpoints` (`FakeOnlineEndpointsOperations`) — begin_create_or_update/get
  - `online_deployments` (`FakeOnlineDeploymentsOperations`) — begin_create_or_update
  - Все `begin_*`-методы возвращают `_FakePoller` (имитирует
    `azure.core.polling.LROPoller` — у него тоже есть `.result()`), поэтому
    тестовый код проходит **точно ту же ветку**, что и реальный SDK-вызов.

Конфиг: `contracts/azure_ml.py::AzureMLConfig` — один YAML-файл
(`configs/azure_ml.yaml`, копия `azure_ml.example.yaml`) на все 4
Azure-клиента (training/eval/registry/serving). Секции:
`subscription_id/resource_group/workspace_name/compute_name`,
`authentication`, `environment` (для train/eval jobs),
`job` (timeout, tags, output_name), `data` (input_mode, asset_uri),
`serving` (**новое**, для Online Endpoint — instance_type/count, auth_mode,
опциональный отдельный `environment` для serving).

---

## 3. Azure ML Training Runner (`tools/training_runner.py`)

Обучение YOLO. Три режима (`TrainingMode`):

- **`local_dry_run`** — обрабатывается прямо внутри `tools/yolo_trainer.py::YoloTrainer`
  (отдельного класса-раннера для этого режима нет), просто пишет
  `training_request.json`, не вызывает ultralytics вообще. Используется
  по умолчанию для `--dry-run`.
- **`local_train`** (`LocalYOLOTrainingRunner`) — РЕАЛЬНЫЙ, вызывает
  `ultralytics.YOLO(...).train(...)` локально. Требует `pip install
  ultralytics` (мягкая зависимость, соответствующий `ImportError` →
  понятный `RuntimeError`, если пакет не установлен).
- **`azure_train`** (`AzureMLTrainingRunner`) — РЕАЛЬНЫЙ. Собирает
  `azure.ai.ml.command(...)` (`CommandJob`), код —
  `azure_jobs/train_yolo.py` (самодостаточный скрипт, без импорта
  `agentic_mlops` — в Azure загружается только папка `azure_jobs/`).
  Отправляет job, стримит логи (`ml_client.jobs.stream`) или поллит статус
  (`_wait_for_completion`, таймаут из `AzureMLConfig.job.timeout_minutes`),
  скачивает `best.pt`/`last.pt`/`results.csv`.

**Стоимость реального теста**: 1 эпоха на coco8 на Standard_NC6 ≈
15-30 минут, ≈ $0.50-1.50.

---

## 4. Azure ML Evaluation Runner (`tools/evaluation_runner.py`)

Симметрично Training Runner'у:

- **`FakeEvaluationRunner`** — детерминированные фейковые метрики
  (mAP50=0.862), без вызовов ultralytics.
- **`LocalYOLOEvaluationRunner`** — РЕАЛЬНЫЙ, `.val()` локально.
- **`AzureMLEvaluationRunner`** — РЕАЛЬНЫЙ, отправляет CommandJob
  (`azure_jobs/eval_yolo.py`), скачивает `metrics.json` +
  confusion-matrix/PR-curve графики.

---

## 5. Azure ML Online Endpoint — serving (`integrations/azure_ml_online_endpoint.py`)

**Самый новый клиент** (добавлен в этой сессии). `AzureMLOnlineEndpointDeployer`
— РЕАЛЬНЫЙ, настоящие вызовы `MLClient.online_endpoints` /
`online_deployments`:

1. Создаёт/обновляет `ManagedOnlineEndpoint` (`auth_mode` из
   `AzureMLConfig.serving.auth_mode`)
2. Создаёт/обновляет `ManagedOnlineDeployment`, ссылаясь на уже
   зарегистрированный Azure ML Model asset (`azure_model_name` +
   `azure_model_version` — не сырой файл!). Скрипт для инференса:
   `azure_jobs/score.py` (функции `init()`/`run()`, тоже самодостаточный).
3. Переключает 100% трафика на новый deployment.

**Важные ограничения (не реализовано)**:
- Никакого blue/green или canary rollout — только одномоментное
  переключение 100% трафика.
- Модель ДОЛЖНА быть уже зарегистрирована как Azure ML Model asset
  (через `register-model --backend azure_ml`) — сырой `.pt`/`.onnx`
  файл сюда напрямую не передать.
- Нет отдельного serving-окружения по умолчанию — если не задать
  `AzureMLConfig.serving.environment`, используется то же окружение, что
  и для тренировки (может не содержать inference-server зависимостей —
  это ответственность того, кто настраивает `azure_ml.yaml`).

Вызывается из `DeploymentAgent`/`ModelDeployer`
(`tools/deployer.py`) при `DeploymentInput.backend="azure_ml"`; H6-гейт
(rollback_plan + одобренный production_approval_path) проверяется ДО
ветвления на backend — то есть работает одинаково для local и azure_ml.

---

## 6. MLflow клиент (`integrations/mlflow_client.py`)

Четыре реализации одного интерфейса `MLflowTrackingClientBase`:

- **`NoOpMLflowTrackingClient`** — ничего не делает, mlflow-пакет не
  нужен. Дефолт, когда трекинг выключен.
- **`LocalMLflowTrackingClient`** — РЕАЛЬНЫЙ, использует пакет `mlflow`.
  Работает и с SQLite (`sqlite:///outputs/mlruns.db`, рекомендуемый URI),
  и с `file:./mlruns` (автоматически выставляет
  `MLFLOW_ALLOW_FILE_STORE=true`, т.к. MLflow ≥3.14 по умолчанию
  отключает file-store), и с удалённым MLflow-сервером.
- **`FakeMLflowTrackingClient`** (алиас `FakeMLflowClient`) — in-memory
  дублёр, хранит все runs/params/metrics/tags/artifacts в
  `self.runs: dict`, ничего не пишет на диск.
- **`MLflowClient`** — **legacy-заглушка**, все методы кидают
  `NotImplementedError`. Оставлена только для обратной совместимости,
  для новых мест использования — не использовать.

**Кто реально включает MLflow-трекинг сейчас**:
- `MVPWorkflow` (для `run-mvp`) — один parent run на весь пайплайн.
- `OrchestratorWorkflow` (для `run-workflow`) — **добавлено в этой
  сессии** — тоже один parent run, но с важным отличием: `run_id`
  сохраняется в `state.json`, поэтому `--resume` после паузы/ошибки
  логирует в ТОТ ЖЕ run, а не создаёт новый. Run закрывается
  (`FINISHED`/`FAILED`) только на терминальном статусе — пауза
  `PENDING_APPROVAL` оставляет run открытым.
- Каждый отдельный агент (все 11 в оркестраторе + все в `run-mvp`)
  умеет логировать в родительский run, если ему передали
  `mlflow_client=`/`mlflow_run_id=` — но самостоятельно run не создаёт.

Включение: `--enable-mlflow --mlflow-config configs/mlflow.yaml` (флаги
есть у `run-mvp`, `run-workflow`, и у большинства отдельных команд типа
`train`/`evaluate`).

---

## 7. Model Registry (`integrations/model_registry.py`)

Три реальных бэкенда + один тестовый, все реализуют
`ModelRegistryClientBase.register()`:

- **`LocalModelRegistryClient`** — РЕАЛЬНЫЙ, файловая система.
  `<registry_dir>/<model>/versions/<N>/model/best.pt` + `lineage.json` +
  `model_card.md`. SHA-256 проверка после копирования. `latest.json`
  обновляется только после успеха. Частично записанная версия удаляется
  при ошибке.
- **`MLflowModelRegistryClient`** — РЕАЛЬНЫЙ. Логирует `best.pt` как
  артефакт (на существующий parent run, если он есть, иначе создаёт
  короткоживущий run), затем `create_registered_model()` +
  `create_model_version()` (специально НЕ `mlflow.register_model()` —
  тот в MLflow ≥3.x требует "Logged Model"-сущность, которую простой
  `log_artifact()` не создаёт).
- **`AzureMLModelRegistryClient`** — РЕАЛЬНЫЙ. Регистрирует `best.pt` как
  Azure ML Model asset через `MLClient.models.create_or_update()`.
  **Не может быть создан автоматически** — `create_registry_client()`
  кидает `ValueError` для `AZURE_ML` (нет данных о подписке/группе/
  workspace в `ModelRegistrationInput`), CLI строит клиент вручную и
  инжектит его (`register-model --backend azure_ml --azure-config ...`).
- **`FakeModelRegistryClient`** — только для тестов, ничего не пишет.

---

## 8. Dataset Registry (`integrations/dataset_registry.py`)

- **`LocalDatasetVersionRegistry`** — РЕАЛЬНЫЙ, файловая система,
  зеркалит `LocalModelRegistryClient`. Дедупликация по контент-хэшу
  (SHA-256 над отсортированными парами `(relative_path, file_sha256)`)
  — если хэш совпадает с существующей версией, новая копия НЕ
  создаётся, возвращается существующая версия (`status=deduplicated`).
- **`AzureMLDatasetRegistryClient`** — РЕАЛЬНЫЙ (добавлено 2026-07-13).
  Настоящий вызов `MLClient.data.create_or_update()`, регистрирует
  директорию датасета как Azure ML Data asset (`AssetTypes.URI_FOLDER`).
  Как и `AzureMLModelRegistryClient`, не может быть создан
  автоматически — `create_dataset_registry_client()` кидает `ValueError`
  для `AZURE_ML` (нужен `AzureMLConfig` извне), CLI строит клиент вручную
  (`version-dataset --backend azure_ml --azure-config ...`). **Важно: НЕТ
  локальной дедупликации по хэшу для этого бэкенда** — Azure ML сам
  владеет версионированием ассета, повторная регистрация одинакового
  контента создаст новую Azure-версию (в отличие от local backend).
  `OrchestratorWorkflow`'s `dataset_versioning`-шаг поддерживает
  `dataset_registry_backend="azure_ml"` так же, как `model_registry`/
  `deployment`.
- **`FakeDatasetVersionRegistry`** — тестовый дублёр.

---

## 9. Workflow State Store (`integrations/workflow_state_store.py`)

Не совсем "клиент" внешней системы — просто два локальных файла на
`workflow_id`:
```
runs/<workflow_id>/state.json        — снапшот состояния (перезаписывается)
runs/<workflow_id>/audit_log.jsonl   — журнал событий (только дозапись)
```
Используется исключительно `OrchestratorWorkflow`. Никакого сетевого
State Store не существует — это осознанное решение по спеке
(`00_orchestrator_agent.md`: "Orchestrator может быть простым Python
классом + JSON state file").

---

## 10. Что дальше нужно сделать (реальные пробелы)

По убыванию значимости:

1. **Azure Monitor / Application Insights для `MonitoringAgent`** —
   сейчас читает только локальный JSONL-файл логов предсказаний. Нет
   интеграции с живыми метриками endpoint'а. Если Online Endpoint (п.5)
   реально задеплоен, естественный источник логов для Monitoring — как
   раз Application Insights, к которому Managed Online Endpoint можно
   подключить в Azure, но это не реализовано ни на стороне сбора логов,
   ни на стороне их выгрузки в наш JSONL-формат.
2. **Notification client (Teams/Slack/Email) для `OrchestratorWorkflow`**
   — переходы состояний видны только через `audit_log.jsonl` и
   структурированные логи. Реального push-уведомления нет вообще.
3. **Docker image build / AKS / CI-CD trigger для `DeploymentAgent`** —
   реализован только `local` (локальный релиз) и `azure_ml` (Managed
   Online Endpoint) бэкенды. Классический контейнерный деплой (свой
   Docker-образ + собственный Kubernetes) не начат.
4. ~~Azure ML Data Asset для `DatasetVersioningAgent`~~ — **РЕАЛИЗОВАНО
   2026-07-13**, см. раздел 8 (`AzureMLDatasetRegistryClient`).
5. **VOC label format** для Dataset Structuring Agent — есть только
   `yolo`/`coco`.
6. **Многошаговый Azure ML Pipeline** — сейчас каждый Azure-шаг
   (train/eval) — это отдельный `CommandJob`, а не единый связанный
   `azure.ai.ml.dsl.pipeline`. Из-за этого нет нативного Azure-lineage
   между шагами (это компенсируется собственным `lineage.json`).
7. **`H4 Training Approval`, реализован, НО**: расположен между
   `dataset_versioning` и `training` — это моя (не из спеки буквально)
   трактовка; спека (`00_orchestrator_agent.md`) в примере таблицы
   переходов скорее намекает на approval "перед повторным обучением"
   (после `model_decision`, если тот рекомендует RETRAIN) — то есть
   концепция "human-in-the-loop retraining loop" пока не реализована
   вообще (нет цикла "закончили оценку → решили RETRAIN → снова прогнали
   training с новым workflow_id"). Сейчас каждый `workflow_id` — это
   строго линейный проход один раз.

### ~~Найденная попутно~~ — ИСПРАВЛЕНО (2026-07-13)

Изначально здесь было описано, что `pytest -m azure_integration
--azure-config configs/azure_ml.yaml` из `CLAUDE.md` — нерабочая команда
(маркер и CLI-опция нигде не зарегистрированы). Исправлено:

- `pyproject.toml` → `[tool.pytest.ini_options].markers` теперь регистрирует
  `azure_integration`.
- `tests/conftest.py::pytest_addoption` регистрирует `--azure-config`.
- `tests/integration/test_azure_ml_live.py` — новый, реальный, opt-in тест.

**Важно — объём теста сознательно урезан**: это НЕ полноценный прогон
реального training job (как изначально буквально предполагала
формулировка в `CLAUDE.md`/`README.md`), а лёгкая **connectivity-проверка**
— `ml_client.workspaces.get(...)` и `ml_client.compute.get(...)`
(read-only, без создания compute, практически бесплатно). Причины:
1. Я не могу выполнить/проверить реальный платный training job сам (нет
   доступа к Azure-подписке) — писать код, который заставит пользователя
   тратить реальные деньги, не проверив его лично, было бы безответственно.
2. Connectivity-проверка — более полезный ПЕРВЫЙ интеграционный тест на
   практике: прежде чем платить за training job, вы и так захотите
   убедиться, что `azure_ml.yaml` + `az login` вообще работают.

Формулировки в `CLAUDE.md`/`README.md` обновлены под реальное поведение.
Полноценный платный integration-тест (реальный `train --runner azure-ml`
job внутри pytest) — если он всё ещё нужен — отдельная будущая задача,
не сделана в рамках этого исправления.

Проверено вручную (см. раздел 11.3 ниже): без `--azure-config` тесты
корректно `SKIPPED` (и в `pytest tests/unit`, и в голом `pytest`); с
синтаксически верным, но ненастоящим `azure_ml.yaml` — тест доходит до
реального вызова `DefaultAzureCredential()` и падает с настоящей
`ClientAuthenticationError` (то есть код реально достигает Azure SDK, а
не тихо не работает).

---

## 11. Как тестировать то, что есть сейчас

### 11.1. Юнит-тесты (быстро, бесплатно, без интернета) — уже всё покрыто

```bash
pip install -e ".[dev]"
pytest tests/unit -v          # все 489 тестов
pytest tests/unit -q          # коротко
```

Тесты по каждому клиенту:

| Клиент | Файл теста |
|---|---|
| Azure ML SDK-клиент / Training Runner | `tests/unit/test_azure_training_runner.py`, `tests/unit/test_training_runner.py` |
| Azure ML Evaluation Runner | `tests/unit/test_evaluation_runner.py` |
| Azure ML Online Endpoint | `tests/unit/test_azure_ml_online_endpoint.py` (9 тестов) |
| MLflow клиент | `tests/unit/test_mlflow_integration.py` (56 тестов, включая `TestMVPWorkflowMLflow` и `TestOrchestratorWorkflowMLflow`) |
| Model Registry (все 3 бэкенда) | `tests/unit/test_model_registry.py` |
| Dataset Registry | `tests/unit/test_dataset_versioning.py` |
| Workflow State Store / Orchestrator | `tests/unit/test_orchestrator.py` (33 теста) |
| Deployment (local + azure_ml) | `tests/unit/test_deployment.py` (32 теста) |

Все они используют **Fake-клиенты** (`FakeAzureMLClientFactory`,
`FakeMLflowClient`, `FakeModelRegistryClient`, `FakeDatasetVersionRegistry`)
— реальные учётные данные, интернет, Azure-подписка НЕ нужны. Но
выполняется настоящий "бизнес-код" (сборка `CommandJob`, `Model`,
`ManagedOnlineEndpoint` и т.д.) — просто финальный сетевой вызов
подменяется.

### 11.2. Проверка вручную одного клиента (без pytest)

Пример — Azure ML Online Endpoint с фейковым клиентом (безопасно, ничего
реального не создаёт):

```python
from pathlib import Path
from agentic_mlops.contracts.azure_ml import AzureMLConfig, AzureMLEnvironmentConfig
from agentic_mlops.contracts.deployment import DeploymentInput, DeploymentBackend
from agentic_mlops.integrations.azure_ml_client import FakeAzureMLClientFactory
from agentic_mlops.integrations.azure_ml_online_endpoint import AzureMLOnlineEndpointDeployer

cfg = AzureMLConfig(
    subscription_id="sub", resource_group="rg", workspace_name="ws", compute_name="cpu",
    environment=AzureMLEnvironmentConfig(registered_environment="azureml:env:1"),
)
deployer = AzureMLOnlineEndpointDeployer(cfg, client_factory=FakeAzureMLClientFactory())
out = deployer.deploy(
    DeploymentInput(model_name="m", backend=DeploymentBackend.AZURE_ML,
                     azure_model_name="m-model", azure_model_version=1),
    Path("/tmp/out"),
)
print(out.success, out.scoring_uri)
```

Так же можно проверить `FakeMLflowClient`, `FakeModelRegistryClient` и
т.д. — все они лежат в `integrations/*.py` рядом с реальными классами.

### 11.3. Реальный тест против живого Azure ML

**a) Connectivity-проверка (бесплатно, автоматизирована через pytest)**

```bash
pip install -e ".[dev,azure]"
az login   # или AZURE_CLIENT_ID/AZURE_TENANT_ID/AZURE_CLIENT_SECRET
cp configs/azure_ml.example.yaml configs/azure_ml.yaml
# заполнить subscription_id/resource_group/workspace_name/compute_name

pytest tests/integration -m azure_integration --azure-config configs/azure_ml.yaml
```
Без `--azure-config` эти же тесты просто `SKIPPED` — безопасно гонять
`pytest`/`pytest tests/unit` без опасений случайно задеть Azure.

**b) Полный платный прогон (стоит денег, только вручную через CLI — pytest-теста для этого нет и сознательно не сделано, см. раздел "Найденная попутно")**

```bash
# Тренировка (реальный CommandJob, ~15-30 мин, ~$0.5-1.5 на 1 эпоху coco8)
agentic-mlops train --dataset-path ... --data-yaml ... \
  --training-config configs/training.yaml --runner azure-ml \
  --azure-config configs/azure_ml.yaml --output-dir ./runs/azure_run_001

# Регистрация как Azure ML Model asset
agentic-mlops register-model --model-name my-model \
  --training-output ./runs/azure_run_001/training_output.json \
  --evaluation-output ... --approval-decision ... \
  --backend azure_ml --azure-config configs/azure_ml.yaml

# Реальный деплой на Managed Online Endpoint
agentic-mlops deploy-model --model-name my-model --backend azure_ml \
  --azure-config configs/azure_ml.yaml \
  --azure-model-name my-model --azure-model-version 1
```

⚠️ Managed Online Endpoint **не бесплатен** — за выделенный instance
(`Standard_DS2_v2` по умолчанию) списывается плата, пока endpoint не
удалён вручную через `az ml online-endpoint delete` или портал. В коде
нет автоудаления/TTL.

Готового `pytest`-теста для этого пути нет (см. пункт про
`azure_integration` в разделе 10) — только ручной прогон через CLI.

### 11.4. Реальный тест MLflow (бесплатно, только диск)

```bash
pip install -e ".[dev,mlflow]"
cp configs/mlflow.example.yaml configs/mlflow.yaml   # tracking_uri: sqlite:///outputs/mlruns.db

agentic-mlops run-workflow --workflow-id wf_test \
  --config configs/orchestrator.yaml --runs-dir runs \
  --enable-mlflow --mlflow-config configs/mlflow.yaml

# посмотреть результат
mlflow ui --backend-store-uri sqlite:///outputs/mlruns.db
```
