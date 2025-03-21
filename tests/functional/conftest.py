# SPDX-License-Identifier: Apache-2.0

# pylint: disable=redefined-outer-name

# Standard
from unittest.mock import AsyncMock, patch

# Third Party
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from openai.types.model import Model as OpenAIModel
import pytest

# First Party
# pylint: disable=import-error
from lls_openai_client.client_adapter import OpenAIClientAdapter

#
# Common fixtures and testing utilities
#

RUN_YAML = """version: '2'
image_name: remote-vllm
apis:
- inference
- telemetry
providers:
  inference:
  - provider_id: vllm-inference
    provider_type: remote::vllm
    config:
      url: http://mocked.localhost:12345
      max_tokens: 4096
      api_token: fake
  telemetry:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      service_name: ${env.OTEL_SERVICE_NAME:llama-stack}
      sinks: ${env.TELEMETRY_SINKS:console,sqlite}
      sqlite_db_path: ${env.SQLITE_DB_PATH:~/.llama/distributions/remote-vllm/trace_store.db}
metadata_store:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:~/.llama/distributions/remote-vllm}/registry.db
models:
- metadata: {}
  model_id: foo
  provider_id: vllm-inference
  model_type: llm
shields: []
vector_dbs: []
datasets: []
scoring_fns: []
sdg_fns: []
benchmarks: []
tool_groups: []
server:
  port: 8321
"""


@pytest.fixture
def model_id():
    return "foo"


@pytest.fixture
def mock_openai_models_list():
    with patch(
        "openai.resources.models.AsyncModels.list", new_callable=AsyncMock
    ) as mock_list:
        yield mock_list


@pytest.fixture
def lls_client(tmp_path, mock_openai_models_list, model_id):
    run_yaml_path = str(tmp_path / "run.yaml")
    with open(run_yaml_path, "w", encoding="utf-8") as f:
        f.write(RUN_YAML)
    client = LlamaStackAsLibraryClient(run_yaml_path)

    async def mock_openai_models():
        yield OpenAIModel(id=model_id, created=1, object="model", owned_by="test")

    mock_openai_models_list.return_value = mock_openai_models()

    client.initialize()
    return client


@pytest.fixture
def client(lls_client):
    return OpenAIClientAdapter(lls_client)
