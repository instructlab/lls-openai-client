# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import AsyncMock, patch
import os

# Third Party
from llama_stack.apis.inference import CompletionResponse
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack.providers.remote.inference.vllm.config import (
    VLLMInferenceAdapterConfig,
)
from llama_stack.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter
from openai.types.model import Model as OpenAIModel
import pytest
import pytest_asyncio

# First Party
from lls_openai_client.client_adapter import OpenAIClientAdapter

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
tool_groups:
- toolgroup_id: builtin::websearch
  provider_id: tavily-search
- toolgroup_id: builtin::rag
  provider_id: rag-runtime
- toolgroup_id: builtin::code_interpreter
  provider_id: code-interpreter
- toolgroup_id: builtin::wolfram_alpha
  provider_id: wolfram-alpha
server:
  port: 8321
"""


@pytest_asyncio.fixture
async def vllm_inference_adapter():
    config = VLLMInferenceAdapterConfig(url="http://mocked.localhost:12345")
    inference_adapter = VLLMInferenceAdapter(config)
    inference_adapter.model_store = AsyncMock()
    await inference_adapter.initialize()
    return inference_adapter


@pytest.fixture
def model_id():
    return "foo"


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


@pytest.fixture
def mock_openai_models_list():
    with patch(
        "openai.resources.models.AsyncModels.list", new_callable=AsyncMock
    ) as mock_list:
        yield mock_list


@pytest.fixture
def mock_completion_response():
    with patch(
        "llama_stack.providers.remote.inference.vllm.vllm.process_completion_response"
    ) as mock_response:
        mock_response.return_value = CompletionResponse(
            stop_reason="end_of_message",
            content="mock_response",
        )
        yield mock_response


@pytest.fixture
def mock_openai_completion(mock_completion_response):
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_completion:
        yield mock_completion


def test_batch_prompts(client, model_id, mock_openai_completion):
    args = []
    kwargs = {
        "model": model_id,
        "prompt": ["test1", "test2"],
        "max_tokens": 1,
        "n": 3,
    }
    client.completions.create(*args, **kwargs)
    mock_openai_completion.assert_called()
    call_args = mock_openai_completion.call_args_list
    assert call_args[0].kwargs["model"] == model_id
    assert call_args[0].kwargs["max_tokens"] == 1
    # 2 prompts * n of 3 == 6 expected completions
    assert len(call_args) == 6

@pytest.mark.xfail(
    reason="bug in remote::vllm provider apply Llama ChatFormat to completions",
    strict=True,
)
def test_completion_prompt_unchanged(client, model_id, mock_openai_completion):
    prompt = "expected prompt"
    args = []
    kwargs = {
        "model": model_id,
        "prompt": prompt,
    }
    client.completions.create(*args, **kwargs)
    mock_openai_completion.assert_called()
    call_kwargs = mock_openai_completion.call_args.kwargs
    assert call_kwargs["model"] == model_id
    assert call_kwargs["prompt"] == prompt


def test_guided_decoding_valid_json_response(client, model_id, mock_completion_response):
    args = []
    kwargs = {
        "model": model_id,
        "prompt": "foo bar",
        "extra_body": {
            "guided_choice": ["joy", "sadness"]
        },
    }
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_openai_completion:
        mock_completion_response.return_value = CompletionResponse(
            stop_reason="end_of_message",
            content="[\"joy\"]",
        )
        response = client.completions.create(*args, **kwargs)
        mock_openai_completion.assert_called()
        call_kwargs = mock_openai_completion.call_args.kwargs
        assert call_kwargs["extra_body"]
        assert response.choices
        assert response.choices[0].text == "joy"


@pytest.mark.xfail(
    reason="figure out how to round-trip guided choice directly instead of via json schema",
    strict=True,
)
def test_guided_decoding_uses_guided_choice(client, model_id, mock_completion_response):
    guided_choices = ["joy", "sadness"]
    args = []
    kwargs = {
        "model": model_id,
        "prompt": "foo bar",
        "extra_body": {
            "guided_choice": guided_choices
        },
    }
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_openai_completion:
        mock_completion_response.return_value = CompletionResponse(
            stop_reason="end_of_message",
            content="joy",
        )
        response = client.completions.create(*args, **kwargs)
        mock_openai_completion.assert_called()
        call_kwargs = mock_openai_completion.call_args.kwargs
        assert call_kwargs["extra_body"]
        assert call_kwargs["extra_body"]["guided_choice"]
        assert call_kwargs["extra_body"]["guided_choice"] == guided_choices
        assert response.choices
        assert response.choices[0].text == "joy"


def test_guided_decoding_invalid_json_response(client, model_id, mock_completion_response):
    args = []
    kwargs = {
        "model": model_id,
        "prompt": "foo bar",
        "extra_body": {
            "guided_choice": ["joy", "sadness"]
        },
    }
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new_callable=AsyncMock
    ) as mock_openai_completion:
        mock_completion_response.return_value = CompletionResponse(
            stop_reason="end_of_message",
            content="something that is not valid json",
        )
        response = client.completions.create(*args, **kwargs)
        mock_openai_completion.assert_called()
        call_kwargs = mock_openai_completion.call_args.kwargs
        assert call_kwargs["extra_body"]
        assert response.choices
        assert response.choices[0].text == "something that is not valid json"
